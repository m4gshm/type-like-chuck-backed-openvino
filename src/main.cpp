#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/streamer_base.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>

#include "runner.hpp"

namespace {

using json = nlohmann::json;

// Byte-scan marker: the extension reads the runner binary for this prefix
// (without loading runtime DLLs) to learn which OpenVINO GenAI runtime
// version to download — a mismatched runtime crashes the process at load.
constexpr char kRuntimeVersionMarker[] = "ovgenai-runtime-version: " TLC_OVGENAI_OV_VERSION;

// Protocol type names (contract: openspec/specs/openvino-genai-runner/spec.md).
constexpr char kEventVersion[] = "version";
constexpr char kEventReady[] = "ready";
constexpr char kEventToken[] = "token";
constexpr char kEventDone[] = "done";
constexpr char kEventError[] = "error";
constexpr char kRequestGenerate[] = "generate";
constexpr char kRequestCancel[] = "cancel";
constexpr char kRequestShutdown[] = "shutdown";

// A stalled or half-dead client is cut off after this instead of freezing
// the generation worker; the client counts as disconnected.
constexpr unsigned kWriteTimeoutMs = 5000;
constexpr unsigned kPollIntervalMs = 200;

// OpenVINO's NPU compiler logs to the raw stdout descriptor (bypassing C++
// streams) and child processes inherit the stdout handle; neither can be
// caught by a std::cout rdbuf swap. openEventStream() re-points descriptor 1
// at stderr so every writer lands there, and protocol events go through the
// saved real stdout descriptor instead.
FILE* openEventStream() {
#ifdef _WIN32
    const int saved = _dup(1);
#else
    const int saved = dup(1);
#endif
    if (saved < 0) {
        return stdout;
    }
#ifdef _WIN32
    _dup2(2, 1);
    FILE* stream = _fdopen(saved, "w");
#else
    dup2(2, 1);
    FILE* stream = fdopen(saved, "w");
#endif
    // OpenVINO logs write through the stdout FILE; keep it unbuffered so
    // compile progress streams to stderr instead of sitting in the pipe buffer.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    return stream != nullptr ? stream : stdout;
}

// Set once at the top of main(), before any OpenVINO code can log.
FILE* g_events = nullptr;

void logError(const json& event) {
    // Error events are part of the protocol (stdout), mirrored to stderr.
    if (g_events != nullptr) {
        std::fputs((event.dump() + '\n').c_str(), g_events);
        std::fflush(g_events);
    }
    std::cerr << "[runner] " << event.dump() << '\n';
}

class Sink {
public:
    virtual ~Sink() = default;
    // False = the client is gone (or stalled past the write timeout); the
    // caller must drop it. Serialized per client by the implementation.
    virtual bool writeLine(const std::string& line) = 0;
    virtual void close() = 0;
};

class StdioSink : public Sink {
public:
    bool writeLine(const std::string& line) override {
        if (g_events == nullptr) {
            return false;
        }
        std::fputs((line + '\n').c_str(), g_events);
        std::fflush(g_events);
        return true;
    }

    void close() override {}
};

#ifdef _WIN32

class PipeSink : public Sink {
public:
    explicit PipeSink(HANDLE handle) : handle_(handle) {}

    ~PipeSink() override { close(); }

    bool writeLine(const std::string& line) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (broken_ || handle_ == INVALID_HANDLE_VALUE) {
            return false;
        }
        const std::string data = line + '\n';
        OVERLAPPED ov{};
        ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (ov.hEvent == nullptr) {
            broken_ = true;
            return false;
        }
        if (!WriteFile(handle_, data.data(), static_cast<DWORD>(data.size()), nullptr, &ov) &&
            GetLastError() != ERROR_IO_PENDING) {
            broken_ = true;
            CloseHandle(ov.hEvent);
            return false;
        }
        DWORD written = 0;
        if (WaitForSingleObject(ov.hEvent, kWriteTimeoutMs) != WAIT_OBJECT_0 ||
            !GetOverlappedResult(handle_, &ov, &written, FALSE) || written != data.size()) {
            CancelIo(handle_);
            broken_ = true;
            CloseHandle(ov.hEvent);
            return false;
        }
        CloseHandle(ov.hEvent);
        return true;
    }

    void close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (handle_ != INVALID_HANDLE_VALUE) {
            CancelIo(handle_);
            FlushFileBuffers(handle_);
            DisconnectNamedPipe(handle_);
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

    HANDLE handle() const { return handle_; }

private:
    HANDLE handle_;
    std::mutex mutex_;
    bool broken_ = false;
};

std::wstring wide(const std::string& text) {
    return std::wstring(text.begin(), text.end());
}

// Exclusive ownership of an endpoint: a guard instance with FIRST_PIPE fails
// when another runner already owns the name (named pipes otherwise allow any
// number of server instances of the same name). Nobody ever connects to it.
HANDLE makeGuard(const std::string& endpoint) {
    return CreateNamedPipeW(wide(endpoint + ".guard").c_str(),
                            PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
                            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                            1, 1024, 1024, 0, nullptr);
}

HANDLE makeListener(const std::string& endpoint, bool first) {
    DWORD flags = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED;
    if (first) {
        // Second runner on the same name fails here → startup error, exit 1.
        flags |= FILE_FLAG_FIRST_PIPE_INSTANCE;
    }
    return CreateNamedPipeW(wide(endpoint).c_str(), flags,
                            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                            PIPE_UNLIMITED_INSTANCES, 1 << 20, 1 << 20, 0, nullptr);
}

#else

class UnixSink : public Sink {
public:
    explicit UnixSink(int fd) : fd_(fd) {
        const timeval tv{static_cast<long>(kWriteTimeoutMs / 1000),
                         static_cast<long>((kWriteTimeoutMs % 1000) * 1000)};
        setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    ~UnixSink() override { close(); }

    bool writeLine(const std::string& line) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fd_ < 0) {
            return false;
        }
        const std::string data = line + '\n';
        size_t sent = 0;
        while (sent < data.size()) {
            const ssize_t n = ::send(fd_, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
            if (n <= 0) {
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    void close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fd_ >= 0) {
            ::shutdown(fd_, SHUT_RDWR);
            ::close(fd_);
            fd_ = -1;
        }
    }

    int fd() const { return fd_; }

private:
    int fd_;
    std::mutex mutex_;
};

// A stale socket file after a previous crash is removed before bind; a bind
// failure with a live listener surfaces to main() as exit 1.
int makeListener(const std::string& endpoint) {
    ::unlink(endpoint.c_str());
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (endpoint.size() >= sizeof(addr.sun_path) ||
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", endpoint.c_str()) < 0 ||
        ::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(fd, 8) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

#endif

// Overlapped read loop; feeds complete NDJSON lines into onLine and calls
// onEof on disconnect, read failure, or quit. Waits on the connection event
// and the quit event together — a read timeout must never be read as a
// disconnect.
#ifdef _WIN32
void readClient(HANDLE handle, HANDLE quitEvent, const std::function<void(const std::string&)>& onLine,
                const std::function<bool()>& keepGoing, const std::function<void()>& onEof) {
    std::string buffer;
    char chunk[8192];
    for (;;) {
        if (!keepGoing()) {
            return;
        }
        OVERLAPPED ov{};
        ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (ov.hEvent == nullptr) {
            onEof();
            return;
        }
        if (!ReadFile(handle, chunk, sizeof(chunk), nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) {
            CloseHandle(ov.hEvent);
            onEof();
            return;
        }
        HANDLE waits[] = {ov.hEvent, quitEvent};
        if (WaitForMultipleObjects(2, waits, FALSE, INFINITE) != WAIT_OBJECT_0) {
            CancelIo(handle);
            CloseHandle(ov.hEvent);
            onEof();
            return;
        }
        DWORD read = 0;
        if (!GetOverlappedResult(handle, &ov, &read, FALSE) || read == 0) {
            CloseHandle(ov.hEvent);
            onEof();
            return;
        }
        CloseHandle(ov.hEvent);
        buffer.append(chunk, read);
        size_t at = 0;
        for (size_t nl; (nl = buffer.find('\n', at)) != std::string::npos; at = nl + 1) {
            onLine(buffer.substr(at, nl - at));
        }
        buffer.erase(0, at);
    }
}
#else
void readClient(int fd, const std::function<void(const std::string&)>& onLine,
                const std::function<bool()>& keepGoing, const std::function<void()>& onEof) {
    std::string buffer;
    char chunk[8192];
    for (;;) {
        if (!keepGoing()) {
            return;
        }
        pollfd pfd{fd, POLLIN, 0};
        if (::poll(&pfd, 1, kPollIntervalMs) <= 0) {
            continue;
        }
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            onEof();
            return;
        }
        buffer.append(chunk, static_cast<size_t>(n));
        size_t at = 0;
        for (size_t nl; (nl = buffer.find('\n', at)) != std::string::npos; at = nl + 1) {
            onLine(buffer.substr(at, nl - at));
        }
        buffer.erase(0, at);
    }
}
#endif

// The runner generates one request at a time; each queued job remembers the
// client it came from so tokens and done/error events go back only there.
// Cancel-by-id also matches only within the requesting client's jobs: ids
// are per-client, so two clients may send the same id.
class Runner {
public:
    explicit Runner(std::unique_ptr<ov::genai::LLMPipeline> pipeline) : pipeline_(std::move(pipeline)) {}

    // Called when a client sink stops accepting writes; listen mode uses it
    // to drop the stalled client (its refcount must not hold the process).
    void onSinkBroken(std::function<void(const std::shared_ptr<Sink>&)> callback) {
        sinkBroken_ = std::move(callback);
    }

    void runStdio() {
        const auto sink = std::make_shared<StdioSink>();
        worker_ = std::thread([this] { processQueue(); });
        std::string line;
        while (std::getline(std::cin, line)) {
            handleLine(line, sink);
            if (exit_.load()) {
                break;
            }
        }
        requestExit();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // Listen mode: the worker processes the queue; the caller's thread waits
    // until requestExit() (a shutdown request or the last client leaving).
    void runShared() {
        worker_ = std::thread([this] { processQueue(); });
        std::unique_lock<std::mutex> lock(exitMutex_);
        exitCv_.wait(lock, [this] { return exit_.load(); });
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void enqueue(const runner::GenerateRequest& request, const std::shared_ptr<Sink>& sink) {
        std::cerr << "[runner] generate id=" << request.id << " promptChars=" << request.prompt.size()
                  << " maxTokens=" << request.maxTokens << " stops=" << request.stop.size() << '\n';
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(Job{request, sink});
        }
        cv_.notify_one();
    }

    void cancelRequest(const json& message, const std::shared_ptr<Sink>& sink) {
        if (!message.contains("id") || !message["id"].is_number_integer()) {
            cancelled_.store(true);
            return;
        }
        const long long id = message["id"].get<long long>();
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_ && id == activeId_ && activeSink_ == sink) {
                cancelled_.store(true);
                return;
            }
            const auto it = std::find_if(queue_.begin(), queue_.end(),
                                         [&, id](const Job& job) { return job.request.id == id && job.sink == sink; });
            if (it != queue_.end()) {
                const std::shared_ptr<Sink> jobSink = it->sink;
                queue_.erase(it);
                removed = true;
                jobSink->writeLine(json{{"type", kEventDone}, {"id", id}, {"finishReason", runner::finishReason(true, 0)}}.dump());
            }
        }
        if (removed) {
            std::cerr << "[runner] cancel id=" << id << " (queued)\n";
        }
    }

    void requestExit() {
        std::vector<Job> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
            dropped.assign(queue_.begin(), queue_.end());
            queue_.clear();
            cancelled_.store(true);
        }
        for (const Job& job : dropped) {
            job.sink->writeLine(
                json{{"type", kEventDone}, {"id", job.request.id}, {"finishReason", runner::finishReason(true, 0)}}.dump());
        }
        {
            std::lock_guard<std::mutex> lock(exitMutex_);
            exit_.store(true);
        }
        exitCv_.notify_all();
        cv_.notify_all();
    }

private:
    struct Job {
        runner::GenerateRequest request;
        std::shared_ptr<Sink> sink;
    };

    void handleLine(const std::string& line, const std::shared_ptr<Sink>& sink) {
        if (line.empty()) {
            return;
        }
        json message;
        try {
            message = json::parse(line);
        } catch (const std::exception&) {
            sink->writeLine(json{{"type", kEventError}, {"id", nullptr}, {"message", "unparsable input line"}}.dump());
            return;
        }
        const std::string type = message.value("type", "");
        if (type == kRequestGenerate) {
            enqueue(runner::parseRequest(message), sink);
        } else if (type == kRequestCancel) {
            cancelRequest(message, sink);
        } else if (type == kRequestShutdown) {
            requestExit();
        }
    }

    void processQueue() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
                if (queue_.empty()) {
                    return;
                }
                job = std::move(queue_.front());
                queue_.pop_front();
                activeId_ = job.request.id;
                activeSink_ = job.sink;
                active_ = true;
                cancelled_.store(false);
            }
            generate(job);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_ = false;
            }
        }
    }

    void generate(const Job& job) {
        const runner::GenerateRequest& request = job.request;
        ov::genai::GenerationConfig config;
        config.max_new_tokens = request.maxTokens;
        config.stop_strings = std::set<std::string>(request.stop.begin(), request.stop.end());
        config.apply_chat_template = false;
        std::size_t tokens = 0;
        try {
            const ov::genai::DecodedResults results = pipeline_->generate(
                request.prompt,
                config,
                [this, &job, &request, &tokens](const std::string& token) {
                    if (cancelled_.load()) {
                        return ov::genai::StreamingStatus::CANCEL;
                    }
                    tokens++;
                    if (!job.sink->writeLine(json{{"type", kEventToken}, {"id", request.id}, {"text", token}}.dump())) {
                        // A client that stopped reading stops costing tokens.
                        cancelled_.store(true);
                        if (sinkBroken_) {
                            sinkBroken_(job.sink);
                        }
                        return ov::genai::StreamingStatus::CANCEL;
                    }
                    return ov::genai::StreamingStatus::RUNNING;
                });
            const int nativeReason = results.finish_reasons.empty() ? 0
                                                                    : static_cast<int>(results.finish_reasons.front());
            const std::string reason = runner::finishReason(cancelled_.load(), nativeReason);
            std::cerr << "[runner] generate id=" << request.id << " tokens=" << tokens << " finish=" << reason << '\n';
            job.sink->writeLine(json{{"type", kEventDone}, {"id", request.id}, {"finishReason", reason}}.dump());
        } catch (const std::exception& e) {
            std::cerr << "[runner] generate id=" << request.id << " tokens=" << tokens << " finish=error\n";
            job.sink->writeLine(
                json{{"type", kEventError}, {"id", request.id}, {"message", std::string("generate failed: ") + e.what()}}.dump());
        } catch (...) {
            std::cerr << "[runner] generate id=" << request.id << " tokens=" << tokens << " finish=error\n";
            job.sink->writeLine(json{{"type", kEventError}, {"id", request.id}, {"message", "generate failed: unknown exception"}}.dump());
        }
    }

    std::unique_ptr<ov::genai::LLMPipeline> pipeline_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    std::atomic<bool> cancelled_{false};
    bool shutdown_ = false;
    long long activeId_ = 0;
    std::shared_ptr<Sink> activeSink_;
    bool active_ = false;
    std::mutex exitMutex_;
    std::condition_variable exitCv_;
    std::atomic<bool> exit_{false};
    std::function<void(const std::shared_ptr<Sink>&)> sinkBroken_;
};

// Listen mode: accepts clients on the endpoint, routes each client's requests
// into the Runner with the client's sink, broadcasts the ready event, and
// self-exits when the last client disconnects.
class Server {
public:
    Server(std::string endpoint, Runner& runner)
        : endpoint_(std::move(endpoint)), runner_(runner) {}

    ~Server() { stop(); }

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // False when the endpoint is owned by another runner: the caller
    // reports a startup error and exits 1.
    bool start() {
#ifdef _WIN32
        guard_ = makeGuard(endpoint_);
        if (guard_ == INVALID_HANDLE_VALUE) {
            return false;
        }
        listener_ = makeListener(endpoint_, true);
        if (listener_ == INVALID_HANDLE_VALUE) {
            return false;
        }
        quitEvent_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (quitEvent_ == nullptr) {
            return false;
        }
#else
        listener_ = makeListener(endpoint_);
        if (listener_ < 0) {
            return false;
        }
#endif
        acceptor_ = std::thread([this] { acceptLoop(); });
        return true;
    }

    void setReady(const json& readyEvent) {
        const std::string line = readyEvent.dump();
        std::lock_guard<std::mutex> lock(mutex_);
        readyLine_ = line;
        for (const auto& client : clients_) {
            client->sink->writeLine(line);
        }
    }

    void dropBySink(const std::shared_ptr<Sink>& sink) {
        std::shared_ptr<Client> client;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = std::find_if(clients_.begin(), clients_.end(),
                                         [&](const std::shared_ptr<Client>& c) { return c->sink == sink; });
            if (it == clients_.end()) {
                return;
            }
            client = *it;
        }
        dropClient(client.get());
    }

    void stop() {
        quit_.store(true);
#ifdef _WIN32
        if (quitEvent_ != nullptr) {
            SetEvent(quitEvent_);
        }
#endif
        if (acceptor_.joinable()) {
            acceptor_.join();
        }
        // Readers exit via quit; join before destroying shared state.
        for (auto& reader : readers_) {
            if (reader.joinable()) {
                reader.join();
            }
        }
        readers_.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        clients_.clear();
    }

private:
    struct Client {
        std::shared_ptr<Sink> sink;
#ifdef _WIN32
        HANDLE handle = INVALID_HANDLE_VALUE;
#else
        int fd = -1;
#endif
    };

    void acceptLoop() {
        while (!quit_.load()) {
#ifdef _WIN32
            OVERLAPPED ov{};
            ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (ov.hEvent == nullptr) {
                break;
            }
            const BOOL connected = ConnectNamedPipe(listener_, &ov);
            const bool hasClient = connected || GetLastError() == ERROR_PIPE_CONNECTED;
            if (!hasClient && GetLastError() != ERROR_IO_PENDING) {
                CloseHandle(ov.hEvent);
                continue;
            }
            HANDLE waits[] = {ov.hEvent, quitEvent_};
            if (WaitForMultipleObjects(2, waits, FALSE, INFINITE) != WAIT_OBJECT_0) {
                CancelIo(listener_);
                CloseHandle(ov.hEvent);
                continue;
            }
            CloseHandle(ov.hEvent);
            if (quit_.load()) {
                DisconnectNamedPipe(listener_);
                CloseHandle(listener_);
                break;
            }
            const HANDLE handle = listener_;
            // A fresh listening instance right away, so the next client can
            // connect while this one is being served.
            listener_ = makeListener(endpoint_, false);
            registerClient(std::make_shared<PipeSink>(handle), handle);
#else
            pollfd pfd{listener_, POLLIN, 0};
            if (::poll(&pfd, 1, static_cast<int>(kPollIntervalMs)) <= 0) {
                continue;
            }
            const int fd = ::accept(listener_, nullptr, nullptr);
            if (fd < 0) {
                continue;
            }
            if (quit_.load()) {
                ::close(fd);
                break;
            }
            registerClient(std::make_shared<UnixSink>(fd), fd);
#endif
        }
    }

    void registerClient(std::shared_ptr<Sink> sink,
#ifdef _WIN32
                        HANDLE handle
#else
                        int fd
#endif
    ) {
        auto client = std::make_shared<Client>();
        client->sink = sink;
#ifdef _WIN32
        client->handle = handle;
#else
        client->fd = fd;
#endif
        {
            std::lock_guard<std::mutex> lock(mutex_);
            clients_.push_back(client);
            if (!readyLine_.empty()) {
                sink->writeLine(readyLine_);
            }
        }
        connected_.fetch_add(1);
        hadClient_.store(true);
        readers_.emplace_back([this, client] { readLoop(client); });
    }

    // Detached reader per client; on quit the Server::stop() clears the
    // client list and closes the sinks, ending these loops.
    void readLoop(const std::shared_ptr<Client>& client) {
#ifdef _WIN32
        readClient(
            client->handle,
            quitEvent_,
            [this, client](const std::string& line) { dispatch(line, client->sink); },
            [this] { return !quit_.load(); },
            [this, client] { dropClient(client.get()); });
#else
        readClient(
            client->fd,
            [this, client](const std::string& line) { dispatch(line, client->sink); },
            [this] { return !quit_.load(); },
            [this, client] { dropClient(client.get()); });
#endif
    }

    void dispatch(const std::string& line, const std::shared_ptr<Sink>& sink) {
        json message;
        try {
            message = json::parse(line);
        } catch (const std::exception&) {
            sink->writeLine(json{{"type", kEventError}, {"id", nullptr}, {"message", "unparsable input line"}}.dump());
            return;
        }
        const std::string type = message.value("type", "");
        if (type == kRequestGenerate) {
            runner_.enqueue(runner::parseRequest(message), sink);
        } else if (type == kRequestCancel) {
            runner_.cancelRequest(message, sink);
        } else if (type == kRequestShutdown) {
            runner_.requestExit();
        }
    }

    void dropClient(const Client* client) {
        std::shared_ptr<Sink> sink;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = std::find_if(clients_.begin(), clients_.end(),
                                         [client](const std::shared_ptr<Client>& c) { return c.get() == client; });
            if (it == clients_.end()) {
                return;
            }
            sink = (*it)->sink;
            clients_.erase(it);
        }
        sink->close();
        if (connected_.fetch_sub(1) == 1 && hadClient_.load()) {
            // Last client left: shut down and exit, no owner will kill us.
            runner_.requestExit();
        }
    }

    std::string endpoint_;
    Runner& runner_;
#ifdef _WIN32
    HANDLE listener_ = INVALID_HANDLE_VALUE;
    HANDLE guard_ = INVALID_HANDLE_VALUE;
    HANDLE quitEvent_ = nullptr;
#else
    int listener_ = -1;
#endif
    std::thread acceptor_;
    std::vector<std::thread> readers_;
    std::mutex mutex_;
    std::vector<std::shared_ptr<Client>> clients_;
    std::string readyLine_;
    std::atomic<bool> quit_{false};
    std::atomic<int> connected_{0};
    std::atomic<bool> hadClient_{false};
};

} // namespace

int main(int argc, char** argv) {
    // Must run before any OpenVINO code can log (see openEventStream).
    g_events = openEventStream();
    const runner::Options options = runner::parseArgs(argc, argv);
    if (options.printVersion) {
        // Strips the marker prefix: the value doubles as the byte-scan anchor.
        const json version{{"type", kEventVersion},
                           {"app", TLC_OVGENAI_APP_VERSION},
                           {"ovGenai", std::string(kRuntimeVersionMarker).substr(sizeof("ovgenai-runtime-version: ") - 1)},
                           {"pid", static_cast<long long>(
#ifdef _WIN32
                                       GetCurrentProcessId()
#else
                                       getpid()
#endif
                                           )}};
        std::fputs((version.dump() + "\n").c_str(), g_events);
        std::fflush(g_events);
        return 0;
    }
    if (options.modelPath.empty()) {
        logError(json{{"type", kEventError}, {"id", nullptr}, {"message", "no --model given"}});
        return 2;
    }
    const bool listenMode = !options.listenEndpoint.empty();
    if (listenMode && !runner::validListenEndpoint(options.listenEndpoint,
#ifdef _WIN32
                                                   true
#else
                                                   false
#endif
                                                   )) {
        logError(json{{"type", kEventError},
                      {"id", nullptr},
                      {"message", "invalid --listen endpoint"
#ifdef _WIN32
                                  " (expected \\\\.\\pipe\\NAME)"
#endif
                      }});
        return 2;
    }

    std::unique_ptr<ov::genai::LLMPipeline> pipeline;
    try {
        ov::genai::Tokenizer tokenizer(options.modelPath);
        auto typed = [](const std::string& raw) -> ov::Any {
            if (raw == "true" || raw == "false") {
                return raw == "true";
            }
            try {
                size_t pos = 0;
                long long n = std::stoll(raw, &pos);
                if (pos == raw.size()) {
                    return n;
                }
            } catch (const std::exception&) {
            }
            return raw;
        };
        ov::AnyMap props;
        for (const auto& [key, raw] : runner::effectiveProperties(
                 options.properties, options.contextLength,
                 options.device.find("NPU") != std::string::npos)) {
            props.emplace(key, typed(raw));
        }
        pipeline = std::make_unique<ov::genai::LLMPipeline>(options.modelPath, tokenizer, options.device, props);
    } catch (const std::exception& e) {
        logError(json{{"type", kEventError}, {"id", nullptr}, {"message", std::string("load failed: ") + e.what()}});
        return 1;
    }

    // Diagnostic only; not every device/plugin exposes its full name.
    std::string deviceName;
    try {
        ov::Core core;
        deviceName = core.get_property(options.device, ov::device::full_name);
    } catch (const std::exception&) {
    }

    const json ready{{"type", kEventReady},
                     {"model", options.modelPath},
                     {"device", options.device},
                     {"ctx", options.contextLength},
                     {"deviceName", deviceName},
                     {"pid", static_cast<long long>(
#ifdef _WIN32
                                 GetCurrentProcessId()
#else
                                 getpid()
#endif
                                     )}};

    Runner runner(std::move(pipeline));
    if (!listenMode) {
        std::fputs((ready.dump() + "\n").c_str(), g_events);
        std::fflush(g_events);
        runner.runStdio();
        return 0;
    }

    Server server(options.listenEndpoint, runner);
    runner.onSinkBroken([&server](const std::shared_ptr<Sink>& sink) { server.dropBySink(sink); });
    if (!server.start()) {
        logError(json{{"type", kEventError},
                      {"id", nullptr},
                      {"message", "listen endpoint already in use: " + options.listenEndpoint}});
        return 1;
    }
    server.setReady(ready);
    runner.runShared();
    server.stop();
    // OpenVINO's NPU plugin keeps background threads that race static
    // teardown and fail the process (0xC0000409) after all work is done;
    // every protocol event is already flushed, so skip the destructors.
    std::_Exit(0);
}
