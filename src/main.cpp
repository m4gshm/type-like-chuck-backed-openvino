#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/streamer_base.hpp>

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

void emit(const json& event) {
    if (event.value("type", "") == kEventError) {
        std::cerr << "[runner] " << event.dump() << '\n';
    }
    if (g_events != nullptr) {
        std::fputs((event.dump() + '\n').c_str(), g_events);
        std::fflush(g_events);
    }
}

class Runner {
  public:
    explicit Runner(std::unique_ptr<ov::genai::LLMPipeline> pipeline)
        : pipeline_(std::move(pipeline)) {}

    void run() {
        worker_ = std::thread([this] { processQueue(); });
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            json message;
            try {
                message = json::parse(line);
            } catch (const std::exception&) {
                emit(json{
                    {"type", kEventError}, {"id", nullptr}, {"message", "unparsable input line"}});
                continue;
            }
            const std::string type = message.value("type", "");
            if (type == kRequestGenerate) {
                enqueue(runner::parseRequest(message));
            } else if (type == kRequestCancel) {
                cancelRequest(message);
            } else if (type == kRequestShutdown) {
                shutdown();
                break;
            }
        }
        shutdown();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

  private:
    void enqueue(const runner::GenerateRequest& request) {
        std::cerr << "[runner] generate id=" << request.id
                  << " promptChars=" << request.prompt.size() << " maxTokens=" << request.maxTokens
                  << " stops=" << request.stop.size() << '\n';
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(request);
        }
        cv_.notify_one();
    }

    void cancelRequest(const json& message) {
        if (!message.contains("id") || !message["id"].is_number_integer()) {
            cancelled_.store(true);
            return;
        }
        const long long id = message["id"].get<long long>();
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_ && id == activeId_) {
                cancelled_.store(true);
                return;
            }
            const auto it =
                std::find_if(queue_.begin(), queue_.end(),
                             [id](const runner::GenerateRequest& r) { return r.id == id; });
            if (it != queue_.end()) {
                queue_.erase(it);
                removed = true;
            }
        }
        if (removed) {
            emit(json{
                {"type", kEventDone}, {"id", id}, {"finishReason", runner::finishReason(true, 0)}});
        }
    }

    void shutdown() {
        std::vector<long long> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
            std::transform(queue_.begin(), queue_.end(), std::back_inserter(dropped),
                           [](const runner::GenerateRequest& r) { return r.id; });
            queue_.clear();
            cancelled_.store(true);
        }
        for (const long long id : dropped) {
            emit(json{
                {"type", kEventDone}, {"id", id}, {"finishReason", runner::finishReason(true, 0)}});
        }
        cv_.notify_all();
    }

    void processQueue() {
        for (;;) {
            runner::GenerateRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
                if (queue_.empty()) {
                    return;
                }
                request = std::move(queue_.front());
                queue_.pop_front();
                activeId_ = request.id;
                active_ = true;
                cancelled_.store(false);
            }
            generate(request);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_ = false;
            }
        }
    }

    void generate(const runner::GenerateRequest& request) {
        ov::genai::GenerationConfig config;
        config.max_new_tokens = request.maxTokens;
        config.stop_strings = std::set<std::string>(request.stop.begin(), request.stop.end());
        config.apply_chat_template = false;
        std::size_t tokens = 0;
        try {
            const ov::genai::DecodedResults results = pipeline_->generate(
                request.prompt, config, [this, &tokens](const std::string& token) {
                    if (cancelled_.load()) {
                        return ov::genai::StreamingStatus::CANCEL;
                    }
                    tokens++;
                    emit(json{{"type", kEventToken}, {"text", token}});
                    return ov::genai::StreamingStatus::RUNNING;
                });
            const int nativeReason = results.finish_reasons.empty()
                                         ? 0
                                         : static_cast<int>(results.finish_reasons.front());
            const std::string reason = runner::finishReason(cancelled_.load(), nativeReason);
            std::cerr << "[runner] generate id=" << request.id << " tokens=" << tokens
                      << " finish=" << reason << '\n';
            emit(json{{"type", kEventDone}, {"id", request.id}, {"finishReason", reason}});
        } catch (const std::exception& e) {
            std::cerr << "[runner] generate id=" << request.id << " tokens=" << tokens
                      << " finish=error\n";
            emit(json{{"type", kEventError},
                      {"id", request.id},
                      {"message", std::string("generate failed: ") + e.what()}});
        } catch (...) {
            std::cerr << "[runner] generate id=" << request.id << " tokens=" << tokens
                      << " finish=error\n";
            emit(json{{"type", kEventError},
                      {"id", request.id},
                      {"message", "generate failed: unknown exception"}});
        }
    }

    std::unique_ptr<ov::genai::LLMPipeline> pipeline_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<runner::GenerateRequest> queue_;
    std::atomic<bool> cancelled_{false};
    bool shutdown_ = false;
    long long activeId_ = 0;
    bool active_ = false;
};

} // namespace

int main(int argc, char** argv) {
    // Must run before any OpenVINO code can log (see openEventStream).
    g_events = openEventStream();
    const runner::Options options = runner::parseArgs(argc, argv);
    if (options.printVersion) {
        // Strips the marker prefix: the value doubles as the byte-scan anchor.
        emit(json{
            {"type", kEventVersion},
            {"app", TLC_OVGENAI_APP_VERSION},
            {"ovGenai",
             std::string(kRuntimeVersionMarker).substr(sizeof("ovgenai-runtime-version: ") - 1)}});
        return 0;
    }
    if (options.modelPath.empty()) {
        emit(json{{"type", kEventError}, {"id", nullptr}, {"message", "no --model given"}});
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
        for (const auto& [key, raw] :
             runner::effectiveProperties(options.properties, options.contextLength,
                                         options.device.find("NPU") != std::string::npos)) {
            props.emplace(key, typed(raw));
        }
        pipeline = std::make_unique<ov::genai::LLMPipeline>(options.modelPath, tokenizer,
                                                            options.device, props);
    } catch (const std::exception& e) {
        emit(json{{"type", kEventError},
                  {"id", nullptr},
                  {"message", std::string("load failed: ") + e.what()}});
        return 1;
    }

    emit(json{{"type", kEventReady},
              {"model", options.modelPath},
              {"device", options.device},
              {"ctx", options.contextLength}});

    Runner runner(std::move(pipeline));
    runner.run();
    return 0;
}
