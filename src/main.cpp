#include <atomic>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>

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

// std::cout's buffer is swapped to stderr in main() so OpenVINO logs (which
// share std::cout) cannot pollute the protocol stream; events keep using the
// original stdout buffer captured here (static init runs before main).
std::streambuf* const kEventBuf = std::cout.rdbuf();

std::ostream& eventOut() {
    static std::ostream stream{kEventBuf};
    return stream;
}

void emit(const json& event) {
    if (event.value("type", "") == kEventError) {
        std::cerr << "[runner] " << event.dump() << '\n';
    }
    eventOut() << event.dump() << '\n';
    eventOut().flush();
}

struct GenerateRequest {
    long long id = 0;
    std::string prompt;
    std::size_t maxTokens = 128;
    std::vector<std::string> stop;
};

class Runner {
public:
    explicit Runner(std::unique_ptr<ov::genai::LLMPipeline> pipeline) : pipeline_(std::move(pipeline)) {}

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            json message;
            try {
                message = json::parse(line);
            } catch (const std::exception&) {
                emit(json{{"type", kEventError}, {"id", nullptr}, {"message", "unparsable input line"}});
                continue;
            }
            const std::string type = message.value("type", "");
            if (type == kRequestGenerate) {
                handleGenerate(message);
            } else if (type == kRequestCancel) {
                cancelled_.store(true);
            } else if (type == kRequestShutdown) {
                break;
            }
        }
        finishGeneration();
    }

private:
    void handleGenerate(const json& message) {
        const runner::GenerateRequest request = runner::parseRequest(message);

        if (busy_.exchange(true)) {
            emit(json{{"type", kEventError}, {"id", request.id}, {"message", "busy"}});
            return;
        }
        cancelled_.store(false);
        if (worker_.joinable()) {
            worker_.join();
        }
        startGeneration(request);
    }

    void startGeneration(const runner::GenerateRequest& request) {
        worker_ = std::thread([this, request] {
            ov::genai::GenerationConfig config;
            config.max_new_tokens = request.maxTokens;
            config.stop_strings = std::set<std::string>(request.stop.begin(), request.stop.end());
            config.apply_chat_template = false;
            try {
                const ov::genai::DecodedResults results = pipeline_->generate(
                    request.prompt,
                    config,
                    [this](const std::string& token) {
                        if (cancelled_.load()) {
                            return ov::genai::StreamingStatus::CANCEL;
                        }
                        emit(json{{"type", kEventToken}, {"text", token}});
                        return ov::genai::StreamingStatus::RUNNING;
                    });
                const int nativeReason = results.finish_reasons.empty() ? 0
                    : static_cast<int>(results.finish_reasons.front());
                const std::string reason = runner::finishReason(cancelled_.load(), nativeReason);
                emit(json{{"type", kEventDone}, {"id", request.id}, {"finishReason", reason}});
            } catch (const std::exception& e) {
                emit(json{{"type", kEventError}, {"id", request.id}, {"message", std::string("generate failed: ") + e.what()}});
            } catch (...) {
                emit(json{{"type", kEventError}, {"id", request.id}, {"message", "generate failed: unknown exception"}});
            }
            busy_.store(false);
        });
    }

    void finishGeneration() {
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::unique_ptr<ov::genai::LLMPipeline> pipeline_;
    std::thread worker_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> cancelled_{false};
};

}  // namespace

int main(int argc, char** argv) {
    // OpenVINO logs (e.g. colored [INFO] lines during NPU compilation) go to
    // stdout, but the protocol reserves stdout for NDJSON events only.
    std::cout.rdbuf(std::cerr.rdbuf());
    const runner::Options options = runner::parseArgs(argc, argv);
    if (options.printVersion) {
        // Strips the marker prefix: the value doubles as the byte-scan anchor.
        emit(json{{"type", kEventVersion},
                  {"app", TLC_OVGENAI_APP_VERSION},
                  {"ovGenai", std::string(kRuntimeVersionMarker).substr(sizeof("ovgenai-runtime-version: ") - 1)}});
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
        for (const auto& [key, raw] : runner::effectiveProperties(
                 options.properties, options.contextLength,
                 options.device.find("NPU") != std::string::npos)) {
            props.emplace(key, typed(raw));
        }
        pipeline = std::make_unique<ov::genai::LLMPipeline>(options.modelPath, tokenizer, options.device, props);
    } catch (const std::exception& e) {
        emit(json{{"type", kEventError}, {"id", nullptr}, {"message", std::string("load failed: ") + e.what()}});
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
