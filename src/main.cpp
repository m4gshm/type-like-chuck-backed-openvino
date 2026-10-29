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

void emit(const json& event) {
    std::cout << event.dump() << '\n';
    std::cout.flush();
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
                emit(json{{"type", "error"}, {"id", nullptr}, {"message", "unparsable input line"}});
                continue;
            }
            const std::string type = message.value("type", "");
            if (type == "generate") {
                handleGenerate(message);
            } else if (type == "cancel") {
                cancelled_.store(true);
            } else if (type == "shutdown") {
                break;
            }
        }
        finishGeneration();
    }

private:
    void handleGenerate(const json& message) {
        const runner::GenerateRequest request = runner::parseRequest(message);

        if (busy_.exchange(true)) {
            emit(json{{"type", "error"}, {"id", request.id}, {"message", "busy"}});
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
                        emit(json{{"type", "token"}, {"text", token}});
                        return ov::genai::StreamingStatus::RUNNING;
                    });
                const int nativeReason = results.finish_reasons.empty() ? 0
                    : static_cast<int>(results.finish_reasons.front());
                const std::string reason = runner::finishReason(cancelled_.load(), nativeReason);
                emit(json{{"type", "done"}, {"id", request.id}, {"finishReason", reason}});
            } catch (const std::exception& e) {
                emit(json{{"type", "error"}, {"id", request.id}, {"message", std::string("generate failed: ") + e.what()}});
            } catch (...) {
                emit(json{{"type", "error"}, {"id", request.id}, {"message", "generate failed: unknown exception"}});
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
    const runner::Options options = runner::parseArgs(argc, argv);
    if (options.modelPath.empty()) {
        emit(json{{"type", "error"}, {"id", nullptr}, {"message", "no --model given"}});
        return 2;
    }

    std::unique_ptr<ov::genai::LLMPipeline> pipeline;
    try {
        ov::genai::Tokenizer tokenizer(options.modelPath);
        pipeline = std::make_unique<ov::genai::LLMPipeline>(options.modelPath, tokenizer, options.device);
    } catch (const std::exception& e) {
        emit(json{{"type", "error"}, {"id", nullptr}, {"message", std::string("load failed: ") + e.what()}});
        return 1;
    }

    emit(json{{"type", "ready"},
              {"model", options.modelPath},
              {"device", options.device},
              {"ctx", options.contextLength}});

    Runner runner(std::move(pipeline));
    runner.run();
    return 0;
}
