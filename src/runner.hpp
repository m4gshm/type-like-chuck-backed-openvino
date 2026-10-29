#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace runner {

using json = nlohmann::json;

struct Options {
    std::string modelPath;
    std::string device = "CPU";
    std::size_t contextLength = 0;
};

inline Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--model") {
            options.modelPath = next();
        } else if (arg == "--device") {
            options.device = next();
        } else if (arg == "--ctx") {
            options.contextLength = std::stoull(next());
        }
    }
    return options;
}

struct GenerateRequest {
    long long id = 0;
    std::string prompt;
    std::size_t maxTokens = 128;
    std::vector<std::string> stop;
};

inline GenerateRequest parseRequest(const json& message) {
    GenerateRequest request;
    request.id = message.value("id", 0LL);
    request.prompt = message.value("prompt", "");
    request.maxTokens = message.value("maxTokens", std::size_t{128});
    if (message.contains("stop") && message["stop"].is_array()) {
        request.stop = message["stop"].get<std::vector<std::string>>();
    }
    return request;
}

inline std::string finishReason(bool cancelled, int nativeReason) {
    if (cancelled) {
        return "cancel";
    }
    // Mirrors ov::genai::GenerationFinishReason: LENGTH == 2 means the token
    // limit was reached; STOP (EOS or stop strings) and everything else is a stop.
    return nativeReason == 2 ? "length" : "stop";
}

}  // namespace runner
