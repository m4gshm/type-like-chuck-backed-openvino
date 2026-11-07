#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace runner {

using json = nlohmann::json;

struct Options {
    std::string modelPath;
    std::string device = "GPU";
    std::size_t contextLength = 0;
    // OpenVINO plugin properties from --prop KEY=VALUE, passed to the pipeline as-is.
    std::map<std::string, std::string> properties;
    // --prompt-cache: prefix caching on the PagedAttention backend (CPU/GPU);
    // NPU has no prefix-cache path, the flag is ignored there.
    bool promptCache = false;
    // --version: print the runtime version event and exit 0 (no model load).
    bool printVersion = false;
};

inline Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--model") {
            options.modelPath = next();
        } else if (arg == "--version") {
            options.printVersion = true;
        } else if (arg == "--device") {
            options.device = next();
        } else if (arg == "--ctx") {
            options.contextLength = std::stoull(next());
        } else if (arg == "--prop") {
            const std::string kv = next();
            const auto eq = kv.find('=');
            if (eq != std::string::npos && eq > 0) {
                options.properties[kv.substr(0, eq)] = kv.substr(eq + 1);
            }
        } else if (arg == "--prompt-cache") {
            options.promptCache = true;
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

// NPU property defaults tuned for LLM latency/throughput (mirrors the proven
// agent NPU config: plugin compiler, dynamic prefill, pyramid attention,
// NPU->GPU fallback). --prop values win over these.
inline std::map<std::string, std::string> npuDefaults() {
    return {
        {"PERFORMANCE_HINT", "LATENCY"},
        {"NPU_USE_NPUW", "YES"},
        {"NPU_TURBO", "NO"},
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPUW_LLM_GENERATE_HINT", "BEST_PERF"},
        {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
        {"NPUW_LLM_PREFILL_CHUNK_SIZE", "256"},
        {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
        {"NPUW_LLM_ENABLE_PREFIX_CACHING", "YES"},
        {"NPUW_LLM_ENABLE_CONTINUOUS_PREFILL", "NO"},
        {"NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE", "NO"},
        {"NPUW_LLM_SHARED_HEAD", "YES"},
        {"NPUW_PARALLEL_COMPILE", "YES"},
        {"NPUW_DEVICES", "NPU,GPU"},
        {"LOG_LEVEL", "LOG_INFO"}, // OpenVINO log level: shows NPU compile progress on stderr
        // Compiled-blob cache relative to the runner cwd; --prop CACHE_DIR=path overrides,
        // CACHE_DIR= (empty) removes it and runs with no blob cache.
        {"CACHE_DIR", ".cache/npu-blobs"},
    };
}

inline std::map<std::string, std::string> effectiveProperties(
    std::map<std::string, std::string> user, std::size_t contextLength, bool npu) {
    if (npu) {
        for (const auto& [key, value] : npuDefaults()) {
            user.emplace(key, value);
        }
        if (contextLength > 0) {
            user.emplace("MAX_PROMPT_LEN", std::to_string(contextLength));
        }
    }
    for (auto it = user.begin(); it != user.end();) {
        // `--prop KEY=` (empty value) removes KEY, e.g. CACHE_DIR= runs without a blob cache.
        if (it->second.empty()) {
            it = user.erase(it);
        } else {
            ++it;
        }
    }
    return user;
}

} // namespace runner
