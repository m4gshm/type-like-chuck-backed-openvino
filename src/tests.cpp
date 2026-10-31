#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "runner.hpp"

using runner::finishReason;
using runner::parseArgs;
using runner::parseRequest;

TEST_CASE("parseArgs") {
    SUBCASE("defaults") {
        char* argv[] = {(char*)"prog"};
        const auto o = parseArgs(1, argv);
        CHECK(o.modelPath.empty());
        CHECK(o.device == "GPU");
        CHECK(o.contextLength == 0);
    }
    SUBCASE("all flags") {
        char* argv[] = {(char*)"prog", (char*)"--model", (char*)"/m", (char*)"--device", (char*)"GPU", (char*)"--ctx", (char*)"4096"};
        const auto o = parseArgs(7, argv);
        CHECK(o.modelPath == "/m");
        CHECK(o.device == "GPU");
        CHECK(o.contextLength == 4096);
    }
    SUBCASE("version flag") {
        char* argv[] = {(char*)"prog", (char*)"--version"};
        const auto o = parseArgs(2, argv);
        CHECK(o.printVersion);
        CHECK(o.modelPath.empty());
    }
    SUBCASE("unknown args ignored") {
        char* argv[] = {(char*)"prog", (char*)"--wat", (char*)"--model", (char*)"/m"};
        const auto o = parseArgs(4, argv);
        CHECK(o.modelPath == "/m");
    }
    SUBCASE("missing value yields empty") {
        char* argv[] = {(char*)"prog", (char*)"--model"};
        const auto o = parseArgs(2, argv);
        CHECK(o.modelPath.empty());
    }
    SUBCASE("ctx parse failure throws") {
        char* argv[] = {(char*)"prog", (char*)"--ctx", (char*)"abc"};
        CHECK_THROWS(parseArgs(3, argv));
    }
    SUBCASE("properties parsed as key=value") {
        char* argv[] = {(char*)"prog", (char*)"--prop", (char*)"NPUW_LLM_MAX_PROMPT_LEN=2048",
                        (char*)"--prop", (char*)"NPUW_DYN_BATCH=true", (char*)"--prop", (char*)"CACHE_DIR=./c",
                        (char*)"--prop", (char*)"novalue", (char*)"--prop", (char*)"=novalue"};
        const auto o = parseArgs(11, argv);
        CHECK(o.properties.size() == 3);
        CHECK(o.properties.at("NPUW_LLM_MAX_PROMPT_LEN") == "2048");
        CHECK(o.properties.at("NPUW_DYN_BATCH") == "true");
        CHECK(o.properties.at("CACHE_DIR") == "./c");
    }
    SUBCASE("npu defaults applied, user props win") {
        char* argv[] = {(char*)"prog", (char*)"--device", (char*)"NPU", (char*)"--ctx", (char*)"4096",
                        (char*)"--prop", (char*)"NPU_TURBO=NO"};
        const auto props = runner::effectiveProperties(parseArgs(7, argv).properties, 4096, true);
        CHECK(props.size() == 16);
        CHECK(props.at("NPU_TURBO") == "NO");                       // user override wins
        CHECK(props.at("NPU_USE_NPUW") == "YES");                   // default added
        CHECK(props.at("NPUW_DEVICES") == "NPU,GPU");               // fallback chain
        CHECK(props.at("MAX_PROMPT_LEN") == "4096");                // --ctx wired on NPU
    }
    SUBCASE("cpu gets no npu props") {
        const auto props = runner::effectiveProperties({}, 4096, false);
        CHECK(props.empty());
    }
}

TEST_CASE("parseRequest") {
    SUBCASE("full message") {
        const auto msg = runner::json::parse(R"({"id":7,"prompt":"p","maxTokens":32,"stop":["\n","END"]})");
        const auto r = parseRequest(msg);
        CHECK(r.id == 7);
        CHECK(r.prompt == "p");
        CHECK(r.maxTokens == 32);
        CHECK(r.stop == std::vector<std::string>{"\n", "END"});
    }
    SUBCASE("defaults") {
        const auto r = parseRequest(runner::json::object());
        CHECK(r.id == 0);
        CHECK(r.prompt.empty());
        CHECK(r.maxTokens == 128);
        CHECK(r.stop.empty());
    }
    SUBCASE("non-array stop ignored") {
        const auto r = parseRequest(runner::json{{"stop", "\n"}});
        CHECK(r.stop.empty());
    }
}

TEST_CASE("finishReason") {
    SUBCASE("cancel wins over everything") {
        CHECK(finishReason(true, 1) == "cancel");
        CHECK(finishReason(true, 2) == "cancel");
    }
    SUBCASE("native LENGTH (2) means token limit") {
        CHECK(finishReason(false, 2) == "length");
    }
    SUBCASE("everything else is a stop") {
        CHECK(finishReason(false, 0) == "stop");  // NONE
        CHECK(finishReason(false, 1) == "stop");  // STOP (EOS / stop strings)
        CHECK(finishReason(false, 3) == "stop");  // TOOL_CALL
    }
}
