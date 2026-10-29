# TLC OpenVINO GenAI runner

Native stdio runner for the `ovgenai` engine kind. Loads an OpenVINO IR model
directory through `ov::genai::LLMPipeline` and serves FIM completions over
newline-delimited JSON on stdin/stdout.

## Protocol

Spawn: `tlc-ovgenai-runner --model <model-dir> --device CPU|GPU|NPU [--ctx N]`

Events on stdout (one JSON object per line):
- `{"type":"ready","model":...,"device":...,"ctx":N}` after the pipeline loads;
- `{"type":"token","text":"..."}` per generated piece;
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel"}`;
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...]}`
and `{"type":"cancel","id":N}`. Logs go to stderr only. One generation at a
time; a request arriving mid-generation is rejected with an `error` event
(`busy`).

## Build

Requires the OpenVINO GenAI archive for the target platform (see the design
appendix in `openspec/changes/openvino-genai-stdio-runner/design.md`) and CMake.

```sh
cmake -S runner -B runner/build -DOpenVINO_DIR=<unpacked-archive>/runtime/cmake
cmake --build runner/build
```

CI builds the runner for windows-latest and ubuntu-latest and attaches the
binaries as release artifacts (see `.github/workflows`).
