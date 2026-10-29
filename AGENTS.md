# AGENTS.md

Native C++ stdio runner for the TLC VS Code extension (`ovgenai` local engine
kind). Loads an OpenVINO IR model via `ov::genai::LLMPipeline`, serves FIM
completions as NDJSON over stdin/stdout. Release targets: Windows x64, Linux
x64; macOS is dev-only.

## Layout

- `src/main.cpp` — the whole runner (arg parsing, NDJSON loop, generation worker thread).
- `CMakeLists.txt` — single-target CMake build; expects `-DOpenVINO_DIR=<unpacked archive>/runtime/cmake`.
- `tasks.yaml` — go-task build automation (runtime download, host builds, Linux-in-Docker, smoke).
- `.github/workflows/runner.yml` — CI build matrix + loader smoke test; `v*` tags publish a release.
- `vendor/` — vendored headers (`nlohmann/json.hpp`).
- `openspec/` — specs and the active change (`openspec/changes/openvino-genai-stdio-runner`).

## Commands

```sh
task -t tasks.yaml build    # native build for the host OS → build/
task -t tasks.yaml linux    # Linux x64 in Docker (--platform linux/amd64)
task -t tasks.yaml smoke    # NDJSON smoke run; requires MODEL_DIR=<int4-ov model dir>
```

Windows binaries require a native Windows host or CI (OpenVINO ships MSVC
libs; no cross-compilation). OpenVINO GenAI archives are downloaded into
`.cache/ov-genai/` automatically.

## Conventions

- Protocol is a contract: request/event schemas are specified in
  `openspec/specs/openvino-genai-runner/spec.md` and README. Changes to event
  names, fields, or exit codes must update both.
- stdout carries NDJSON events only; all logs go to stderr.
- Exit codes: 2 = usage/loader error, 1 = model load failure, 0 = normal.
- One generation at a time; mid-generation `generate` is rejected with a
  `busy` error; `cancel` stops and emits `finishReason:"cancel"`.
- Binaries in `build/` use release-asset names (`tlc-ovgenai-runner-linux-x64`,
  `tlc-ovgenai-runner-win32-x64.exe`); the TLC plugin imports them as-is.
- Keep `src/main.cpp` a single file; no new dependencies beyond OpenVINO GenAI,
  TBB (bundled), and the vendored nlohmann/json header.
