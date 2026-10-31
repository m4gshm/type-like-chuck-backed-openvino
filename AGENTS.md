# AGENTS.md

Native C++ stdio runner for the TLC VS Code extension (`ovgenai` local engine
kind). Loads an OpenVINO IR model via `ov::genai::LLMPipeline`, serves FIM
completions as NDJSON over stdin/stdout. Release targets: Windows x64, Linux
x64; macOS is dev-only.

Counterpart repository: the TLC extension (`m4gshm/type-like-chuck-vs-code`)
imports these binaries and reads the runner's embedded version markers — the
`ovgenai-runtime-version: ` byte-scan marker and the `version` event fields are
a cross-repo contract; change both sides together.

## Layout

- `src/main.cpp` — NDJSON loop, generation worker thread, entry point.
- `src/runner.hpp` — pure, testable logic (arg/request parsing, finishReason); no I/O, no OpenVINO.
- `src/tests.cpp` — unit tests (doctest), run via ctest.
- `CMakeLists.txt` — single-target CMake build; expects `-DOpenVINO_DIR=<unpacked archive>/runtime/cmake`; derives the runner's app version from git at configure time.
- `Taskfile.yaml` — go-task build automation (runtime download, host builds, smoke).
- `.github/workflows/runner.yml` — CI build matrix + loader smoke test; `v*` tags publish a release.
- `vendor/` — vendored headers (`nlohmann/json.hpp`).
- `openspec/` — specs and the active change (`openspec/changes/openvino-genai-stdio-runner`).

## Commands

```sh
task build    # runtime + native build for the host OS → build/
task test
task e2e:protocol
task e2e
python3 scripts/e2e.py manual --model <dir>  # interactive NDJSON client (/fim helper, /quit)
```

Windows binaries require a native Windows host or CI (OpenVINO ships MSVC
libs; no cross-compilation). OpenVINO GenAI archives are downloaded into
`.cache/ov-genai/` automatically. On Windows (PowerShell) do not chain
commands with `&&`.

## Conventions

- Comments: default to **no comments**. Write code that explains itself; add a
  comment only when the code cannot convey non-obvious *why* — invariants,
  workarounds, platform quirks, external-command behavior. Never comment what
  the next line obviously does. If a block needs an explaining comment, first
  try renaming or extracting a well-named function/constant instead. When
  editing code, delete nearby comments that became redundant. Tests follow the
  same rule: assertion names and structure should make the intent clear
  without commentary.
- Protocol is a contract: request/event schemas are specified in
  `openspec/specs/openvino-genai-runner/spec.md` and README. Changes to event
  names, fields, or exit codes must update both.
- stdout carries NDJSON events only; all logs go to stderr.
- Exit codes: 2 = usage/loader error, 1 = model load failure, 0 = normal.
- One generation at a time; mid-generation `generate` is rejected with a
  `busy` error; `cancel` stops and emits `finishReason:"cancel"`.
- Binaries in `build/` use release-asset names (`tlc-ovgenai-runner-linux-x64`,
  `tlc-ovgenai-runner-win32-x64.exe`); the TLC plugin imports them as-is.
- Keep testable logic in `src/runner.hpp` pure (no I/O, no OpenVINO types); unit
  tests go to `src/tests.cpp` and must pass via `task test`. No new
  dependencies beyond OpenVINO GenAI, TBB (bundled), and the vendored
  nlohmann/json and doctest headers.
