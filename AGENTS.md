# AGENTS.md

Native C++ stdio runner for the TLC VS Code extension (`ovgenai` local engine
kind). Loads an OpenVINO IR model via `ov::genai::LLMPipeline`, serves FIM
completions as NDJSON over stdin/stdout. Release targets: Windows x64, Linux
x64/arm64 (Windows arm64 has no OpenVINO GenAI runtime archive; on macOS the
extension uses an MLX backend, not this runner).

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
task e2e:bench   # compile time + inference speed for 0.5B/1.5B/3B, int8 + int4 + nf4 (downloads models)
python3 scripts/e2e.py manual --model <dir>  # interactive NDJSON client (/fim helper, /quit)
```

Windows binaries require a native Windows host or CI (OpenVINO ships MSVC
libs; no cross-compilation). OpenVINO GenAI archives are downloaded into
`.cache/ov-genai/` automatically. On Windows (PowerShell) do not chain
commands with `&&`.

## Conventions

- Formatting follows `.clang-format` (LLVM base, `IndentWidth: 4`,
  `AccessModifierOffset: -4`, `ColumnLimit: 0`, `PointerAlignment: Left`):
  keep line breaks as written — never reflow, wrap or join lines; align
  continuation lines with the opening bracket; `public:`/`private:` at column
  0; a single space before trailing `//` comments. After writing or editing
  C++, run `task fmt` (or clang-format) before committing.
- Comments: default to **no comments**. Write code that explains itself; add a
  comment only when the code cannot convey non-obvious *why* — invariants,
  workarounds, platform quirks, external-command behavior. Never comment what
  the next line obviously does. If a block needs an explaining comment, first
  try renaming or extracting a well-named function/constant instead. When
  editing code, delete nearby comments that became redundant. Tests follow the
  same rule: assertion names and structure should make the intent clear
  without commentary.
- Shared string values that cross a boundary (NDJSON event and request type
  names) are named constants (`kEvent*`/`kRequest*` in `main.cpp`), not
  inline literals at each use site.
- Protocol is a contract: request/event schemas are specified in
  `openspec/specs/openvino-genai-runner/spec.md` and README. Changes to event
  names, fields, or exit codes must update both.
- stdout carries NDJSON events only; all logs go to stderr.
- Exit codes: 2 = usage/loader error, 1 = model load failure, 0 = normal.
- Generate requests are queued and served one at a time in arrival order;
  `cancel` (with or without `id`) stops the active or queued request and emits
  `finishReason:"cancel"`; `shutdown` drains the queue and exits.
- Binaries in `build/` use release-asset names (`tlc-ovgenai-runner-linux-x64`,
  `tlc-ovgenai-runner-win32-x64.exe`); the TLC plugin imports them as-is.
- Keep testable logic in `src/runner.hpp` pure (no I/O, no OpenVINO types); unit
  tests go to `src/tests.cpp` and must pass via `task test`. No new
  dependencies beyond OpenVINO GenAI, TBB (bundled), and the vendored
  nlohmann/json and doctest headers.
