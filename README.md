# Type-Like-Chuck OpenVINO GenAI runner

Native stdio runner used by the TLC VS Code extension as the `ovgenai` local
engine kind. Loads an OpenVINO IR model directory through
`ov::genai::LLMPipeline` and serves FIM completions over newline-delimited
JSON on stdin/stdout. Targets: **Windows x64** and **Linux x64** (macOS builds
work for development only and are not a release target).

## Protocol

Spawn: `tlc-ovgenai-runner --model <model-dir> --device CPU|GPU|NPU [--ctx N] [--prop KEY=VALUE ...]`
(`--device` defaults to GPU). `tlc-ovgenai-runner --version` prints a
`{"type":"version","app":"...","ovGenai":"X.Y.Z.W"}` event and exits 0 without loading a model.
`app` is the runner's own version from git: an exact tag on HEAD (`v1.2.3`,
`-dirty` for a modified tree), otherwise `<tag>-<N>-g<sha>+<branch>.<build-time>`.

`--prop` passes OpenVINO plugin properties to the pipeline (repeatable); values are
typed automatically: `true`/`false` → bool, integer → int64, otherwise string, e.g.
`--prop NPUW_LLM_MAX_PROMPT_LEN=2048 --prop CACHE_DIR=C:\tmp\ovcache`.

On NPU the runner applies tuned defaults (`NPU_TURBO=YES`, `NPU_COMPILER_TYPE=PLUGIN`,
`NPUW_LLM_GENERATE_HINT=BEST_PERF`, dynamic prefill with chunk 256 and pyramid
attention, `NPUW_DEVICES=NPU,GPU` fallback, …); `--prop` overrides win, and `--ctx N`
maps to `MAX_PROMPT_LEN` (compiled NPU graph window). Compiled NPU blobs are cached
in `.cache/npu-blobs/` (relative to the working directory) so only the first start
compiles; override with `--prop CACHE_DIR=...`.

Events on stdout (one JSON object per line):
- `{"type":"version","app":"...","ovGenai":"X.Y.Z.W"}` for `--version` (before any model load);
- `{"type":"ready","model":...,"device":...,"ctx":N}` after the pipeline loads;
- `{"type":"token","text":"..."}` per generated piece;
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel"}`;
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...]}`
and `{"type":"cancel","id":N}`. Logs go to stderr only. One generation at a
time; a request arriving mid-generation is rejected with an `error` event
(`busy`).

## Build

Requires [go-task](https://taskfile.dev) and the OpenVINO GenAI archive for the
target platform (downloaded automatically into `.cache/ov-genai/`):

```sh
task build   # download runtime + native build for the host OS → build/
task smoke   # NDJSON smoke run (MODEL_DIR=<int8-ov model directory>)
```

Release-named binaries land in `build/`:
`tlc-ovgenai-runner-linux-x64`, `tlc-ovgenai-runner-win32-x64.exe`
(Windows requires a native Windows host or CI — OpenVINO ships MSVC libs).

CI (`.github/workflows/runner.yml`) builds both platforms, runs a loader smoke
test (the binary must start and report the missing-model error — this proves
the OpenVINO/TBB libraries resolve), uploads artifacts, and publishes both
binaries as a GitHub release on `v*` tags. The TLC plugin imports these
binaries from the latest release (or a pinned `RUNNER_VERSION` tag) when
packing its `.vsix`.
