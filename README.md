# Type-Like-Chuck OpenVINO GenAI runner

Native stdio runner used by the TLC VS Code extension as the `ovgenai` local
engine kind. Loads an OpenVINO IR model directory through
`ov::genai::LLMPipeline` and serves FIM completions over newline-delimited
JSON on stdin/stdout. Release targets: **Windows x64**, **Linux x64/arm64**.
Windows arm64 is not buildable (no OpenVINO GenAI runtime archive); on macOS
the extension uses an MLX backend instead of this runner.

## Protocol

Spawn: `tlc-ovgenai-runner --model <model-dir> --device CPU|GPU|NPU [--ctx N] [--prop KEY=VALUE ...]`
(`--device` defaults to GPU). `tlc-ovgenai-runner --version` prints a
`{"type":"version","app":"...","ovGenai":"X.Y.Z.W"}` event and exits 0 without loading a model.
`app` is the runner's own version from git: an exact tag on HEAD (`v1.2.3`,
`-dirty` for a modified tree), otherwise `<tag>-<N>-g<sha>+<branch>.<build-time>`.

`--prop` passes OpenVINO plugin properties to the pipeline (repeatable); values are
typed automatically: `true`/`false` → bool, integer → int64, otherwise string, e.g.
`--prop NPUW_LLM_MAX_PROMPT_LEN=2048 --prop CACHE_DIR=C:\tmp\ovcache`.

On NPU the runner applies tuned defaults (`NPU_TURBO=NO`, `NPU_COMPILER_TYPE=PLUGIN`,
`NPUW_LLM_GENERATE_HINT=BEST_PERF`, dynamic prefill with chunk 256 and pyramid
attention, `NPUW_DEVICES=NPU,GPU` fallback, …); `--prop` overrides win, and `--ctx N`
maps to `MAX_PROMPT_LEN` (compiled NPU graph window). Compiled NPU blobs are cached
in `.cache/npu-blobs/` (relative to the working directory) so only the first start
compiles; override with `--prop CACHE_DIR=...`, or run with no blob cache via
`--prop CACHE_DIR=` (empty value removes the property — for users without write
access to the cache location; every start then recompiles).

Events on stdout (one JSON object per line):
- `{"type":"version","app":"...","ovGenai":"X.Y.Z.W","pid":N}` for `--version` (before any model load);
- `{"type":"ready","model":...,"device":...,"ctx":N,"pid":N[,"port":N]}` after the pipeline loads;
- `{"type":"token","id":N,"text":"..."}` per generated piece;
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel"}`;
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...]}`,
`{"type":"cancel","id":N}` (cancels the active generation, or a queued one by id —
the cancelled queued request is answered immediately with
`{"type":"done","id":N,"finishReason":"cancel"}`), and `{"type":"shutdown"}` (drains
the queue answering every pending request with `finishReason:"cancel"`, cancels the
active generation and exits). Logs go to stderr only. The runner queues generate
requests and serves them one at a time in arrival order: `token` events carry the
id of the request being processed, `done`/`error` events carry the request id.
Exit codes: 0 = normal, 1 = model load failure, 2 = usage/loader error.

### Listen mode (shared runner)

`--listen <endpoint>` serves the same protocol over a local IPC endpoint instead of
stdin/stdout: a named pipe (`\\.\pipe\NAME`) on Windows, a unix domain socket on
Linux — no TCP port, no API key. `--listen-tcp <port>` (port `0` = pick a random
free port) serves the protocol over `127.0.0.1:<port>` instead, for setups where
local IPC does not cross the plugin/backend boundary (VS Code remote/WSL); the
chosen port is reported on the `ready` event, which the spawning process also
receives on the child's stdout. The two options are mutually exclusive.

With `--token <secret>`, every listen-mode client must send
`{"type":"auth","token":"..."}` as its first message; anything else — or a wrong
token — is answered with an `error` event and the connection is closed. Stdin/stdout
mode requires no authentication. The runner tracks its connected clients and exits
by itself when the last client disconnects (a client whose writes fail or stall past
a 5 s write timeout counts as disconnected); the process that spawned it does not
need to kill it. A second runner started on an already-owned endpoint fails with an
`error` event and exit code 1. Cancel-by-id matches only within the requesting
client's jobs, so several clients with overlapping ids never cancel or receive each
other's streams. Generation stays one request at a time globally; requests are
served in arrival order regardless of which client sent them.

## Build

Requires [go-task](https://taskfile.dev) and the OpenVINO GenAI archive for the
target platform (downloaded automatically into `.cache/ov-genai/`):

```sh
task build   # download runtime + native build for the host OS → build/
task test    # unit tests (doctest via ctest)
task smoke   # NDJSON smoke run (MODEL_DIR=<int8-ov model directory>)
task smoke:listen  # two-client shared-mode smoke (MODEL_DIR=<int4-sym model directory>)
```

Release-named binaries land in `build/`:
`tlc-ovgenai-runner-linux-x64`, `tlc-ovgenai-runner-linux-arm64`,
`tlc-ovgenai-runner-win32-x64.exe`
(Windows requires a native Windows host or CI — OpenVINO ships MSVC libs).

## Releases

Prebuilt binaries are published on
[GitHub Releases](https://github.com/m4gshm/type-like-chuck-backed-openvino/releases)
for every `v*` tag. The TLC plugin pins the release tag it imports when
packing its `.vsix` — a plugin build always pairs with the runner release it
was developed against (`RUNNER_VERSION` overrides for testing newer builds).
