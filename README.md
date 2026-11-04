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
- `{"type":"version","app":"...","ovGenai":"X.Y.Z.W"}` for `--version` (before any model load);
- `{"type":"ready","model":...,"device":...,"ctx":N}` after the pipeline loads;
- `{"type":"token","text":"..."}` per generated piece;
- `{"type":"done","id":N,"finishReason":"stop"|"length"|"cancel"}`;
- `{"type":"error","id":N|null,"message":"..."}`.

Requests on stdin: `{"type":"generate","id":N,"prompt":"...","maxTokens":N,"stop":[...]}`,
`{"type":"cancel","id":N}` (cancels the active generation, or a queued one by id —
the cancelled queued request is answered immediately with
`{"type":"done","id":N,"finishReason":"cancel"}`), and `{"type":"shutdown"}` (drains
the queue answering every pending request with `finishReason:"cancel"`, cancels the
active generation and exits). Logs go to stderr only. The runner queues generate
requests and serves them one at a time in arrival order: `token` events carry no id
(only one generation is active) and always belong to the request being processed;
`done`/`error` events carry the request id. Exit codes: 0 = normal, 1 = model
load failure, 2 = usage/loader error.

## Build

Requires [go-task](https://taskfile.dev) and the OpenVINO GenAI archive for the
target platform (downloaded automatically into `.cache/ov-genai/`):

```sh
task build   # download runtime + native build for the host OS → build/
task test    # unit tests (doctest via ctest)
task smoke   # NDJSON smoke run (MODEL_DIR=<int8-ov model directory>)
```

Release-named binaries land in `build/`:
`tlc-ovgenai-runner-linux-x64`, `tlc-ovgenai-runner-linux-arm64`,
`tlc-ovgenai-runner-win32-x64.exe`
(Windows requires a native Windows host or CI — OpenVINO ships MSVC libs).

## Benchmark

`task e2e:bench` measures compile time and inference speed (FIM prompt, 64 tokens,
steady state of 2 runs) across the Qwen2.5-Coder int8/int4/nf4 line on the
selected devices:

```text
== NPU ==  host=ASUS EXPERTBOOK B9406CAA  cpu=Intel(R) Core(TM) Ultra X7 358H  ram=63 GB  os=Windows 11 Pro 26H2  name=Intel(R) AI Boost  ctx=0  runs=2  maxTokens=64  props=runner defaults
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-int8           1.4    0.12   1.80     43    23.9
1.5B-int8           3.0    0.41   6.85     49     7.2
3B-int8             5.2    1.16   4.18     11     2.6
0.5B-nf4            0.7    0.06   0.18     16    87.0
1.5B-nf4            1.7    0.10   0.94     51    54.3
3B-nf4              2.8    0.18   1.57     44    28.1
0.5B-int8-sym       1.3    0.04   0.57     51    89.4
1.5B-int8-sym       2.1    0.12   1.37     51    37.3
3B-int8-sym         4.4    0.16   1.55     27    17.4
0.5B-int4-sym       0.8    0.08   0.49     53   108.0
1.5B-int4-sym       1.5    0.16   1.06     51    48.1
3B-int4-sym         2.8    0.31   1.16     32    27.5

== GPU ==  host=ASUS EXPERTBOOK B9406CAA  cpu=Intel(R) Core(TM) Ultra X7 358H  ram=63 GB  os=Windows 11 Pro 26H2  name=Intel(R) Arc(TM) B390 GPU (iGPU)  ctx=0  runs=2  maxTokens=64  props=runner defaults
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-int8           0.7    0.04   0.49     43    87.4
1.5B-int8           1.1    0.06   1.15     49    42.8
3B-int8             2.1    0.10   0.41     11    26.9
0.5B-int4           0.7    0.02   0.37     43   116.6
1.5B-int4           0.9    0.04   0.29     20    69.7
3B-int4             1.5    0.08   0.27     11    41.3
0.5B-nf4            0.8    0.04   0.25     16    65.2
1.5B-nf4            1.8    0.10   1.92     56    29.1
3B-nf4              3.3    0.20   3.89     49    12.6
0.5B-int8-sym       0.7    0.02   0.43     51   118.8
1.5B-int8-sym       1.0    0.06   1.09     51    46.9
3B-int8-sym         1.7    0.04   1.18     27    22.8
0.5B-int4-sym       0.6    0.02   0.35     45   129.5
1.5B-int4-sym       0.8    0.04   0.68     51    75.5
3B-int4-sym         1.3    0.08   1.29     53    41.1
```

`compile_s` — time to the `ready` event (includes OpenVINO graph compilation; the
slow-compiling upstream int4 models are excluded on NPU, their compiled blobs are
cached across runs), `ttft_s` — time to first token, `tok_s` — generation
throughput. Filter models with `E2E_SIZES="0.5B-int8-sym,1.5B-nf4"`, devices with
`E2E_DEVICES=GPU task e2e:bench`.

## Releases

Prebuilt binaries are published on
[GitHub Releases](https://github.com/m4gshm/type-like-chuck-backed-openvino/releases)
for every `v*` tag. The TLC plugin pins the release tag it imports when
packing its `.vsix` — a plugin build always pairs with the runner release it
was developed against (`RUNNER_VERSION` overrides for testing newer builds).
