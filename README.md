# Type-Like-Chuck OpenVINO GenAI runner

Native stdio runner used by the TLC VS Code extension as the `ovgenai` local
engine kind. Loads an OpenVINO IR model directory through
`ov::genai::LLMPipeline` and serves FIM completions over newline-delimited
JSON on stdin/stdout. Release targets: **Windows x64**, **Linux x64/arm64**.
Windows arm64 is not buildable (no OpenVINO GenAI runtime archive); on macOS
the extension uses an MLX backend instead of this runner.

## Protocol

Same contract as the MLX runner (`type-like-chuck-backed-mlx`) — the TLC
extension speaks one protocol for both runners.

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

CI (`.github/workflows/runner.yml`) builds the Windows/Linux matrix, runs the
smoke test, and publishes the binaries as a GitHub release on `v*` tags. The TLC
plugin imports it from the latest release (or a pinned `RUNNER_VERSION` tag)
when packing its `.vsix`.

## Real file experiment

`task e2e:real-file` runs raw FIM against the bundled fixture
(`scripts/fixtures/api_test.go`, cursor at line 96, column 9) with the plugin's
prompt shape (qwen2 markers, 8000/2000 clamp, reserved-token masking) but no
newline stop — it shows the backend's natural, possibly multi-line completion
plus an overlap report against the line after the cursor. `FILE`, `LINE`,
`COLUMN`, `MAX_TOKENS`, `MODEL_DIR`, `DEVICE` override the defaults;
`task e2e:real-file:npu` pins NPU with the nf4 model (first load compiles the
graph — long ready timeout).

## Benchmark

`task e2e:bench` measures compile time and inference speed across the Qwen2.5-Coder
quantization line on the selected devices (NPU runs the m4gshm nf4/int8-sym/int4-sym
models; the upstream int8/int4 and the m4gshm sym models run on GPU/CPU). The
default prompt is a plugin-shaped FIM over the bundled fixture (4815 chars ≈ 1.5–2k
token prefill — what the plugin actually sends); `--prompt tiny` switches to the
one-line FIM. 64 generated tokens, steady state of 2 runs (tables below: an earlier
run with the one-line prompt).

### ASUS EXPERTBOOK B9406CAA  (Intel(R) Core(TM) Ultra X7 358H, 63 GB, Windows 11 Pro 26H2)

#### NPU — Intel(R) AI Boost  (ctx=0, runs=2, maxTokens=64, props=runner defaults)

```text
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-nf4            0.7    0.06   0.18     16    87.2
1.5B-nf4            1.1    0.10   0.86     51    59.5
3B-nf4              1.8    0.18   1.49     44    29.6
0.5B-int8-sym       0.8    0.04   0.55     51    92.7
1.5B-int8-sym       1.5    0.12   1.32     51    38.6
3B-int8-sym         2.6    0.16   1.42     27    19.0
0.5B-int4-sym       0.7    0.08   0.49     53   108.5
1.5B-int4-sym       1.2    0.16   1.04     51    49.1
3B-int4-sym         2.1    0.29   1.10     32    29.1
```

#### GPU — Intel(R) Arc(TM) B390 GPU (iGPU)  (ctx=0, runs=2, maxTokens=64, props=runner defaults)

```text
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-int8           0.9    0.04   0.45     43    95.3
1.5B-int8           1.0    0.06   1.02     49    47.9
3B-int8             1.2    0.10   0.39     11    28.3
0.5B-int4           1.1    0.04   0.31     43   139.5
1.5B-int4           1.3    0.04   0.27     20    74.9
3B-int4             1.4    0.06   0.25     11    44.7
0.5B-int8-sym       0.7    0.02   0.41     51   125.0
1.5B-int8-sym       0.9    0.06   1.00     51    50.7
3B-int8-sym         1.1    0.04   1.10     27    24.4
0.5B-int4-sym       0.6    0.02   0.31     45   146.5
1.5B-int4-sym       0.7    0.04   0.64     51    80.1
3B-int4-sym         1.0    0.06   1.15     53    46.2
```

### ThinkPad X9-15 Gen 1  (Intel(R) Core(TM) Ultra 7 258V, 32 GB, Windows 11 Home 26H2)

#### NPU — Intel(R) AI Boost  (ctx=0, runs=2, maxTokens=64, props=runner defaults)

```text
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-nf4            0.9    0.11   0.39     16    40.9
1.5B-nf4            1.7    0.20   1.67     56    33.5
3B-nf4              3.2    0.28   2.92     43    14.7
0.5B-int8-sym       1.1    0.08   1.11     51    46.0
1.5B-int8-sym       2.3    0.24   2.77     51    18.4
3B-int8-sym         4.8    0.28   2.62     27    10.3
0.5B-int4-sym       1.0    0.17   1.16     53    45.8
1.5B-int4-sym       1.8    0.30   2.11     51    24.2
3B-int4-sym         3.3    0.48   2.51     35    13.9
```

#### GPU — Intel(R) Arc(TM) 140V GPU (iGPU)  (ctx=0, runs=2, maxTokens=64, props=runner defaults)

```text
model          compile_s  ttft_s  gen_s  tokens  tok_s
0.5B-int8           0.8    0.03   0.61     43    70.5
1.5B-int8           1.2    0.08   1.56     49    31.3
3B-int8             2.2    0.14   0.56     11    19.6
0.5B-int4           0.7    0.05   0.52     43    83.3
1.5B-int4           1.0    0.06   0.42     20    47.4
3B-int4             1.7    0.11   0.38     11    29.3
0.5B-int8-sym       0.8    0.03   0.61     51    83.7
1.5B-int8-sym       1.2    0.08   1.58     51    32.3
3B-int8-sym         2.2    0.06   1.64     27    16.5
0.5B-int4-sym       0.6    0.05   0.48     45    93.0
1.5B-int4-sym       1.0    0.06   1.03     51    49.5
3B-int4-sym         1.6    0.11   1.78     53    29.8
```

`compile_s` — time to the `ready` event (includes OpenVINO graph compilation; the
upstream int4 (slow compile) and int8 (superseded by int8-sym) models are excluded
on NPU, their compiled blobs are cached across runs; nf4 is excluded on GPU),
`ttft_s` — time to first token, `tok_s` — generation
throughput. Filter models with `E2E_SIZES="0.5B-int8-sym,1.5B-nf4"`, devices with
`E2E_DEVICES=GPU task e2e:bench`.

