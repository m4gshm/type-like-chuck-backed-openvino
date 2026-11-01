# Design

## Context

The OVMS engine kind failed in practice (garbage inference). Reconnaissance of `openvino-node@2026.4.1` (npm tarball inspected directly) confirmed it is not viable for an LLM backend: the API surface is Core/Model/CompiledModel/InferRequest only — no `LLMPipeline`, no tokenizer, and no state API (`query_state`/`reset_state` absent from both `.d.ts` and the addon build). The `OpenVINO/*` int4-ov models are stateful KV-cache models, so incremental decoding is impossible through those bindings.

Decision: a compact custom C++ runner on the OpenVINO GenAI C++ API, speaking NDJSON over stdio, binaries published as GitHub release assets (imported by the TLC plugin — not built there), OpenVINO runtime libraries downloaded on demand by the extension.

## Goals / Non-Goals

- Goals: full control over inference (no serving layer to debug); native int4-ov IR models; NPU/GPU(Arc)/CPU on Windows x64 + Linux x64; reproducible cross-platform builds from any host (Linux via Docker).
- Non-Goals: HTTP/OpenAI API inside the runner; macOS release support (dev builds only); MLX or other runtime kinds; extension-side transport, packaging and model catalog (owned by the TLC repo).

## Decisions

### D1. Stdio NDJSON protocol instead of HTTP
The runner is a private child process; its stdin/stdout pipes are the transport. No ports, no API keys, no health endpoint — readiness is an explicit `ready` event on stdout.

Protocol (one JSON value per `\n`-terminated line):
- Requests in (stdin): `{"type":"generate","id":1,"prompt":"...","maxTokens":128,"stop":["\n"]}`.
- Events out (stdout): `{"type":"ready","model":"...","device":"NPU","ctx":8192}`, `{"type":"token","id":1,"text":"..."}`, `{"type":"done","id":1,"finishReason":"stop"|"length"|"cancel"}`, `{"type":"error","id":1,"message":"..."}`.
- Logs (stderr): free-form lines, forwarded by the extension to the TLC output channel prefixed `ovgenai:`.
- Cancellation: `{"type":"cancel","id":1}` on stdin; the runner stops generation and emits `done` with `finishReason:"cancel"`.
- Sequential request handling only (FIM completion is one-at-a-time); a request arriving mid-generation is rejected with an `error` event (`busy`), the extension serializes requests itself.

### D2. Release-named binaries in `build/`, plugin imports them
Build tasks produce binaries named after the release assets directly in `build/`:
`tlc-ovgenai-runner-linux-x64`, `tlc-ovgenai-runner-win32-x64.exe`, `tlc-ovgenai-runner-macos-arm64` (dev only). The TLC repo does not build the backend — it copies from this `build/` directory locally, or downloads from this repo's GitHub release in CI (pinnable via `RUNNER_VERSION`). CI smoke-tests the loader (the binary must start with no args, exit 2 and emit `{"type":"error"}` — proving OpenVINO/TBB libraries resolve), uploads artifacts, and `softprops/action-gh-release` publishes both target binaries on `v*` tags.

### D3. C++ runner on `ov::genai::LLMPipeline`
`LLMPipeline` provides tokenization, detokenization, streaming callbacks and stop-string handling. The runner is a few hundred lines: NDJSON line reader on stdin, `LLMPipeline` with a streamer lambda emitting `token` events, generation config per request (max tokens, stop strings), device passed at compile time via the GenAI device string. CMake + static analysis-light build; CI (GitHub Actions matrix windows-latest / ubuntu-latest) builds and attaches the release binaries.

### D4. Build automation
`tasks.yaml` (go-task, cross-platform): downloads the OpenVINO GenAI archive into `.cache/ov-genai/` (resumable, drops proxy HTML error pages), native host builds (mac clang / Windows MSVC), Linux x64 in Docker (`--platform linux/amd64`, works from any host — cross-compilation is not supported because OpenVINO ships MSVC/clang-native libs). A `smoke` task runs a scripted NDJSON session against a real int4-ov model.

## Risks / Trade-offs

- [Binaries not in git] → built by tasks/CI and attached to releases; `build/` and `.cache/` are gitignored.
- [LLMPipeline streaming granularity] → token events may arrive per-piece rather than per-token; acceptable — the extension assembles incrementally either way.
- [Windows library path] → the extension puts the runtime DLL directory on `PATH` before spawning; the runner also resolves libraries next to its own binary.

## Appendix: Recon results (task 1)

### OpenVINO GenAI runtime archives (verified via storage.openvinotoolkit.org filetree)

GitHub releases carry no assets; archives live on S3 at `https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/<version>/<os>/`:

| Platform | Asset (pinned, version `2026.4.0.0`) | Size |
| --- | --- | --- |
| Windows x64 | `windows/openvino_genai_windows_2026.4.0.0_x86_64.zip` | ~252 MB |
| Ubuntu 24 x64 | `linux/openvino_genai_ubuntu24_2026.4.0.0_x86_64.tar.gz` | ~135 MB |
| Ubuntu 22 x64 | `linux/openvino_genai_ubuntu22_2026.4.0.0_x86_64.tar.gz` | ~132 MB |
| macOS arm64 (dev only) | `pre-release/2026.4.1.0rc2/openvino_genai_macos_12_6_2026.4.1.0rc2_arm64.tar.gz` | — |

Archive layout (verified by unpacking the ubuntu24 archive): `openvino_genai_<os>_<ver>_x86_64/` containing `runtime/lib/intel64/` (libopenvino.so, libopenvino_genai.so, all device plugins incl. NPU), `runtime/include/openvino/genai/*`, `runtime/3rdparty/tbb/lib/`, `setupvars.sh`. On Windows the libraries are under `runtime/bin/intel64/Release/`. There is also a `python/` subtree (not needed).

CMake consumption: `-DOpenVINO_DIR=<unpacked>/runtime/cmake` (and link `openvino::runtime`, `openvino::genai`); OpenVINO GenAI is a separate CMake package: `find_package(OpenVINO COMPONENTS Runtime)` + `find_package(OpenVINOGenAI CONFIG PATHS "${OpenVINO_DIR}" NO_DEFAULT_PATH)` + bundled TBB link. Linux CI needs `ocl-icd-opencl-dev`.
