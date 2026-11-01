# Proposal

## Why

The TLC extension's OVMS experiment failed (garbage inference on the FIM completion workload), and a reconnaissance of `openvino-node@2026.4.1` showed it exposes only the low-level runtime API — no `LLMPipeline`, no tokenizer, and no state API (`query_state`/`reset_state`), which makes incremental KV-cache decoding impossible for stateful int4-ov models. A small custom C++ runner built on the OpenVINO GenAI C++ API gives full control over inference (no third-party serving layer to debug) with tokenizer, streaming and stop strings provided out of the box.

## What Changes

- Add a **native OpenVINO GenAI runner**: a compact C++ binary per platform speaking **NDJSON over stdin/stdout** — completion requests in on stdin, streamed token events out on stdout, logs out on stderr. No ports, no HTTP, no API keys.
- Loads native OpenVINO IR models (`.xml` + `.bin` pairs, NNCF int4, e.g. `OpenVINO/Qwen2.5-Coder-*`) directly — no GGUF conversion.
- Sequential request handling (one generation at a time), mid-generation requests rejected with a `busy` error event; `cancel` stops the current generation.
- Build automation (`tasks.yaml`): OpenVINO GenAI runtime download, native host builds, Linux x64 via Docker; release-named binaries in `build/`.
- CI: build matrix (windows-latest, ubuntu-latest) with a loader smoke test; `v*` tags publish both binaries as a GitHub release, from which the TLC plugin imports them.

## Capabilities

### New Capabilities

- `openvino-genai-runner`: the native stdio runner — NDJSON protocol over stdin/stdout, per-platform release binaries, runtime-library resolution, and the IR model inference contract (streaming tokens, stop strings, context window).

## Impact

- New C++ component: NDJSON stdio server on `ov::genai::LLMPipeline`; CMake build for win-x64 and linux-x64; CI producing release binaries.
- Downstream: the TLC plugin imports the binaries from this repo's `build/` directory (local) or GitHub release (CI) and packs them into its `.vsix`; the heavy OpenVINO runtime libraries stay out of the package and are downloaded by the extension on first use.
