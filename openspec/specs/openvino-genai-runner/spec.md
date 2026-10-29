# openvino-genai-runner Specification

## Purpose

Provides the native OpenVINO GenAI inference runner: per-platform release binaries published by this repository and imported by the TLC extension package, communicating with the extension over stdin/stdout with newline-delimited JSON, resolving the OpenVINO runtime libraries and running native IR (int4-ov) models with streamed token output.

## Requirements

### Requirement: Per-platform runner binaries
The repository SHALL produce a prebuilt runner binary for each supported platform variant (Windows x64, Linux x64) with release-asset names, locally via the build tasks (`build/` directory) and in CI; `v*` tags SHALL publish both binaries as GitHub release assets.

#### Scenario: Local build outputs
- **WHEN** the build tasks (`task -t tasks.yaml build`, `task -t tasks.yaml linux`) complete
- **THEN** `build/` contains `tlc-ovgenai-runner-linux-x64` and, on a native Windows host, `tlc-ovgenai-runner-win32-x64.exe`

#### Scenario: Release publication
- **WHEN** a `v*` tag is pushed
- **THEN** CI builds both target binaries, smoke-tests the library loader on each OS, and attaches both binaries to a GitHub release

### Requirement: NDJSON stdio protocol
The runner SHALL read one completion request per stdin line as a JSON object and SHALL write newline-delimited JSON events to stdout: one `token` event per generated piece (with the text), one `done` event at the end of a generation, and `error` events on failures. Human-readable logs SHALL go to stderr only. stdout SHALL carry no non-JSON output.

#### Scenario: Streaming generation
- **WHEN** a completion request is written to the runner's stdin
- **THEN** the extension receives zero or more `token` events followed by a `done` event on stdout, in order

#### Scenario: Logs never corrupt the stream
- **WHEN** the runner logs diagnostics during generation
- **THEN** the log lines appear on stderr and stdout remains valid NDJSON

#### Scenario: Sequential requests
- **WHEN** a `generate` request arrives while a generation is in progress
- **THEN** the runner emits an `error` event (`busy`) and the running generation is unaffected

#### Scenario: Cancel
- **WHEN** a `cancel` request arrives mid-generation
- **THEN** generation stops and a `done` event with `finishReason:"cancel"` follows

### Requirement: Runtime library resolution
The runner SHALL locate the OpenVINO/TBB shared libraries next to its own binary or via the platform library path the extension prepends before spawning (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux).

#### Scenario: Loader smoke
- **WHEN** the runner binary is started with no arguments and the library path set
- **THEN** it exits with code 2 and emits a `{"type":"error"}` event, proving the OpenVINO/TBB libraries resolve

### Requirement: IR model inference contract
The runner SHALL load an OpenVINO IR model (`.xml` + `.bin` pair, or a model directory containing it with tokenizer files) given by path, accept a prompt string, a max-token limit and stop strings per request, and generate a single completion honoring the stop strings. The context window SHALL respect the model's compiled context length.

#### Scenario: Stop string ends generation
- **WHEN** the model generates text containing a request stop string
- **THEN** generation stops, the stop string is not included in the emitted tokens, and a `done` event follows

#### Scenario: Model files missing
- **WHEN** the configured model path does not contain both the `.xml` and `.bin` files
- **THEN** the runner reports an error event
