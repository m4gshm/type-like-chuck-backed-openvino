# Proposal

## Why

Each VS Code window spawns its own runner process and loads its own copy of the model, so two windows double the NPU/GPU/CPU memory footprint for the same model on the same machine. The runner today speaks NDJSON over private stdin/stdout pipes, which no second client can attach to.

## What Changes

- **Listen mode**: the runner gains `--listen <endpoint>` (named pipe on Windows, unix domain socket on Linux) and `--listen-tcp <port|0>` (a localhost TCP port; `0` = pick a random free port and report it on `ready`) — the same NDJSON protocol over a local connection instead of stdin/stdout, without requiring an API key. TCP covers setups where local IPC does not cross the plugin/backend boundary (VS Code remote/WSL).
- **Client refcount and self-exit**: the runner tracks connected clients and exits by itself when the last client disconnects (after having had at least one), so no client needs to kill it; the spawning window closing does not terminate the runner while other clients remain.
- **Per-request token routing**: `token` events carry the request `id` so two concurrent clients cannot receive each other's tokens (today tokens carry no id and belong to "the oldest pending request" — single-client-only).
- **PID and port exposure**: the runner reports its PID — and, with `--listen-tcp 0`, the chosen port — on the `ready` event, so clients can validate backend liveness and record how to connect.
- **No state files in the runner**: instance discovery (a file per runner in a shared directory, matching by model/device/context) is owned by the extension; the runner only listens and reports.
- Readiness for both TLC stdio backends: the same listen-mode contract is specified and implemented in the mlx runner repository (`type-like-chuck-backed-mlx`, change `shared-stdio-runner`) so the extension speaks one protocol with both.

## Capabilities

### New Capabilities

- `runner-sharing`: the shared/listen-mode lifecycle — listen transports (pipe/socket/TCP), client refcount with self-termination, per-client stream isolation, PID and port exposure.

### Modified Capabilities

- `openvino-genai-runner`: the "NDJSON stdio protocol" requirement currently fixes token events as id-less and transport as private stdin/stdout; sharing adds token ids and the listen transports.

## Impact

- `src/main.cpp` (event loop, connection handling, refcount), `src/runner.hpp` (arg parsing for `--listen`), `src/tests.cpp`, `README.md` (protocol contract), `openspec/specs/openvino-genai-runner/spec.md`.
- Cross-repo: the TLC extension (`m4gshm/type-like-chuck-vs-code`, change `shared-stdio-runner`) connects instead of spawning and checks PID liveness; the mlx backend repo carries the same contract. Protocol stays backward compatible: stdin/stdout mode and id-less tokens keep working with old clients.
