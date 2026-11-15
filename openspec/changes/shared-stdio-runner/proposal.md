# Proposal

## Why

Each VS Code window spawns its own runner process and loads its own copy of the model, so two windows double the NPU/GPU/CPU memory footprint for the same model on the same machine. The runner today speaks NDJSON over private stdin/stdout pipes, which no second client can attach to.

## What Changes

- **Listen mode**: the runner gains `--listen <endpoint>` — it serves the same NDJSON protocol over a platform-local named pipe (Windows) / unix domain socket (Linux) instead of (or after) exhausting stdin, without opening any TCP port.
- **Client refcount and self-exit**: the runner tracks connected clients and exits by itself when the last client disconnects (after having had at least one), so no client needs to kill it; the spawning window closing does not terminate the runner while other clients remain.
- **Per-request token routing**: `token` events carry the request `id` so two concurrent clients cannot receive each other's tokens (today tokens carry no id and belong to "the oldest pending request" — single-client-only).
- **PID exposure**: the runner reports its PID on the `ready` event, so clients can validate backend liveness and display it.
- Readiness for both TLC stdio backends: the same listen-mode contract is specified and implemented in the mlx runner repository (`type-like-chuck-backed-mlx`, change `shared-stdio-runner`) so the extension speaks one protocol with both.

## Capabilities

### New Capabilities

- `runner-sharing`: the shared/listen-mode lifecycle — listen endpoint, client refcount with self-termination, per-client stream isolation, PID exposure.

### Modified Capabilities

- `openvino-genai-runner`: the "NDJSON stdio protocol" requirement currently fixes token events as id-less and transport as private stdin/stdout; sharing adds token ids and the listen transport.

## Impact

- `src/main.cpp` (event loop, connection handling, refcount), `src/runner.hpp` (arg parsing for `--listen`), `src/tests.cpp`, `README.md` (protocol contract), `openspec/specs/openvino-genai-runner/spec.md`.
- Cross-repo: the TLC extension (`m4gshm/type-like-chuck-vs-code`, change `shared-stdio-runner`) connects instead of spawning and checks PID liveness; the mlx backend repo carries the same contract. Protocol stays backward compatible: stdin/stdout mode and id-less tokens keep working with old clients.
