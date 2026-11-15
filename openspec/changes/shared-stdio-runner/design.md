# Design

## Context

`src/main.cpp` runs an NDJSON loop over stdin/stdout with a generation worker thread; requests queue in arrival order and tokens are emitted id-less. The extension (`type-like-chuck-vs-code`, change `shared-stdio-runner`) is the only client and owns the process lifetime. The mlx runner repo carries the identical protocol and gets the same contract (its own change `shared-stdio-runner`).

## Goals / Non-Goals

**Goals:** one runner per (model, device, context) across VS Code windows; runner lifetime bound to connected clients; backward-compatible protocol.

**Non-Goals:** TCP or remote endpoints; concurrent generation; authentication (loopback-local IPC, no ACL work beyond platform defaults).

## Decisions

### Transport: named pipe / unix socket via `--listen <endpoint>`
`\\.\pipe\...` on Windows, a socket path on Linux. The endpoint value is computed by the extension (digest of start options), the runner treats it as opaque. Same NDJSON framing on the socket; the event loop is parameterized over "line source" (stdin or accepted connections).

Alternative considered: localhost TCP — rejected: violates the "no network port" invariant and needs auth.

### Refcount lives inside the runner
Count open client connections; on 1→0 transition (after having been ≥1) exit cleanly with the current generation cancelled. A blocked/failed client write counts as disconnect. No lock files, no owner election — cross-process state is what this design avoids.

### Token ids
Add `id` to every `token` event (constant `kEventToken` gains the field). Old clients ignore unknown fields; old runners keep working with the new extension, which falls back to oldest-pending routing when tokens arrive id-less. The only behavior change in the queue: tokens of the active generation are tagged with its id — no concurrency introduced.

### PID on `ready`
`getpid()`-equivalent added to the ready event (`kEventReady` gains `pid`). Cheap, and the extension uses it for liveness pre-checks and status display.

## Risks / Trade-offs

- [Windows named pipe blocking writes to a stalled client] → write with timeout / overlapped I/O; failure = disconnect.
- [Two listeners on the same pipe name] → second bind fails; runner exits with a `error` event and code 1; the extension's connect-or-spawn falls back to its own private runner.
- [Backend and extension land out of order] → both directions are compatible: old runner + new extension = private spawn fallback; new runner + old extension = unchanged behavior.

## Migration Plan

1. Land listen mode + refcount + token ids here; publish a new `v*` release.
2. The extension imports it via `RUNNER_VERSION` and enables connect-or-spawn.
3. Rollback: the extension setting `tlc.local.sharedRunner: false` restores private-pipe spawns without any runner change.
