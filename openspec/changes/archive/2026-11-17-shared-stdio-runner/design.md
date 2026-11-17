# Design

## Context

`src/main.cpp` runs an NDJSON loop over stdin/stdout with a generation worker thread; requests queue in arrival order and tokens are emitted id-less. The extension (`type-like-chuck-vs-code`, change `shared-stdio-runner`) is the only client and owns the process lifetime. The mlx runner repo carries the identical protocol and gets the same contract (its own change `shared-stdio-runner`).

## Goals / Non-Goals

**Goals:** one runner per (model, device, context) across VS Code windows; runner lifetime bound to connected clients; backward-compatible protocol.

**Non-Goals:** TCP or remote endpoints; concurrent generation; authentication (loopback-local IPC, no ACL work beyond platform defaults).

## Decisions

### Transport: named pipe / unix socket by default, TCP for IPC boundaries
`--listen` uses `\\.\pipe\...` on Windows, a socket path on Linux; `--listen-tcp <port|0>` adds a `127.0.0.1` TCP transport for setups where local IPC does not cross the plugin/backend boundary (VS Code remote/WSL). Port `0` binds a random free port and reports it on `ready`, so the spawning extension can publish it in its discovery file. The endpoint/port value is computed by the extension, the runner treats it as opaque. Same NDJSON framing on every transport; the event loop is parameterized over "line source" (stdin or accepted connections).

Alternative considered: localhost TCP as the only transport — rejected as the default: violates the "no network port by default" invariant; kept as the opt-in boundary workaround.

### Refcount lives inside the runner
Count open client connections; on 1→0 transition (after having been ≥1) exit cleanly with the current generation cancelled. A blocked/failed client write counts as disconnect. No lock files, no owner election — cross-process state is what this design avoids.

### Token ids
Add `id` to every `token` event (constant `kEventToken` gains the field). Old clients ignore unknown fields; old runners keep working with the new extension, which falls back to oldest-pending routing when tokens arrive id-less. The only behavior change in the queue: tokens of the active generation are tagged with its id — no concurrency introduced.

### PID on `ready`
`getpid()`-equivalent added to the ready event (`kEventReady` gains `pid`). Cheap, and the extension uses it for liveness pre-checks and status display.

### Per-instance token
The extension generates a random secret per spawn and passes it as `--token <secret>`; in listen mode the runner requires `auth` as the first message of each client connection and closes the connection on a wrong or missing token. This protects the TCP transport, which any local process can reach; stdio mode keeps no authentication (its pipes are private). Token lives on the command line and in the extension's discovery file, nowhere else.

## Risks / Trade-offs

- [Windows named pipe blocking writes to a stalled client] → write with timeout / overlapped I/O; failure = disconnect.
- [Two listeners on the same pipe name] → the guard instance (`FILE_FLAG_FIRST_PIPE_INSTANCE`) makes the second runner fail; on Linux the bind-then-probe check does the same.
- [TCP on localhost without auth] → opt-in only, for IPC-boundary setups; a foreign local process could generate completions but cannot read anything private. Risk accepted; noted for review.
- [Instance discovery lives outside the runner] → per-runner discovery files, option matching (model+device+ctx for ovgenai — NPU compiles per ctx; model-only for mlx) are the extension's responsibility; the runner only reports pid/port on `ready`.
- [Backend and extension land out of order] → both directions are compatible: old runner + new extension = private spawn fallback; new runner + old extension = unchanged behavior.

## Migration Plan

1. Land listen mode + refcount + token ids here; publish a new `v*` release.
2. The extension imports it via `RUNNER_VERSION` and enables connect-or-spawn.
3. Rollback: the extension setting `tlc.local.sharedRunner: false` restores private-pipe spawns without any runner change.
