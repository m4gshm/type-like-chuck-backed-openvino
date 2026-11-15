# Tasks

## 1. Listen mode

- [x] 1.1 Parse `--listen <endpoint>` in `src/runner.hpp` (arg tests in `src/tests.cpp`: missing value, invalid path forms) and verify `task test`
- [x] 1.2 Implement the Windows named pipe server and Linux unix socket server (bind-then-probe ownership check, guard instance on Windows) in `src/main.cpp`, parameterizing the NDJSON loop over the line source (stdin or connection); verified by `scripts/smoke_listen.ps1` and `scripts/smoke_listen_linux.py` (podman ubuntu24)
- [x] 1.3 Report usage/loader failure on a taken endpoint (`error` event, exit 1); verified in both smoke scripts

## 2. Client refcount and self-exit

- [x] 2.1 Track open connections; exit cleanly (cancel active generation with `finishReason:"cancel"`) when the count returns to zero after having been ≥1; verified in both smoke scripts
- [x] 2.2 Treat a failed/blocked client write as a disconnect (write timeout on Windows, SO_SNDTIMEO on Linux, `sinkBroken_` drop); verified with a client that dies mid-generation

## 3. Token ids, PID and port

- [x] 3.1 Add the request `id` to every `token` event (named constant field, `kEventToken`); keep stdin/stdout behavior otherwise identical; verified `task test` and `task e2e:protocol` (ALL PASSED)
- [x] 3.2 Add `pid` to the `ready` event and verify it in the protocol e2e
- [ ] 3.5 Accept `--token <secret>` and require `auth` as the first message of every listen-mode client connection (wrong/missing token → `error` + close); stdin mode unchanged; verify in the smoke scripts (with-token client works, token-less client is rejected)
- [ ] 3.3 Parse `--listen-tcp <port|0>` in `src/runner.hpp` (arg tests) and implement the localhost TCP server (port 0 = random free port) in `src/main.cpp`; verify with the smoke scripts over TCP
- [ ] 3.4 Report the chosen `port` on the `ready` event when `--listen-tcp 0`; verify in the protocol e2e / smoke

## 4. Contract and release

- [x] 4.1 Update `README.md` protocol section (token `id`, `ready.pid`, `--listen`, refcount self-exit) and `openspec/specs/openvino-genai-runner/spec.md` via this change's delta on archive
- [x] 4.2 Sync the mlx backend repo (`type-like-chuck-backed-mlx`, same change name) so both runners honor the identical contract; verify by diffing the protocol sections of both READMEs
- [ ] 4.3 Update `README.md` with `--listen-tcp` and the ready `port` field
- [ ] 4.4 Run `task build`, `task test`, `task e2e:protocol`; publish a `v*` release so the extension can pin it via `RUNNER_VERSION`
