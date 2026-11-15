# Tasks

## 1. Listen mode

- [ ] 1.1 Parse `--listen <endpoint>` in `src/runner.hpp` (arg tests in `src/tests.cpp`: missing value, invalid path forms) and verify `task test`
- [ ] 1.2 Implement the Windows named pipe server and Linux unix socket server in `src/main.cpp`, parameterizing the NDJSON loop over the line source (stdin or connection); verify with a manual two-connection smoke script
- [ ] 1.3 Report usage/loader failure on a taken pipe name (`error` event, exit 1) and verify with a unit or manual test

## 2. Client refcount and self-exit

- [ ] 2.1 Track open connections; exit cleanly (cancel active generation with `finishReason:"cancel"`) when the count returns to zero after having been ≥1; verify by connecting and dropping clients in a scripted smoke test
- [ ] 2.2 Treat a failed/blocked client write as a disconnect (write timeout on Windows); verify with a client that stops reading

## 3. Token ids and PID

- [ ] 3.1 Add the request `id` to every `token` event (named constant field, `kEventToken`); keep stdin/stdout behavior otherwise identical; verify `task test` and `task e2e:protocol`
- [ ] 3.2 Add `pid` to the `ready` event and verify it in the protocol e2e

## 4. Contract and release

- [ ] 4.1 Update `README.md` protocol section (token `id`, `ready.pid`, `--listen`, refcount self-exit) and `openspec/specs/openvino-genai-runner/spec.md` via this change's delta on archive
- [ ] 4.2 Sync the mlx backend repo (`type-like-chuck-backed-mlx`, same change name) so both runners honor the identical contract; verify by diffing the protocol sections of both READMEs
- [ ] 4.3 Run `task build`, `task test`, `task e2e:protocol`; publish a `v*` release so the extension can pin it via `RUNNER_VERSION`
