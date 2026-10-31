# Tasks

## 1. Reconnaissance

- [x] 1.1 Pin the OpenVINO GenAI archive download URL pattern for Windows x64 and Linux x64 and confirm the archive layout (where the GenAI libs and TBB land after unpack) (artifact: URL pattern + layout notes in `design.md` appendix; verified by downloading one archive and listing its contents)

## 2. Runner implementation

- [x] 2.1 Scaffold the C++ project: CMakeLists linking OpenVINO GenAI, NDJSON line reader on stdin, writer on stdout/stderr, ready handshake; build locally on macOS against downloaded libs to validate toolchain (artifact: repo compiles; verified by `cmake --build` and a manual `echo ... | ./tlc-ovgenai-runner` smoke run)
- [x] 2.2 Implement the generation loop: `LLMPipeline` load from IR path, `generate` request handling (prompt, maxTokens, stop strings), streamer lambda emitting `token` events, `done`/`error` events; verify with a real tiny IR model (artifact: NDJSON round-trip test script; verified by streaming tokens for a fixed prompt)
- [x] 2.3 Implement `cancel` handling and sequential-request policy (reject with `busy` while generating); verify via scripted stdin session (artifact: test script; verified by cancel mid-generation returning `finishReason:"cancel"`)
- [x] 2.4 Add runtime-library path resolution to the runner (libs next to binary, then engine dir) and document the Windows DLL/`PATH` and Linux `LD_LIBRARY_PATH` handling; verify startup against unpacked runtime libs (artifact: notes + code; verified by launching the runner with libs outside the binary directory)
- [x] 2.5 Embed the runner's own application version from git at configure time (exact tag on HEAD, otherwise `<tag>-<N>-g<sha>+<branch>.<build-time>`, `unknown` outside a repo) and report it as the `app` field of the `--version` event next to `ovGenai`; sync the protocol wording in README and the main spec (artifact: CMake git-derivation + `main.cpp` + README/spec; verified by a standalone cmake configure run — a C string define needs no unit test)

## 3. Build automation and CI

- [x] 3.1 `tasks.yaml`: runtime download (resumable, proxy-robust), native host build, Docker Linux x64 build, smoke task; release-named outputs in `build/` (artifact: `tasks.yaml`; verified by mac + docker linux builds)
- [ ] 3.2 CI workflow building the runner for windows-latest and ubuntu-latest with a loader smoke test, uploading artifacts; `v*` tags publish a GitHub release with both binaries; verify the first tag run produces a release importable by the plugin (artifact: `.github/workflows/runner.yml` + first `v*` release; verified by green run with both release assets)
