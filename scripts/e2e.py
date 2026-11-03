#!/usr/bin/env python3
"""Manual client and E2E test suite for the tlc-ovgenai-runner.

Stdlib-only, cross-platform (macOS/Linux/Windows).

Usage:
  python3 scripts/e2e.py test [--model DIR]   # run the test suite
      Without --model only protocol tests that need no weights run.
      With --model DIR (or MODEL_DIR env) the full suite incl. FIM runs.
  python3 scripts/e2e.py manual [--model DIR] # interactive NDJSON client
  python3 scripts/e2e.py download-model [--out DIR]
                                              # fetch the default test model
  python3 scripts/e2e.py bench [--device NPU] [--prop KEY=VALUE] [--cold]
                                              # compile time + inference speed
                                              # for 0.5B/1.5B/3B models
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")
CACHE_DIR = os.path.join(ROOT, ".cache")

DEFAULT_MODEL_REPO = "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int8-ov"
DEFAULT_MODEL_DIR = os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-int8-ov")

# The upstream int4 models take very long to compile on NPU; excluded there.
BENCH_MODELS = [
    ("0.5B-int8", "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int8-ov",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-int8-ov"), True),
    ("1.5B-int8", "OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int8-ov",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-int8-ov"), True),
    ("3B-int8", "OpenVINO/Qwen2.5-Coder-3B-Instruct-int8-ov",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-int8-ov"), True),
    ("0.5B-int4", "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-int4-ov"), False),
    ("1.5B-int4", "OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-int4-ov"), False),
    ("3B-int4", "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-int4-ov"), False),
    ("0.5B-nf4", "m4gshm/Qwen2.5-Coder-0.5B-nf4-sym-g-1-se-awq",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-nf4-ov"), True),
    ("1.5B-nf4", "m4gshm/Qwen2.5-Coder-1.5B-nf4-sym-g-1-se-awq",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-nf4-ov"), True),
    ("3B-nf4", "m4gshm/Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-nf4-ov"), True),
    ("0.5B-int8-sym", "m4gshm/Qwen2.5-Coder-0.5B-int8-sym",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-int8-sym"), True),
    ("1.5B-int8-sym", "m4gshm/Qwen2.5-Coder-1.5B-int8-sym",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-int8-sym"), True),
    ("3B-int8-sym", "m4gshm/Qwen2.5-Coder-3B-int8-sym",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-int8-sym"), True),
    ("0.5B-int4-sym", "m4gshm/Qwen2.5-Coder-0.5B-int4-sym-g128-se-awq",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-0.5b-int4-sym"), True),
    ("1.5B-int4-sym", "m4gshm/Qwen2.5-Coder-1.5B-int4-sym-g128-se-awq",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-1.5b-int4-sym"), True),
    ("3B-int4-sym", "m4gshm/Qwen2.5-Coder-3B-int4-sym-g128-se-awq",
     os.path.join(CACHE_DIR, "models", "qwen2.5-coder-3b-int4-sym"), True),
]
BENCH_MAX_TOKENS = 64
BENCH_RUNS = 2

# Device for runner processes; resolved by select_device() unless set explicitly.
RUNNER_DEVICE = "CPU"
# Extra --prop KEY=VALUE arguments for every runner process (set via --prop).
RUNNER_PROPS = []

FIM_PROMPT = "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n    return a + b<|fim_middle|>"
FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


def fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def binary_path():
    if sys.platform == "win32":
        p = os.path.join(BUILD_DIR, "tlc-ovgenai-runner-win32-x64.exe")
    else:
        arch = platform.machine().lower()
        name = "tlc-ovgenai-runner-linux-arm64" if arch in ("aarch64", "arm64") else "tlc-ovgenai-runner-linux-x64"
        p = os.path.join(BUILD_DIR, name)
    if not os.path.isfile(p):
        fail(f"{p} not found - run `task build` first")
    return p


def library_env():
    """Replicate the library-path setup from Taskfile.yaml."""
    env = dict(os.environ)
    env.setdefault("OPENVINO_NPUW_LOG_LEVEL", "INFO")  # show NPU compile progress
    ov = os.path.join(CACHE_DIR, "ov-genai")
    if sys.platform == "win32":
        sub = "windows/openvino_genai_windows_2026.4.1.0_x86_64/runtime/bin"
        env["PATH"] = os.pathsep.join([
            os.path.join(ov, sub, "intel64", "Release"),
            os.path.join(ov, sub, "..", "3rdparty", "tbb", "bin"),
        ]) + os.pathsep + env.get("PATH", "")
    else:
        arch = platform.machine().lower()
        sub = "linux/openvino_genai_ubuntu22_2026.4.1.0_arm64" if arch in ("aarch64", "arm64") \
            else "linux/openvino_genai_ubuntu24_2026.4.1.0_x86_64"
        libdir = "aarch64" if arch in ("aarch64", "arm64") else "intel64"
        lib = os.pathsep.join([
            os.path.join(ov, sub, "runtime", "lib", libdir),
            os.path.join(ov, sub, "runtime", "3rdparty", "tbb", "lib"),
        ])
        env["LD_LIBRARY_PATH"] = lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


class Runner:
    """Spawned runner process; events are collected on a reader thread."""

    def __init__(self, model=None, device=None, props=None):
        self.binary = binary_path()
        args = [self.binary]
        if model:
            args += ["--model", model, "--device", device or RUNNER_DEVICE]
            args += [x for kv in (RUNNER_PROPS if props is None else props) for x in ("--prop", kv)]
        self.proc = subprocess.Popen(
            args, cwd=ROOT, env=library_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1)
        self.events = []
        self.lines = []  # raw stdout lines, for non-JSON assertions
        import threading
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.events.append(ev)
            if ev.get("type") == "done":
                text = ""
                for prev in reversed(self.events[:-1]):
                    if prev.get("type") in ("done", "error"):
                        break
                    if prev.get("type") == "token":
                        text = prev.get("text", "") + text
                print(f"<< {ev} text={text!r}", file=sys.stderr)
            elif ev.get("type") == "error":
                print(f"<< {ev}", file=sys.stderr)

    def send(self, obj):
        if obj.get("type") == "generate":
            print(f">> {json.dumps(obj, ensure_ascii=False)}", file=sys.stderr)
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, pred, timeout=120, since=0):
        """Block until an event matching pred appears after index `since`; returns it.
        timeout=None waits forever (first NPU compile of a big model can take an hour)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        seen = since
        while deadline is None or time.monotonic() < deadline:
            for ev in self.events[seen:]:
                seen += 1
                if pred(ev):
                    return ev
            if self.proc.poll() is not None:
                # the child is gone but the reader thread may not have drained
                # its stdout yet; give it a moment before declaring a timeout
                self._thread.join(2.0)
                for ev in self.events[seen:]:
                    seen += 1
                    if pred(ev):
                        return ev
                break
            time.sleep(0.02)
        raise AssertionError(f"timeout waiting for event (exit={self.proc.poll()}, events={self.events[seen:]})")

    def close(self, timeout=10):
        try:
            self.send({"type": "shutdown"})
        except OSError:
            pass
        try:
            self.proc.stdin.close()  # avoid noisy stdin flush at GC when the child is gone
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.proc.returncode


def check(cond, name):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}")
    return cond


def lastError(r):
    """Message of the first error event (may be empty if the reader thread lost the race)."""
    err = next((e.get("message", "") for e in r.events if e.get("type") == "error"), "")
    return err.strip().replace("\n", " ")[:300] or "(no error event captured)"


def select_device(model, requested, props=()):
    """Resolve --device; 'auto' probes NPU, then GPU, then CPU against the model."""
    global RUNNER_DEVICE, RUNNER_PROPS
    RUNNER_PROPS = list(props)
    if requested != "auto":
        r = Runner(model=model, device=requested)
        try:
            expect_ready(r, model)
        except Exception:
            fail(f"runner cannot load the model on {requested}: {lastError(r)} "
                 "(try --device auto or add OpenVINO properties with --prop)")
        finally:
            r.close()
        RUNNER_DEVICE = requested
        return
    for cand in ("NPU", "GPU", "CPU"):
        r = Runner(model=model, device=cand)
        try:
            expect_ready(r, model)
            RUNNER_DEVICE = cand
            print(f"device: using {cand}", flush=True)
            return
        except Exception:
            print(f"device: {cand} unavailable: {lastError(r)}", flush=True)
        finally:
            r.close()
    fail("no working device (tried NPU, GPU, CPU)")


# ---------------------------------------------------------------- protocol tests (no model)

def test_no_model():
    print("protocol: no --model given")
    r = Runner()
    code = r.proc.wait(timeout=15)
    ok = check(code == 2, f"exit code is 2 (got {code})")
    ok &= check(any(e.get("type") == "error" and "model" in e.get("message", "") for e in r.events),
                "error event mentions the model")
    return ok


def test_bad_model():
    print("protocol: bad model path")
    r = Runner(model=os.path.join(ROOT, "does-not-exist"))
    code = r.proc.wait(timeout=30)
    ok = check(code == 1, f"exit code is 1 (got {code})")
    ok &= check(any(e.get("type") == "error" and e.get("message", "").startswith("load failed") for e in r.events),
                "error event starts with 'load failed'")
    return ok


def test_unparsable_input(model):
    print("model: unparsable stdin line")
    r = Runner(model=model)
    expect_ready(r, model)
    r.proc.stdin.write("this is not json\n")
    r.proc.stdin.flush()
    ev = r.wait_for(lambda e: e.get("type") == "error")
    ok = check(ev.get("id") is None, "error event has id null")
    r.send({"type": "generate", "id": 99, "prompt": "hi", "maxTokens": 4})
    ok &= check(r.wait_for(lambda e: e.get("type") == "done").get("id") == 99,
                "process survived and completed the next request")
    ok &= check(r.close() == 0, "clean shutdown, exit 0")
    return ok


def test_unknown_type_ignored(model):
    print("model: unknown request type ignored")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "frobnicate"})
    r.send({"type": "generate", "id": 98, "prompt": "hi", "maxTokens": 4})
    ok = check(r.wait_for(lambda e: e.get("type") == "done").get("id") == 98, "no crash, request handled")
    ok &= check(all(e.get("type") != "error" for e in r.events), "no error events")
    ok &= check(r.close() == 0, "clean shutdown, exit 0")
    return ok


def test_stdout_ndjson_only():
    print("protocol: stdout carries JSON lines only")
    r = Runner(model=os.path.join(ROOT, "does-not-exist"))
    r.proc.wait(timeout=30)
    import json as _json
    ok = check(all(_json.loads(l).get("type") for l in r.lines), "every stdout line parses as a JSON object")
    return ok


# ---------------------------------------------------------------- model tests

def expect_ready(r, model):
    # First NPU load can compile the graph for many minutes; cold-cache NPU
    # needs a much larger window than CPU/GPU (seconds).
    ev = r.wait_for(lambda e: e.get("type") == "ready",
                    timeout=int(os.environ.get("E2E_READY_TIMEOUT", "300")))
    return ev


def test_generation(model):
    print("model: streaming generation")
    r = Runner(model=model)
    ready = expect_ready(r, model)
    ok = check(ready.get("device") == RUNNER_DEVICE and isinstance(ready.get("ctx"), int), f"ready event: {ready}")
    r.send({"type": "generate", "id": 1, "prompt": "The capital of France is", "maxTokens": 24})
    types = []
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token" and e.get("id", 1) == 1 or e.get("type") == "token"]
    ok &= check(done.get("id") == 1, "done event with id 1")
    ok &= check(done.get("finishReason") in ("stop", "length"), f"finishReason is stop|length (got {done.get('finishReason')})")
    ok &= check(len(tokens) > 0, f"streamed {len(tokens)} token event(s)")
    ok &= check(all("text" in t for t in tokens), "token events carry text")
    ok &= check("".join(t["text"] for t in tokens).strip() != "", "completion text is non-empty")
    code = r.close()
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


def test_max_tokens(model):
    print("model: maxTokens limit")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 2, "prompt": "tell me a very long story", "maxTokens": 64})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    ok = check(done.get("finishReason") == "length", f"finishReason is length (got {done.get('finishReason')})")
    ok &= check(len(tokens) <= 64, f"at most 64 token events (got {len(tokens)})")
    r.close()
    return ok


def test_stop_strings(model):
    print("model: stop strings")
    r = Runner(model=model)
    expect_ready(r, model)
    # greedy decoding is deterministic: the model counts on
    # ' six seven eight ...' on every supported model/device, so stopping
    # on 'seven' must cut generation right after ' six '.
    r.send({"type": "generate", "id": 3, "prompt": "count: one two three four five", "maxTokens": 64, "stop": ["seven"]})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    text = "".join(t["text"] for t in tokens)
    ok = check(done.get("finishReason") == "stop", f"finishReason is stop (got {done.get('finishReason')})")
    ok &= check("seven" not in text, f"stop string not present in output: {text!r}")
    ok &= check(len(tokens) < 64, f"generation stopped early ({len(tokens)} tokens)")
    r.close()
    return ok


def test_fim(model):
    print("model: FIM completion")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 4, "prompt": FIM_PROMPT, "maxTokens": 48, "stop": ["\n\n"]})
    done = r.wait_for(lambda e: e.get("type") == "done")
    tokens = [e for e in r.events if e.get("type") == "token"]
    text = "".join(t["text"] for t in tokens)
    ok = check(done.get("finishReason") in ("stop", "length"), f"done event (finishReason={done.get('finishReason')})")
    ok &= check(len(tokens) >= 1, f"streamed {len(tokens)} token event(s)")
    ok &= check(len(text.strip()) > 0, f"FIM produced a middle part: {text!r}")
    ok &= check(not any(tok in text for tok in FIM_TOKENS), "no FIM control tokens leak into output")
    r.close()
    return ok


def test_cancel(model):
    print("model: cancel mid-generation")
    r = Runner(model=model)
    expect_ready(r, model)
    r.send({"type": "generate", "id": 5, "prompt": "tell me a very long story", "maxTokens": 1024})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120)
    r.send({"type": "cancel", "id": 5})
    done = r.wait_for(lambda e: e.get("type") == "done", timeout=60)
    ok = check(done.get("finishReason") == "cancel", f"finishReason is cancel (got {done.get('finishReason')})")
    ok &= check(done.get("id") == 5, "done event carries the request id")
    code = r.close()
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


def test_queue(model):
    print("model: request queue (no busy, cancel by id, shutdown drain)")
    r = Runner(model=model)
    expect_ready(r, model)
    # two generates back to back: the runner queues them, no busy rejection
    r.send({"type": "generate", "id": 6, "prompt": "tell me a story", "maxTokens": 64})
    r.send({"type": "generate", "id": 7, "prompt": "another"})
    ok = check(r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 6, timeout=300).get("finishReason") in ("stop", "length"),
               "id 6 completed first (queue order)")
    ok &= check(r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 7, timeout=300) is not None,
                "id 7 completed after id 6 (queued, no busy error)")
    ok &= check(not any(e.get("type") == "error" for e in r.events), "no error events (busy removed)")
    # cancel a queued request: resolved immediately with finishReason=cancel
    r.send({"type": "generate", "id": 8, "prompt": "tell me a very long story", "maxTokens": 1024})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120)
    r.send({"type": "generate", "id": 9, "prompt": "queued"})
    r.send({"type": "cancel", "id": 9})
    ev = r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 9, timeout=30)
    ok &= check(ev.get("finishReason") == "cancel", f"queued id 9 cancelled (got {ev})")
    # cancel the active generation
    r.send({"type": "cancel", "id": 8})
    ev = r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 8, timeout=60)
    ok &= check(ev.get("finishReason") == "cancel", f"active id 8 cancelled (got {ev})")
    # shutdown drains queued requests with finishReason=cancel
    r.send({"type": "generate", "id": 10, "prompt": "tell me a very long story", "maxTokens": 1024})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120)
    r.send({"type": "generate", "id": 11, "prompt": "queued"})
    r.send({"type": "shutdown"})
    ev = r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 11, timeout=30)
    ok &= check(ev.get("finishReason") == "cancel", f"shutdown drained id 11 with cancel (got {ev})")
    ok &= check(r.wait_for(lambda e: e.get("type") == "done" and e.get("id") == 10, timeout=60).get("finishReason") == "cancel",
               "shutdown cancelled the active id 10")
    code = r.proc.wait(timeout=60)
    ok &= check(code == 0, f"clean shutdown, exit 0 (got {code})")
    return ok


# ---------------------------------------------------------------- bench

def bench_generate(r, run_id):
    """One fixed-length FIM generation; returns (ttft_s, total_s, tokens)."""
    start = len(r.events)
    t0 = time.monotonic()
    r.send({"type": "generate", "id": run_id, "prompt": FIM_PROMPT, "maxTokens": BENCH_MAX_TOKENS})
    r.wait_for(lambda e: e.get("type") == "token", timeout=120, since=start)
    ttft = time.monotonic() - t0
    done = r.wait_for(lambda e: e.get("type") == "done", timeout=300, since=start)
    total = time.monotonic() - t0
    tokens = sum(1 for e in r.events[start:] if e.get("type") == "token")
    if done.get("finishReason") not in ("length", "stop"):
        raise AssertionError(f"bench run {run_id} failed: {done}")
    return ttft, total, tokens


def blob_cache_dir(model_dir, props):
    """Per-model, per-prop-set compiled-blob cache; variants must not reuse blobs
    compiled with other parameters (or the compile-time comparison is void)."""
    key = f"{os.path.basename(model_dir)}-{hashlib.sha1('|'.join(sorted(props)).encode()).hexdigest()[:8]}"
    return os.path.join(CACHE_DIR, "npu-blobs", key)


def clear_blobs(path):
    """Delete a compiled-blob cache dir. The NPU plugin writes cached blobs
    read-only, which plain rmtree cannot remove on Windows."""
    if not os.path.isdir(path):
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IWRITE)
    shutil.rmtree(path)


def bench_compile(device, props, runs=BENCH_RUNS, cold=False, sizes=None):
    """Download 0.5B/1.5B/3B, measure compile time and inference speed on each."""
    if device == "auto":
        fail("bench needs an explicit --device (NPU, GPU or CPU)")
    rows = []
    # NPU_TURBO=true trades higher power draw for better throughput; keep both
    # variants on separate blob caches (the cache key includes the props)
    variants = ([("turbo-no", ["NPU_TURBO=false"]), ("turbo-yes", ["NPU_TURBO=true"])]
                if device == "NPU" else [("", [])])
    for name, repo, out, on_npu in BENCH_MODELS:
        if sizes and name not in sizes:
            continue
        if device == "NPU" and not on_npu and not (sizes and name in sizes):
            print(f"bench: skipping {name} on NPU (slow int4 compile)", flush=True)
            continue
        out = os.environ.get(f"E2E_{name.replace('.', '_').replace('-', '_')}_DIR", out)
        if not os.path.isfile(os.path.join(out, "openvino_model.xml")):
            print(f"bench: downloading {name} ({repo})", flush=True)
            download_model(out, repo)
        for tag, extra in variants:
            cache = blob_cache_dir(out, props + extra)
            run_props = props + extra if any(p.startswith("CACHE_DIR=") for p in props) else props + extra + [f"CACHE_DIR={cache}"]
            if cold:
                clear_blobs(cache)
            label = f"{name}{'-' + tag if tag else ''}"
            print(f"bench: {label} on {device}", flush=True)
            r = Runner(model=out, device=device, props=run_props)
            t0 = time.monotonic()
            r.wait_for(lambda e: e.get("type") == "ready", timeout=None)
            compile_s = time.monotonic() - t0
            gens = []
            for i in range(runs):
                gens.append(bench_generate(r, i))
            r.close()
            ttft, total, tokens = gens[-1]  # steady state
            rows.append((label, compile_s, ttft, total, tokens))
            print(f"  {label}: compile={compile_s:.1f}s ttft={ttft:.2f}s "
                  f"gen={total:.2f}s ({tokens} tok, {tokens/total:.1f} tok/s)", flush=True)
    print(f"\nmodel  device={device}  runs={runs}  maxTokens={BENCH_MAX_TOKENS}  "
          f"props={props if props else 'runner defaults'}")
    w = max((len(name) for name, *_rest in rows), default=len("model"))
    print(f"{'model':<{w}}  compile_s  ttft_s  gen_s  tokens  tok_s")
    for name, c, ttft, total, tokens in rows:
        print(f"{name:<{w}} {c:>9.1f} {ttft:>7.2f} {total:>6.2f} {tokens:>6} {tokens/total:>7.1f}")


# ---------------------------------------------------------------- model download

def hf_api(repo):
    out = subprocess.run(["curl", "-fsSL", f"https://huggingface.co/api/models/{repo}"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fetch(url, dest):
    """Resumable download via curl (same approach as Taskfile.yaml); tolerates flaky connections."""
    tmp = dest + ".part"
    if os.path.isfile(dest):
        print(f"  exists  {os.path.basename(dest)}")
        return
    print(f"  fetch   {os.path.basename(dest)}")
    for attempt in range(1, 11):
        r = subprocess.run(["curl", "-fSsL", "-C", "-", "--retry", "3", "--retry-delay", "2", url, "-o", tmp])
        if r.returncode == 0:
            os.replace(tmp, dest)
            return
        print(f"  attempt {attempt} failed (curl {r.returncode}), resuming...", file=sys.stderr)
        time.sleep(3)
    subprocess.run(["curl", "-fSsL", "-C", "-", url, "-o", tmp], check=True)  # last try, raise
    os.replace(tmp, dest)


def download_model(out_dir, repo=DEFAULT_MODEL_REPO):
    info = hf_api(repo)
    files = [s["rfilename"] for s in info.get("siblings", [])]
    keep = [f for f in files if f in ("config.json", "generation_config.json", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
            or (f.endswith(".xml") and "openvino" in f.lower())
            or (f.endswith(".bin") and "openvino" in f.lower())]
    if not any(f.endswith(".xml") and "tokenizer" in f for f in keep):
        print(f"warning: no openvino_tokenizer.xml listed in {repo}", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)
    for f in keep:
        dest = os.path.join(out_dir, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fetch(f"https://huggingface.co/{repo}/resolve/main/{f}", dest)


# ---------------------------------------------------------------- manual client

def manual_client(model, device="auto", props=()):
    if not model:
        fail("manual mode needs --model or MODEL_DIR")
    select_device(model, device, props)
    print(f"interactive client вЂ” type a JSON request line, or /fim <code>, /quit")
    r = Runner(model=model)
    ready = r.wait_for(lambda e: e.get("type") == "ready", timeout=300)
    print(f"ready: model={ready.get('model')} device={ready.get('device')} ctx={ready.get('ctx')}")

    import threading

    def printer():
        idx = 0
        while True:
            events = r.events
            while idx < len(events):
                ev = events[idx]
                idx += 1
                t = ev.get("type")
                if t == "token":
                    sys.stdout.write(ev.get("text", ""))
                    sys.stdout.flush()
                elif t == "done":
                    print(f"\n[done id={ev.get('id')} finishReason={ev.get('finishReason')}]")
                elif t == "error":
                    print(f"\n[error id={ev.get('id')} message={ev.get('message')}]")
                elif t == "ready":
                    pass
                else:
                    print(f"\n[{json.dumps(ev)}]")

    threading.Thread(target=printer, daemon=True).start()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "/quit":
                break
            if line.startswith("/fim"):
                code = line[4:]
                prompt = f"<|fim_prefix|>{code}<|fim_suffix|><|fim_middle|>"
                r.send({"type": "generate", "id": int(time.time()), "prompt": prompt, "maxTokens": 64})
                continue
            try:
                r.send(json.loads(line))
            except json.JSONDecodeError:
                print("not JSON вЂ” ignored")
    except KeyboardInterrupt:
        pass
    code = r.close()
    print(f"runner exited with {code}")


# ---------------------------------------------------------------- main

PROTOCOL_TESTS = [test_no_model, test_bad_model, test_stdout_ndjson_only]
MODEL_TESTS = [test_unparsable_input, test_unknown_type_ignored, test_generation,
               test_max_tokens, test_stop_strings, test_fim, test_cancel, test_queue]


def run_suite(model, device="auto", props=()):
    ok = True
    print(f"== protocol tests (no model) ==")
    for t in PROTOCOL_TESTS:
        ok &= t()
    if model:
        if not (os.path.isfile(os.path.join(model, "openvino_model.xml")) or
                any(f.endswith(".xml") and "tokenizer" in f for f in os.listdir(model))):
            fail(f"{model} does not look like an OpenVINO IR model directory")
        select_device(model, device, props)
        print(f"== model tests ({model}, device={RUNNER_DEVICE}) ==")
        for t in MODEL_TESTS:
            ok &= t(model)
    else:
        print("== model tests skipped (no --model; pass --model or set MODEL_DIR) ==")
    print(f"\n{'ALL PASSED' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cmd = ap.add_subparsers(dest="cmd", required=True)
    for name in ("test", "manual"):
        p = cmd.add_parser(name)
        p.add_argument("--model", default=os.environ.get("MODEL_DIR"), help="OpenVINO IR model directory")
        p.add_argument("--device", default="auto", help="NPU, GPU, CPU or auto (NPU -> GPU -> CPU)")
        p.add_argument("--prop", action="append", default=[],
                       help="runner OpenVINO property KEY=VALUE, repeatable")
    p = cmd.add_parser("download-model")
    p.add_argument("--repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--out", default=DEFAULT_MODEL_DIR)
    p = cmd.add_parser("bench")
    p.add_argument("--device", default="NPU", help="NPU, GPU or CPU (no auto)")
    p.add_argument("--prop", action="append", default=[],
                   help="runner OpenVINO property KEY=VALUE, repeatable")
    p.add_argument("--runs", type=int, default=BENCH_RUNS, help="generation runs per model (steady state is reported)")
    p.add_argument("--cold", action="store_true",
                   help="clear the NPU compiled-blob cache (.cache/npu-blobs) before each model")
    p.add_argument("--sizes", default=None,
                   help="comma-separated model sizes to bench (default: all BENCH_MODELS; on NPU the slow-compiling upstream int4 models are skipped unless listed explicitly; per-model dir override E2E_<name with . and - replaced by _>_DIR, e.g. E2E_0_5B_int8_DIR)")
    args = ap.parse_args()

    if args.cmd == "download-model":
        download_model(args.out, args.repo)
        print(f"model at {args.out}")
    elif args.cmd == "bench":
        bench_compile(args.device, args.prop, args.runs, args.cold,
                      args.sizes.split(",") if args.sizes else None)
    elif args.cmd == "manual":
        manual_client(args.model, args.device, args.prop)
    else:
        run_suite(args.model, args.device, args.prop)


if __name__ == "__main__":
    main()
