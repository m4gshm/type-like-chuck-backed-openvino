#!/usr/bin/env python3
"""Two-client smoke for --listen mode over a unix socket (Linux, or a container).

Usage: python3 scripts/smoke_listen_linux.py [--model DIR] [--device CPU]
Defaults to .cache/models/qwen2.5-coder-0.5b-int4-sym (compiles fast; the
int4/nf4-ov variants are slow on the NPU).
"""

import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OV_DIR = os.path.join(ROOT, ".cache/ov-genai/linux/openvino_genai_ubuntu24_2026.4.1.0_x86_64/runtime")
OV_LIB = OV_DIR + "/lib/intel64:" + OV_DIR + "/3rdparty/tbb/lib"
SOCK = "/tmp/tlc-runner-smoke.sock"


def library_env():
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = OV_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


class Client:
    def __init__(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(150):
            try:
                self.sock.connect(SOCK)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.2)
        else:
            raise RuntimeError("cannot connect to " + SOCK)
        self.buffer = b""

    def line(self, timeout=300):
        self.sock.settimeout(timeout)
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("connection closed")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def send(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def close(self):
        self.sock.close()


def main():
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv \
        else os.path.join(ROOT, ".cache/models/qwen2.5-coder-0.5b-int4-sym")
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "CPU"
    binary = os.path.join(ROOT, "build/linux/tlc-ovgenai-runner")
    env = library_env()

    def spawn():
        return subprocess.Popen([binary, "--model", model, "--device", device, "--listen", SOCK],
                                env=env, stderr=subprocess.DEVNULL)

    proc = spawn()
    time.sleep(15)  # model load

    c1 = Client()
    ready = c1.line()
    assert ready["type"] == "ready" and ready["pid"] == proc.pid, ready
    print("client1 ready: pid", ready["pid"])

    # a second runner on the owned endpoint must fail with exit 1
    other = spawn()
    code = other.wait(timeout=60)
    assert code == 1, f"second runner exit {code}"
    print("second runner on taken endpoint: exit=1")

    c2 = Client()
    ready2 = c2.line()
    assert ready2["type"] == "ready", ready2

    for client, name in ((c1, "client1"), (c2, "client2")):
        client.send({"type": "generate", "id": 7, "prompt": "def add(a, b):", "maxTokens": 16, "stop": ["\n"]})
        tokens = 0
        while True:
            ev = client.line()
            if ev["type"] == "token":
                tokens += 1
                assert ev.get("id") == 7, f"wrong token id: {ev}"
            elif ev["type"] == "done":
                assert ev["id"] == 7
                print(f"{name} done: finish={ev['finishReason']} tokens={tokens} id=7")
                break
            else:
                assert ev["type"] != "error", ev

    c1.close()
    time.sleep(2)
    assert proc.poll() is None, "runner exited while client2 was connected"
    print("runner alive after client1 left")

    c2.send({"type": "generate", "id": 8, "prompt": "def sub(a, b):", "maxTokens": 16, "stop": ["\n"]})
    while True:
        ev = c2.line()
        if ev["type"] == "done":
            print("client2 served after client1 left (id=%s)" % ev["id"])
            break
        assert ev["type"] != "error", ev

    c2.close()
    code = proc.wait(timeout=30)
    assert code == 0, f"self-exit code {code}"
    print("runner self-exit after last client: exit=0")

    # a client that dies mid-generation must be dropped (broken writes)
    proc = spawn()
    time.sleep(15)
    c3 = Client()
    assert c3.line()["type"] == "ready"
    c3.send({"type": "generate", "id": 1, "prompt": "tell me a very long story", "maxTokens": 8192, "stop": []})
    time.sleep(3)  # generation is streaming into the socket
    c3.close()     # die without reading: writes to it must fail
    code = proc.wait(timeout=60)
    assert code == 0, f"dead client drop code {code}"
    print("dead client dropped, runner exited: exit=0")

    os.unlink(SOCK)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
