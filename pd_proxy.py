#!/usr/bin/env python3
"""pd_proxy — prefill/decode split for one llama-server on two heterogeneous GPUs.

`-sm layer` prefills about twice as fast as `-sm tensor`, `-sm tensor` decodes about 30%
faster. Only one copy of the model fits in VRAM, so this proxy runs one llama-server at a
time and carries the KV cache across a restart with /slots save + restore (via /dev/shm).

Policy: stay in decode (tensor) mode, because decode dominates an agent session; borrow
prefill (layer) mode only when the incoming prompt needs enough new tokens to pay for two
mode switches. See README.md for the measured numbers behind `prefill_threshold`.

usage: pd_proxy.py [--config config.json] [--port N] [--threshold N] [--verbose]
"""
import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


class Upstream:
    """Thin JSON client for the llama-server this proxy owns."""

    def __init__(self, port):
        self.port = port

    def __call__(self, path, body=None, method='POST', timeout=3600):
        c = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        try:
            c.request(method, path, json.dumps(body) if body is not None else None,
                      {'Content-Type': 'application/json'})
            r = c.getresponse()
            data = r.read()
            if r.status >= 400:
                raise RuntimeError(f'{path} -> {r.status} {data[:200]!r}')
            return json.loads(data) if data else None
        finally:
            c.close()


class Engine:
    """Owns exactly one llama-server; moves the slot KV between split modes."""

    STATE = 'session.bin'

    def __init__(self, cfg):
        self.cfg = cfg
        self.bin = os.path.expanduser(cfg['bin'])
        self.slot_dir = cfg['slot_dir']
        self.port = cfg['upstream_port']
        self.up = Upstream(self.port)
        self.mode = None
        self.proc = None
        self.cached = []          # tokens we believe the slot holds
        self.lock = threading.RLock()
        os.makedirs(self.slot_dir, exist_ok=True)
        self.args = (['-m', os.path.expanduser(cfg['model'])]
                     + list(cfg['common_args'])
                     + ['--slot-save-path', self.slot_dir,
                        '--host', '127.0.0.1', '--port', str(self.port)])

    # ---- process lifecycle --------------------------------------------
    def start(self, mode):
        env = dict(os.environ, **self.cfg.get('env', {}))
        logfile = os.path.join(self.slot_dir, f'llama-{mode}.log')
        self.proc = subprocess.Popen(
            [self.bin] + self.args + list(self.cfg['modes'][mode]),
            env=env, stdout=open(logfile, 'ab'), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f'{mode} server exited rc={self.proc.returncode}, see {logfile}')
            try:
                socket.create_connection(('127.0.0.1', self.port), 1).close()
                self.up('/health', method='GET', timeout=5)
                self.mode = mode
                log(f'{mode} server ready in {time.time() - t0:.0f}s')
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f'{mode} server did not become healthy')

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(180)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        self.proc = None
        self.mode = None

    def switch(self, mode):
        """Restart in `mode`, carrying the slot KV across the restart."""
        with self.lock:
            if self.mode == mode:
                return
            t0 = time.time()
            carry = False
            if self.mode is not None:
                if self.cached:
                    r = self.up(f'/slots/0?action=save', {'filename': self.STATE})
                    log(f'saved {r["n_saved"]} tok / {r["n_written"] >> 20} MiB in '
                        f'{r["timings"]["save_ms"]:.0f} ms')
                    carry = True
                self.stop()
            self.start(mode)
            if carry:
                r = self.up(f'/slots/0?action=restore', {'filename': self.STATE})
                log(f'restored {r["n_restored"]} tok in {r["timings"]["restore_ms"]:.0f} ms')
                # the KV is back in VRAM; the file is ~35 KiB/token of tmpfs (8.9 GiB at 262K)
                # and would otherwise sit in RAM until the next switch overwrites it
                try:
                    os.unlink(os.path.join(self.slot_dir, self.STATE))
                except OSError:
                    pass
            log(f'switch to {mode} took {time.time() - t0:.0f}s')

    def ensure(self, mode):
        with self.lock:
            if self.mode is None:
                self.start(mode)
            elif self.mode != mode:
                self.switch(mode)

    # ---- request path -------------------------------------------------
    def tokens_for(self, body):
        """Render a chat request the way the server will, then tokenize it."""
        rendered = self.up('/apply-template', body)['prompt']
        return self.up('/tokenize', {'content': rendered, 'parse_special': True})['tokens']

    def n_new_tokens(self, toks):
        n = 0
        for a, b in zip(self.cached, toks):
            if a != b:
                break
            n += 1
        return len(toks) - n

    def prepare(self, body, threshold):
        """Optionally prefill this prompt in layer mode. Always ends in decode mode.

        The prompt is prefilled up to its second-to-last token: llama.cpp forces a full
        re-prefill when an incoming prompt is a *strict prefix* of the restored state
        ("forcing full prompt re-processing due to lack of cache data"), so the decode
        request must contribute at least one new token.
        """
        with self.lock:
            self.ensure('decode')
            try:
                toks = self.tokens_for(body)
            except Exception as e:
                log('apply-template failed, decoding without a prefill phase:', e)
                return
            n_new = self.n_new_tokens(toks)
            if n_new < threshold:
                log(f'{n_new} new tokens (< {threshold}): staying in decode mode')
                self.cached = toks
                return
            log(f'{n_new} new tokens: borrowing prefill mode')
            t0 = time.time()
            self.switch('prefill')
            r = self.up('/completion', {'prompt': toks[:-1], 'n_predict': 1,
                                        'cache_prompt': True, 'temperature': 0})
            t = r['timings']
            log(f'prefilled {t["prompt_n"]} tok @ {t["prompt_per_second"]:.0f} t/s '
                f'({t["prompt_ms"] / 1000:.0f}s)')
            self.cached = toks[:-1]
            self.switch('decode')
            log(f'prefill phase total {time.time() - t0:.0f}s')


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _proxy(self, body_bytes):
        engine = self.server.engine
        c = http.client.HTTPConnection('127.0.0.1', engine.port, timeout=3600)
        try:
            c.request(self.command, self.path, body_bytes,
                      {'Content-Type': 'application/json'})
            r = c.getresponse()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in ('transfer-encoding', 'connection', 'content-length'):
                    self.send_header(k, v)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                self.wfile.write(b'%x\r\n' % len(chunk) + chunk + b'\r\n')
                self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n')
        finally:
            c.close()

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n)
        engine = self.server.engine
        if self.path.rstrip('/').endswith('chat/completions'):
            try:
                engine.prepare(json.loads(raw), self.server.threshold)
            except Exception as e:
                log('prepare failed, forwarding anyway:', e)
        else:
            engine.ensure('decode')
        self._proxy(raw)

    def do_GET(self):
        self.server.engine.ensure('decode')
        self._proxy(None)


def load_config(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(HERE, 'config.json'))
    ap.add_argument('--port', type=int)
    ap.add_argument('--threshold', type=int)
    a = ap.parse_args()

    cfg = load_config(a.config)
    engine = Engine(cfg)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port or cfg['listen_port']), Handler)
    srv.engine = engine
    srv.threshold = a.threshold if a.threshold is not None else cfg['prefill_threshold']

    def bye(*_):
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    engine.ensure('decode')
    log(f'pd_proxy on :{srv.server_port} -> llama-server :{engine.port} '
        f'(prefill threshold {srv.threshold} tok)')
    try:
        srv.serve_forever()
    finally:
        engine.stop()


if __name__ == '__main__':
    main()
