# pd-proxy

Prefill/decode split for a single `llama-server` on two mismatched GPUs.

On a box with an RTX 5060 Ti (16 GB, PCIe 5.0 x8) and an RTX 4060 Ti (16 GB, PCIe 4.0 x1,
no P2P), the two llama.cpp split modes are good at opposite things:

| 128K depth, Qwen3.8-27B Q4_K_M, q8_0 KV, MTP n=3 | prefill | decode |
|---|---|---|
| `-sm layer -ts 1,0.8` | **663 t/s** | 21.2 t/s |
| `-sm tensor -ts 1,1` | 331 t/s | **28.2 t/s** |

Layer mode is a sequential pipeline, so prefill runs at roughly the sum of both cards'
throughput while decode waits on the slow card. Tensor mode (TP) splits every tensor, so
decode is bounded by the *slower* card instead of the sum, but prefill pays for the
all-reduce traffic that has to cross a PCIe x1 link through host memory.

The industry answer to this is P/D disaggregation (vLLM, SGLang, NVIDIA Dynamo,
TensorRT-LLM, MAX, llm-d), but every one of those assumes prefill and decode workers own
*separate GPUs* with *separate model copies* and a KV connector between them. One Q4 copy
plus a 224K q8_0 KV cache already fills both 16 GB cards here, so there is no second set of
GPUs to give the decode worker.

pd-proxy does the same thing in time instead of space: it runs **one** llama-server, and
when it wants the other split mode it saves the slot KV, restarts the server in that mode,
and restores the KV.

## Does that actually work?

Yes. `llama_kv_cache::state_write_data` serializes the KV logically — `v_trans`, `n_layer`,
then per layer a type, a row size and cell ranges. Nothing about devices or split layout is
in the file, so a state written by a layer-mode server loads into a tensor-mode server.

Measured (Qwen3.8-27B Q4_K_M, q8_0/q8_0 KV, MTP n=3, `-c 40960`):

```
layer   prefill 32768 tok @ 1057.7 t/s   (31.0 s)
save    32768 tok / 1239 MiB / 556 ms    (to /dev/shm)
        stop layer server, start tensor server          7 s
restore 32768 tok / 694 ms
decode  next request -> prompt_n = 17 (only the new tokens), tg = 39.57 t/s
```

Tensor mode would have needed ~81 s to prefill those same 32K tokens. The whole switch cost
8.3 s.

At the real working point (`-c 229376`, a 229000-token prompt):

```
layer   prefill 229000 tok @ 494.6 t/s  (463 s)     VRAM 14848 / 14907 MiB
save    229000 tok / 7759 MiB / 3.41 s
        stop layer server, start tensor server          8 s
restore 229000 tok / 4.58 s
decode  next request -> prompt_n = 17, tg = 19.03 t/s   VRAM 14028 / 14029 MiB
```

A switch costs **16 s** end to end. Decode goes from 15.3 t/s (layer mode, the profile this
box ran in production) to **19.03 t/s, +24%**, while prefill keeps layer mode's 463 s
instead of the 700–850 s tensor mode would need. Break-even is
`16 / (1/15.3 - 1/19.03)` ≈ **1250 generated tokens** — one turn of a thinking agent pays it
back many times over.

### The gotcha that makes it look broken

llama-server re-processes the entire prompt after a restore **if the incoming prompt is a
strict prefix of the restored state**. In `tools/server/server-context.cpp` the resume path
computes

```c
const bool has_new_tokens = (n_past < slot.task->n_tokens());
const auto pos_min_thold  = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));
```

With no new tokens the threshold drops by one, the checkpoint search that follows finds
nothing (a restored slot carries no checkpoints), and the server logs

```
forcing full prompt re-processing due to lack of cache data
```

So the prefill phase deliberately stops one token short: it prefills `tokens[:-1]` and lets
the decode request supply the last token. With that, restores are reused in full.

A second, separate trap: when the RAM prompt cache (`--cache-ram`, on by default) is
enabled, `get_available_slot` calls `prompt_save()` then `prompt_load()`, and a cache miss
runs `prompt_clear()` → `mem.seq_rm(id, -1, -1)`, which wipes a manually restored slot
(`clearing prompt with N tokens` in the log). pd-proxy sidesteps both by always leaving at
least one new token and by not relying on the RAM cache across a switch.

Upstream has related open issues — #28619 (`/slots` save/restore never persists the draft
model's `ctx_dft`, so the MTP draft comes back cold), #25913, #21831 — but the target KV
does survive.

## Policy

A switch is a process restart plus two KV copies: 8 s at 32K, 16 s at 224K. Prefill in
layer mode saves `N * (1/331 - 1/663)` ≈ `N * 0.0015` seconds for `N` new tokens, and a
round trip costs two switches, so borrowing layer mode only pays off above roughly **22K new
tokens**. Below that, staying in tensor mode and eating the slower prefill is cheaper — and
decode, which dominates an agent session with a 10–16K thinking budget, is 30% faster there.

Hence: **live in tensor (decode) mode; borrow layer (prefill) mode only for a prefill larger
than `prefill_threshold`.** The one place this really pays is the first load of a large
context.

## Usage

```
python3 pd_proxy.py --config config.json          # listens on 127.0.0.1:8099
python3 pd_proxy.py --threshold 8000 --port 8080  # override
```

`config.json` holds the model, the llama-server binary, the arguments shared by both modes,
and the two mode profiles (`modes.prefill`, `modes.decode`). pd_proxy starts and stops the
llama-server itself — do not run one separately on `upstream_port`.

Point an OpenAI-compatible client at `http://127.0.0.1:<listen_port>/v1`. `/v1/chat/completions`
goes through the policy above; everything else is proxied straight through to whichever
server is up.

`systemd/qwen-agent-pd.service` is an example user unit.

## Limitations

- One slot, one session (`-np 1`). Two clients with different conversations will fight over
  the slot and force re-prefills.
- A switch takes the server down for its duration; requests in flight are not migrated.
- The MTP draft context is not saved (upstream #28619), so the draft restarts cold after a
  switch: draft acceptance right after the 224K restore was 0.507 against a usual 0.68–0.70.
  It warms back up during a long generation, and decode was still 19.03 t/s with it cold.
- The state file is ~35 KiB per token (q8_0 KV): 1.2 GiB at 32K, 7.6 GiB at 224K. It lives
  in `/dev/shm`, so it costs RAM, not SSD writes.
- Tested against llama.cpp build 11045 (`2b1847030`). The resume logic quoted above is not a
  stable interface.
