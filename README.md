# pd-proxy

Prefill/decode split for a single `llama-server` on two mismatched GPUs.

## The machine this exists for

| | |
|---|---|
| CPU | AMD Ryzen 5 9600X (6C/12T) |
| RAM | 64 GB DDR5 (62.3 GB usable), `/dev/shm` 30 GB |
| GPU A | **RTX 5060 Ti 16 GB** — CPU slot, PCIe 5.0 x8 (`LnkCap 32GT/s Width x8`), 448 GB/s |
| GPU B | **RTX 4060 Ti 16 GB** — chipset slot, **link downgraded to x1** (`LnkCap 16GT/s x8`, `LnkSta 2.5GT/s (downgraded), Width x1 (downgraded)`), 288 GB/s |
| Interconnect | `PHB` — both cards reach each other through the CPU host bridge. No NVLink, and GeForce P2P is unsupported (`nvidia-smi topo -p2p r` → `NS`), so **every cross-GPU byte goes through host RAM** |
| Display | driven by the CPU's iGPU, so neither card loses VRAM to the desktop |
| Software | driver 595.91.07 / CUDA 13.2, llama.cpp build 11045 (`2b1847030`), `CUDA_SCALE_LAUNCH_QUEUES=4x` |
| Model | Qwen3.8-27B Q4_K_M with an embedded MTP layer, 262144 native context, q8_0/q8_0 KV |

Three constraints drive everything here:

1. **32 GB total VRAM, in two 16 GB pieces.** One Q4 copy of the model plus a 262K q8_0 KV
   cache fills both cards (~14.8 GB each). There is no room for a second model copy, and no
   single card can hold the model alone.
2. **The second card is on a PCIe x1 link.** Anything that streams data to it — tensor-
   parallel all-reduce, CPU-expert offload — is throttled to ~1.7 GB/s round trip.
3. **No P2P.** Cross-GPU traffic is copied GPU → host RAM → GPU.

Constraint 2 has a non-obvious consequence: llama.cpp enumerates these cards as
`CUDA0 = 4060 Ti` and `CUDA1 = 5060 Ti`, the reverse of nvidia-smi. Leaving the x1 card
first in the pipeline puts operation-offload traffic across the slow link; passing
`-dev CUDA1,CUDA0` to put the fast card first was worth **6.3x** on a prefill-heavy
workload during earlier benchmarking. Every profile here passes it, and `-ts` values follow
that order, so `-ts 1,0.9` means 5060 Ti = 1, 4060 Ti = 0.9.

## The problem

Given that hardware, the two llama.cpp split modes are good at opposite things:

| Qwen3.8-27B Q4_K_M, q8_0 KV, 262144 ctx | prefill | decode |
|---|---|---|
| `-sm layer -ts 1,0.9` (no speculation) | **545.6 t/s** | — |
| `-sm tensor -ts 1,1` + MTP n=3 | 267 t/s | **18.9–19.8 t/s** |

Layer mode is a pipeline, so prefill runs near the sum of both cards' throughput while
decode waits on the slow card for every token. Tensor mode (TP) splits every tensor, so
decode is bounded by the *slower* card instead of the *sum*, but prefill pays for
all-reduce traffic that has to cross a PCIe x1 link through host memory.

The industry answer is P/D disaggregation (vLLM, SGLang, NVIDIA Dynamo, TensorRT-LLM, MAX,
llm-d), but all of those assume prefill and decode workers own *separate GPUs* with
*separate model copies* and a KV connector between them. One Q4 copy plus a 262K q8_0 KV
cache already fills both 16 GB cards here, so there is no second set of GPUs for a decode
worker.

pd-proxy does the same thing in time instead of space: it runs **one** llama-server, and
when it wants the other split mode it saves the slot KV, restarts the server in that mode,
and restores the KV.

## Does that actually work?

Yes. `llama_kv_cache::state_write_data` serializes the KV logically — `v_trans`, `n_layer`,
then per layer a type, a row size and cell ranges. Nothing about devices or split layout is
in the file, so a state written by a layer-mode server loads into a tensor-mode server.

Full native context, measured end to end (Qwen3.8-27B Q4_K_M, q8_0/q8_0 KV,
`-c 262144`, a 262000-token prompt):

```
prefill server up (layer 1,0.9, no speculation)      7 s    VRAM 14492 / 14145 MiB
prefill 262000 tok @ 545.6 t/s                     480 s
save    262000 tok / 8856 MiB                      4.30 s   (to /dev/shm)
        stop prefill server, start decode server      8 s    VRAM 14786 / 14763 MiB
restore 262000 tok                                 5.21 s
decode  next request -> prompt_n = 17, tg 18.86 t/s         VRAM 14812 / 14801 MiB
```

A switch costs **17.5 s**. Against running tensor mode alone at the same context
(TTFT 978 s, tg 19.83 t/s), the hybrid cuts time-to-first-token roughly in half — 498 s —
and keeps essentially all of TP's decode speed. Against layer mode alone, it keeps layer's
prefill and gains ~25–30% decode.

The same thing at 32K, for scale: prefill 31.0 s, save 556 ms, restart 7 s, restore 694 ms,
and the follow-up request reported `prompt_n = 17` — only the new tokens.

### Three findings the profiles are built on

**1. For layer-mode prefill, balance beats favouring the fast card.** With pipeline
parallelism on, throughput is `1 / max(stage time)`, not `1 / sum`, so an even split wins:

| `-ts` (5060,4060) | prefill 32768 tok |
|---|---|
| 1,1 | 1396.6 t/s |
| **1,0.9** | **1396.7 t/s** |
| 1,0.8 | 1334.5 t/s |
| 1,0.7 | 1224.5 t/s |

This is the opposite of the decode-side intuition, where the slow card is the ceiling.

**2. The prefill server should not load speculative decoding at all.** MTP only helps
decode, costs ~18% of prefill throughput, and takes ~1.4 GB of VRAM. Dropping it at 262144
is what lets the prefill side keep pipeline parallelism with room to spare:

| 262144, layer | PP | free VRAM (5060/4060) |
|---|---|---|
| with MTP n=3, `-ts 1,0.8` | on | 607 / 501 MiB |
| with MTP n=3, `-ts 0.85/0.75/0.7` | **fell back** | 827–1165 / 1187–1781 MiB |
| **no speculation, `-ts 1,0.9`** | **on** | **1819 / 2235 MiB** |

**3. A KV state saved without MTP restores into a server *with* MTP.** Verified: save from
`--spec-type none`, restore into `--spec-type draft-mtp`, next request reported
`prompt_n = 1`. The draft context is not part of the state file either way (upstream
#28619), so nothing is lost.

**4. The decode profile has to use MTP, not DFlash2.** DFlash2 (`--spec-type draft-dflash`,
a separate 1.06 GB draft model) is a reasonable alternative on paper — at 128K in layer mode
it measured 21.63 t/s against MTP n=3's 21.20 — but it **cannot run in tensor mode at all**:

```
ggml/src/ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

`GGML_BACKEND_SPLIT_AXIS_0` is the axis `-sm tensor` splits along. The assert fires even at
131072 with 3.6/4.2 GB of VRAM free, so this is an architectural incompatibility, not a
memory limit — the same class of restriction as `--override-tensor` under `-sm tensor`.
DFlash2 loads fine under `-sm layer`, which is the phase where speculation does not help.
Restoring the same 262K state into a tensor+MTP decode server reproduced tg 18.86 t/s with
1525/1617 MiB free, so VRAM was never the constraint.

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

Related upstream issues, all open: #28619 (`/slots` save/restore never persists the draft
model's `ctx_dft`), #25913, #21831.

## Policy

A switch is a process restart plus two KV copies: 8 s at 32K, 17.5 s at 262K. Prefilling in
layer mode instead of tensor mode saves `N * (1/267 - 1/546)` ≈ `N * 0.0019` seconds for `N`
new tokens, and a round trip costs two switches, so borrowing the prefill profile only pays
off above roughly **18K new tokens**. Below that, staying in decode mode and eating the
slower prefill is cheaper — and decode, which dominates an agent session with a 10–16K
thinking budget, is ~30% faster there.

Hence: **live in decode mode; borrow prefill mode only for a prefill larger than
`prefill_threshold`.** The place this really pays is loading a large context for the first
time.

## Usage

```
python3 pd_proxy.py --config config.json          # listens on 127.0.0.1:8099
python3 pd_proxy.py --threshold 8000 --port 8080  # override
```

`config.json` holds the model, the llama-server binary, the arguments shared by both modes,
and the two mode profiles (`modes.prefill`, `modes.decode`). pd_proxy starts and stops the
llama-server itself — do not run one separately on `upstream_port`.

For a real deployment, copy it to `local.json` (gitignored), point that at your own model
file, and run `--config local.json`.

Point an OpenAI-compatible client at `http://127.0.0.1:<listen_port>/v1`.
`/v1/chat/completions` goes through the policy above; everything else is proxied straight
through to whichever server is up.

`systemd/qwen-agent-pd.service` is an example user unit.

### Verified end to end

A 5749-token chat request through the proxy: it borrowed the prefill profile (6 s), prefilled
at 1120 t/s, saved 340 MiB in 152 ms, switched back (8 s), restored in 177 ms, and the decode
server reported **`prompt eval = 1 token`** — the rendering from `/apply-template` matched
what `/v1/chat/completions` builds, so the prefill was reused in full.

## Limitations

- One slot, one session (`-np 1`). Two clients with different conversations will fight over
  the slot and force re-prefills.
- A switch takes the server down for its duration; requests in flight are not migrated.
- The MTP draft context is not saved (upstream #28619), so the draft restarts cold after a
  switch: draft acceptance right after a 224K restore was 0.507 against a usual 0.68–0.70.
  It warms back up during a long generation.
- The state file is ~35 KiB per token: 1.2 GiB at 32K, 8.9 GiB at 262K. It lives in
  `/dev/shm`, so it costs RAM, not SSD writes.
- Tested against llama.cpp build 11045 (`2b1847030`). The resume logic quoted above is not a
  stable interface.
