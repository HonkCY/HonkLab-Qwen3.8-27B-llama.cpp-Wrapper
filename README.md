# pd-proxy

Runs one `llama-server` and switches its split mode between prefill and decode, carrying the
KV cache across the restart. Built for a box whose two GPUs are good at opposite halves of
inference and cannot both hold a model copy.

> **Status: not deployed.** On the machine it was built for, the owner moved back to plain
> `-sm layer` because tensor mode keeps both GPUs at 100% at once (108/126 W) and kept
> triggering fan ramps. With decode in layer mode too, this proxy's only remaining benefit is
> prefilling without MTP: 285 s versus 384 s at 200K, minus ~30 s of switching — 69 s saved,
> and only when a context is filled from scratch. Not worth an extra process that takes the
> server down for 15 s per switch. Kept here as a measured result, and because the KV-across-
> split-modes mechanism below is reusable.


## What it buys

Qwen3.8-27B Q4_K_M, q8_0/q8_0 KV, MTP n=3 for decode, at the model's full 262144 context:

| | prefill | TTFT (262K prompt) | decode |
|---|---|---|---|
| `-sm layer` + MTP n=3 | not measured at 262K | — | not measured at 262K |
| `-sm tensor` + MTP n=3 | 267 t/s | **978 s** | **19.83 t/s** |
| **pd-proxy** (layer prefill → tensor decode) | **545.6 t/s** | **498 s** | **18.86 t/s** |

**Gain: TTFT drops 49% (978 s → 498 s) while decode keeps 95% of tensor mode's speed.**

Against the layer-mode profile this machine used to run in production, measured at 224K
where that profile fits comfortably:

| 224K | prefill | TTFT | decode |
|---|---|---|---|
| `-sm layer -ts 1,0.8` + MTP n=3 | 494.6 t/s | 463 s | 15.27 t/s |
| **pd-proxy at 224K** | **494.6 t/s** (same layer prefill) | 480 s | **19.03 t/s** |

**Gain: +25% decode for 17 s of switching, at identical prefill speed.**

The two modes measured head to head at 128K, which is where the trade-off is clearest:

| 128K, MTP n=3 | prefill | decode |
|---|---|---|
| `-sm layer -ts 1,0.9` | **663.3 t/s** | 21.20 t/s |
| `-sm tensor -ts 1,1` | 330.6 t/s | **28.16 t/s** |

Layer prefills 2.0x faster. Tensor decodes 1.33x faster. pd-proxy takes both.

### Switch cost

| context | save | restart | restore | total |
|---|---|---|---|---|
| 32K | 0.56 s | 7 s | 0.69 s | **8.3 s** |
| 224K | 4.30 s | 8 s | 5.21 s | **17.5 s** |
| 262K | 4.30 s | 8 s | 5.21 s | **17.5 s** |

Break-even against just prefilling in tensor mode: `2 x 17.5 / (1/267 - 1/546)` ≈ **18000 new
tokens**, which is the default `prefill_threshold`. Below it the proxy stays in decode mode.

### Production traffic, first hour

| request | new tokens | what happened |
|---|---|---|
| short turn | 17 | stayed in decode, answered in 1.0 s |
| medium turn | 1 402 | stayed in decode |
| document | 34 704 | borrowed layer: **1389 t/s**, 25 s prefill, 44.5 s total (≈86 s if tensor had prefilled it) |
| restored session | 189 520 | borrowed layer: **683 t/s**, 278 s prefill (≈710 s if tensor had prefilled it) |

## Why the machine needs this

| | |
|---|---|
| CPU / RAM | AMD Ryzen 5 9600X, 64 GB DDR5, `/dev/shm` 30 GB |
| GPU A | RTX 5060 Ti 16 GB — CPU slot, PCIe 5.0 x8, 448 GB/s |
| GPU B | RTX 4060 Ti 16 GB — chipset slot, **link runs at x1** (`LnkCap 16GT/s x8`, `LnkSta 2.5GT/s (downgraded), Width x1 (downgraded)`), 288 GB/s |
| Interconnect | `PHB`, no NVLink, **GeForce P2P unsupported** (`nvidia-smi topo -p2p r` → `NS`) — every cross-GPU byte goes GPU → host RAM → GPU |
| Display | on the CPU iGPU, so neither card loses VRAM |
| Software | driver 595.91.07, CUDA 13.2, llama.cpp build 11045 (`2b1847030`), `CUDA_SCALE_LAUNCH_QUEUES=4x` |

1. **32 GB of VRAM in two 16 GB pieces.** Model + 262K KV = ~14.8 GB on each card. No room
   for a second copy, and no single card can hold the model.
2. **x1 link on the second card.** Tensor-parallel all-reduce is throttled to ~1.7 GB/s round
   trip, which is why tensor mode prefills at half of layer mode's speed.
3. **No P2P**, so that traffic is copied through host RAM on top.

This rules out the standard answer — P/D disaggregation (vLLM, SGLang, NVIDIA Dynamo,
TensorRT-LLM, MAX, llm-d) — because all of those give prefill and decode their own GPUs and
their own model copy. pd-proxy splits the phases in time instead of space.

One more consequence of (2): llama.cpp enumerates these cards as `CUDA0 = 4060 Ti` and
`CUDA1 = 5060 Ti`, the reverse of nvidia-smi. Leaving the x1 card first in the pipeline sends
operation-offload traffic across the slow link — passing `-dev CUDA1,CUDA0` was worth **6.3x**
on a prefill-heavy workload. Every profile passes it, and `-ts` follows that order, so
`-ts 1,0.9` means 5060 Ti = 1, 4060 Ti = 0.9.

## How

The KV state file is layout-independent: `llama_kv_cache::state_write_data` writes `v_trans`,
`n_layer`, then per layer a type, a row size and cell ranges — nothing about devices or
splits. So a state written by a layer-mode server loads into a tensor-mode server.

Per request: render the prompt, count how many tokens are new, and if that exceeds
`prefill_threshold`, save the slot → restart in the prefill profile → prefill `tokens[:-1]` →
save → restart in the decode profile → restore → stream the answer. Otherwise answer straight
from the decode profile.

## Tuning results

**1. For layer prefill, balance beats favouring the fast card.** With pipeline parallelism on,
throughput is `1 / max(stage)`, not `1 / sum`:

| `-ts` (5060,4060) | prefill 32768 tok |
|---|---|
| 1,1 | 1396.6 t/s |
| **1,0.9** | **1396.7 t/s** |
| 1,0.8 | 1334.5 t/s |
| 1,0.7 | 1224.5 t/s |

The opposite of the decode side, where the slow card sets the ceiling.

**2. The prefill profile must not load speculative decoding.** MTP costs ~18% of prefill
throughput and ~1.4 GB of VRAM, and only helps decode. Dropping it is what leaves room for
pipeline parallelism at 262144:

| 262144, layer | pipeline parallelism | free VRAM (5060/4060) |
|---|---|---|
| MTP n=3, `-ts 1,0.8` | on | 607 / 501 MiB |
| MTP n=3, `-ts 1,0.85` / `1,0.75` / `1,0.7` | **fell back** | 827–1165 / 1187–1781 MiB |
| **no speculation, `-ts 1,0.9`** | on | **1819 / 2235 MiB** |
| no speculation, `-ts 1,1` | on | 2273 / 1783 MiB |

**3. A state saved without MTP restores into a server with MTP.** Save under
`--spec-type none`, restore under `--spec-type draft-mtp`: `prompt_n = 1` on the next request.
The draft context is never in the state file anyway (upstream #28619). This is what lets the
two phases disagree about speculation.

**4. Decode must use MTP, not DFlash2.** DFlash2 looked competitive in layer mode at 128K
(21.63 t/s vs MTP n=3's 21.20) but **cannot run under `-sm tensor` at all**:

```
ggml/src/ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
```

That is the axis tensor mode splits along. It fires at 131072 with 3.6/4.2 GB free, so it is
an architectural incompatibility, not a memory limit — the same class as `--override-tensor`
under `-sm tensor`. Restoring the same 262K state into tensor+MTP gave 18.86 t/s with
1525/1617 MiB free, confirming VRAM was never the constraint.

## Two llama.cpp behaviours this works around

**A restored slot is thrown away if the new prompt is a strict prefix of it.** In
`tools/server/server-context.cpp`:

```c
const bool has_new_tokens = (n_past < slot.task->n_tokens());
const auto pos_min_thold  = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));
```

With no new tokens the threshold drops by one, the checkpoint search that follows finds
nothing (a restored slot carries no checkpoints), and the server logs `forcing full prompt
re-processing due to lack of cache data`. Hence prefilling `tokens[:-1]` and letting the
decode request contribute the last token.

**The RAM prompt cache wipes a manually restored slot.** With `--cache-ram` on (the default),
`get_available_slot` runs `prompt_save()` then `prompt_load()`, and a cache miss calls
`prompt_clear()` → `mem.seq_rm(id, -1, -1)` — `clearing prompt with N tokens` in the log.

Related open upstream issues: #28619, #25913, #21831.

## Usage

```
python3 pd_proxy.py --config config.json          # 127.0.0.1:8080
python3 pd_proxy.py --threshold 8000 --port 8099  # override
```

`config.json` carries a placeholder model path. For a real deployment copy it to `local.json`
(gitignored), point that at your model, and run `--config local.json`. pd_proxy starts and
stops the llama-server itself — do not run one separately on `upstream_port`. Point any
OpenAI-compatible client at `http://127.0.0.1:<listen_port>/v1`.
`systemd/qwen-agent-pd.service` is an example user unit.

## Limits

- One slot, one session (`-np 1`). Two clients with different conversations will evict each
  other.
- During a switch the server is down and the client sees no bytes — up to 5 minutes for a
  189K prefill. Clients need a generous idle timeout.
- A restart mid-prefill loses that prefill entirely.
- The MTP draft context is not saved (#28619), so the draft is cold right after a switch:
  acceptance 0.507 against a usual 0.68–0.70, recovering over a long generation.
- State file is ~35 KiB/token: 1.2 GiB at 32K, 8.9 GiB at 262K, in tmpfs. Deleted after each
  successful restore.
- Tested against llama.cpp build 11045 (`2b1847030`). The resume logic quoted above is not a
  stable interface.
