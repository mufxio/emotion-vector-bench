# Hardware Notes

Practical gotchas when running this on a Mac (M-series with MPS). Documenting because we hit each of these and they cost time.

## MPS allocator warmup must be disabled

HuggingFace transformers' `caching_allocator_warmup` tries to allocate one giant buffer at model load time. For a 7B model in fp16, that's ~12GB in a single allocation, which exceeds Apple's MPS single-buffer limit and crashes:

```
RuntimeError: Invalid buffer size: 12.30 GiB
```

Workaround (do this BEFORE importing anything that triggers transformers loading):

```python
import transformers.modeling_utils as _modeling_utils
_modeling_utils.caching_allocator_warmup = lambda *a, **kw: None
```

Already done in `extract.py`.

## `device_map="auto"` silently offloads to disk

When memory is tight, `device_map="auto"` decides to offload some layers to disk. This makes forward passes ~10x slower without any error. Fix:

```python
model = LanguageModel(name, device_map={"": "mps"}, ...)
```

Forces the entire model onto MPS. If it doesn't fit, you'll get a clean OOM instead of silent disk offloading.

## nnsight returns 2D tensors for single-input traces

When tracing a single string, nnsight strips the batch dimension. Activations come back as `[seq_len, hidden_dim]`, not `[1, seq_len, hidden_dim]`. Index accordingly.

## Dict comprehensions inside `model.trace()` don't bind

This silently fails:
```python
with model.trace(text):
    saved = {l: model.model.layers[l].output[0].save() for l in layers}
```

`saved` ends up undefined after the with block exits. Use an explicit loop:
```python
with model.trace(text):
    saved = {}
    for l in layers:
        saved[l] = model.model.layers[l].output[0].save()
```

## MPS allocator fragmentation under sustained load

After ~thousand forward passes, throughput degrades from ~1.2 stories/sec to ~0.05 stories/sec. The allocator gets fragmented and spends more time managing memory than computing.

Workaround: call `torch.mps.empty_cache()` periodically (every N stories). Already done in `extract.py` every 50 stories. Maintains stable rate.

## Checkpoint everything

For a 3000-story extraction at ~1 story/sec, even a brief interruption is costly. `extract.py` saves a checkpoint every 50 stories with the `.npz.tmp.npz` filename collision avoided (numpy auto-appends `.npz` if missing — be careful).

## Process backgrounding gotcha

If you launch via `nohup python script.py > log.txt &`, Python buffers stdout when the descriptor isn't a TTY. Progress lines won't appear in the log until the buffer flushes (every ~4KB).

Either:
- Use `python -u` (unbuffered): `nohup python -u script.py > log.txt &`
- Or call `flush=True` in every print: `print(..., flush=True)`

`extract.py` does both.

## Close Ollama before running

Ollama camps on RAM even when no model is actively loaded. We saw it holding 9GB on idle. `killall Ollama` before extraction.

## Memory budget reference (M4 Pro 24GB)

Realistic limits for fitting on a Mac mini:

| Model class | fp16 | 4-bit |
|---|---|---|
| 1.5B-3B | ✅ | ✅ |
| 7B-9B | ✅ tight | ✅ |
| 13B-14B | ❌ won't fit | ✅ |
| 27B-32B | ❌ | ✅ |
| 70B+ | ❌ | ❌ |

For 13B+ models, use `--quantize 4bit` flag in `extract.py` (requires `bitsandbytes`).
