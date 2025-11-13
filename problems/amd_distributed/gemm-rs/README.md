### GEMM + ReduceScatter (RS) — version history

- `submission*.py` are benchmark entries. Each `vX` iterates on the previous one with noticeable latency wins; decimals (e.g. `v1.1`) indicate micro-optimizations on top of that base.

### TL;DR
- submission.py: plain `matmul` + optional bias add, then `reduce_scatter_tensor`.
- v1: fuse GEMM+bias via `F.linear`, reshape for RS to avoid extra slicing.
- v2: same core as v1 with minor cleanup and explicit `SUM` op.

---

### submission v1.py
- Fuses GEMM and bias add using `torch.nn.functional.linear` (usually picks a faster rocBLAS kernel).
- Shapes the RS input as `[world_size, M // world_size, N]` and calls `reduce_scatter_tensor` directly to avoid per-chunk slicing.
- Keeps the output contiguous for downstream ops.

### submission v2.py
- Small polish on v1:
  - Explicit `op=ReduceOp.SUM` for clarity.
  - Same view/contiguous pattern for predictable RS throughput.
- Equivalent math; tiny wins from reduced ambiguity and cleaner hot path.

### Notes
- On MI300X/ROCm, chunk-list RS (e.g., `torch.distributed.reduce_scatter([...chunks...])`) tends to be more stable than `reduce_scatter_tensor`, and is often faster for small batch sizes; we default to that path in production [[memory:9119814]].
- For very small M (e.g., M ≤ 128) with large `world_size`, `reduce_scatter_tensor` can occasionally time out in some environments; a robust fallback is `all_reduce` followed by local slicing, which is semantically equivalent [[memory:9118904]].


