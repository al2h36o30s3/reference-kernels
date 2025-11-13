import torch
import torch.distributed as dist
from task import input_t, output_t


# ---------------- All2All pytorch impl ----------------
class PyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # num experts per rank
        self.num_local_experts = cfg.num_experts // world_size
        # max recv tokens per rank
        self.max_recv = cfg.max_num_tokens * world_size
        
        # Memory optimization: pre-allocate reusable buffers to reduce allocation overhead
        self.device = torch.device(f"cuda:{rank}")
        max_total_tokens = cfg.max_num_tokens * cfg.experts_per_token * world_size
        
        # Persistent buffers for sorting and communication
        self._sort_buffer = torch.empty(max_total_tokens, dtype=torch.int64, device=self.device)
        self._send_counts = torch.zeros(world_size, dtype=torch.long, device=self.device)
        self._recv_counts = torch.empty(world_size, dtype=torch.long, device=self.device)
        
        # Constants for reducing CPU-GPU sync
        self.rank_tensor = torch.tensor(self.rank, dtype=torch.int32, device=self.device)
        self.world_size_tensor = torch.tensor(world_size, dtype=torch.long, device=self.device)

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        # 1) Optimized flatten: minimize allocations and use fused operations
        num_tokens = dp_x.shape[0]
        experts_per_token = indices.shape[1]
        
        # Kernel fusion: combine reshape, type conversion and division in one pass
        flat_experts = indices.reshape(-1)
        dst_ranks = flat_experts.div(self.num_local_experts, rounding_mode='floor').to(torch.int64)
        
        # Generate indices more efficiently - use views when possible
        t_ids = torch.arange(num_tokens, device=device, dtype=torch.int32).repeat_interleave(experts_per_token)
        k_ids = torch.arange(experts_per_token, device=device, dtype=torch.int32).repeat(num_tokens)

        # 2) Reuse buffers for communication counts
        self._send_counts.zero_()
        self._send_counts.scatter_add_(0, dst_ranks, torch.ones_like(dst_ranks, dtype=torch.long))
        dist.all_to_all_single(self._recv_counts, self._send_counts)

        # 3) Efficient sorting and memory access pattern
        order = torch.argsort(dst_ranks, stable=True)
            
        # Optimize memory access: gather all required data in one pass
        sorted_tokens = t_ids.gather(0, order).to(torch.long)
        sorted_ks = k_ids.gather(0, order)
        sorted_experts = flat_experts.gather(0, order).to(torch.int32)
        
        # Coalesced memory access for send buffer
        send_buf = dp_x.index_select(0, sorted_tokens)

        # Memory-efficient meta construction: pre-allocate and fill
        nsend = order.shape[0]
        send_meta = torch.empty(nsend, self.META_DIM, dtype=torch.int32, device=device)
        send_meta[:, 0] = sorted_experts
        send_meta[:, 1] = self.rank_tensor.expand(nsend)
        send_meta[:, 2] = sorted_tokens.to(torch.int32)
        send_meta[:, 3] = sorted_ks
        send_meta[:, 4].zero_()

        # 4) Reduce CPU-GPU sync: avoid .item() calls
        total_recv = self._recv_counts.sum()
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)
        
        # Use pre-computed sizes to avoid repeated tensor operations
        recv_sizes = self._recv_counts.tolist()
        send_sizes = self._send_counts.tolist()
        
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
        )
        
        recv_meta_sizes = [c * self.META_DIM for c in recv_sizes]
        send_meta_sizes = [c * self.META_DIM for c in send_sizes]
        
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=recv_meta_sizes,
            input_split_sizes=send_meta_sizes,
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # Dynamic allocation: directly return the flattened receive buffer
        return recv_buf, recv_meta

    # ---------- combine ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight
        recv_meta: torch.Tensor,  # flat meta
        expert_y_flat: torch.Tensor,  # flat expert outputs
    ):  # input
        device = out_tokens.device
        cfg = self.cfg

        # 1) Optimized combine with memory reuse and fused operations
        flat_y = expert_y_flat
        flat_meta = recv_meta
        
        # Reuse communication buffers
        dst_ranks = flat_meta[:, 1].to(torch.int64)
        self._send_counts.zero_()
        self._send_counts.scatter_add_(0, dst_ranks, torch.ones_like(dst_ranks, dtype=torch.long))
        dist.all_to_all_single(self._recv_counts, self._send_counts)

        # Efficient sorting with memory coalescing
        order = torch.argsort(dst_ranks, stable=True)
        send_buf = flat_y.gather(0, order.unsqueeze(1).expand(-1, flat_y.shape[1]))
        send_meta = flat_meta.gather(0, order.unsqueeze(1).expand(-1, self.META_DIM))

        # Reduce CPU-GPU sync in combine as well
        total_recv = self._recv_counts.sum()
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # Reuse computed sizes
        recv_sizes = self._recv_counts.tolist()
        send_sizes = self._send_counts.tolist()

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
        )
        
        recv_meta_sizes = [c * self.META_DIM for c in recv_sizes]
        send_meta_sizes = [c * self.META_DIM for c in send_sizes]
        
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=recv_meta_sizes,
            input_split_sizes=send_meta_sizes,
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 3. Optimized weighted aggregation with kernel fusion
        if recv_buf.numel() > 0:
            # Fuse indexing operations
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            
            # Direct weighted contribution computation
            w = weights[src_tokens, src_ks]
            if cfg.out_dtype != torch.float32:
                # Minimize dtype conversions - compute in float32 when needed
                contrib = recv_buf.to(torch.float32) * w.to(torch.float32).unsqueeze(1)
                # Use in-place operations where possible
                out_tokens.index_add_(0, src_tokens, contrib.to(cfg.out_dtype))
            else:
                contrib = recv_buf * w.unsqueeze(1)
                out_tokens.index_add_(0, src_tokens, contrib)

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    
    # Kernel fusion optimization: combine expert computation with scale
    with torch.inference_mode():
        ata = PyTorchAllToAll(cfg, rank, world_size)
        recv_buf, recv_meta = ata.dispatch(rank_data.x, rank_data.indices)
        
        # Fused type conversion and scaling - avoid intermediate tensors
        scale_factor = 1 + rank
        if cfg.out_dtype == recv_buf.dtype:
            expert_y = recv_buf * scale_factor
        else:
            # Single-pass type conversion and scaling
            expert_y = recv_buf.to(cfg.out_dtype) * scale_factor
        
        # Pre-allocate output with optimal memory layout
        y = torch.zeros(
            cfg.max_num_tokens,
            cfg.hidden_dim,
            dtype=cfg.out_dtype,
            device=rank_data.x.device
        )
        
        ata.combine(y, rank_data.weights, recv_meta, expert_y)
        
        # Avoid unnecessary copy if possible
        if rank_data.num_tokens == cfg.max_num_tokens:
            return y
        else:
            return y[:rank_data.num_tokens].contiguous()