import torch
import torch.distributed as dist
from task import input_t, output_t


# ---------------- All2All optimized PyTorch impl (vectorized, buffer reuse) ----------------
class PyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size

        # persistent buffers
        device = torch.device(f"cuda:{rank}")
        self.device = device
        # local maximum send tokens = tokens on this rank * k
        self.max_local_send = cfg.max_num_tokens * cfg.experts_per_token
        self._send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        self._recv_counts = torch.empty(world_size, dtype=torch.long, device=device)

        self.rank_i32 = torch.tensor(rank, dtype=torch.int32, device=device)

        # Ones for scatter_add count
        self._ones_full = torch.ones(self.max_local_send, dtype=torch.long, device=device)
        # Prealloc meta/send buffers (capacity only; we slice per use)
        self._send_meta_full = torch.empty(self.max_local_send, self.META_DIM, dtype=torch.int32, device=device)
        self._send_buf_full = torch.empty(self.max_local_send, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)

        # Prealloc recv buffers for dispatch/combine to avoid per-call allocations
        self._max_total_recv = cfg.max_num_tokens * cfg.experts_per_token * world_size
        self._recv_buf_in_full = torch.empty(self._max_total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        self._recv_meta_full = torch.empty(self._max_total_recv, self.META_DIM, dtype=torch.int32, device=device)
        self._recv_buf_out_full = torch.empty(self._max_total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        num_tokens = dp_x.shape[0]
        k = indices.shape[1]

        flat_experts = indices.reshape(-1)
        dst_ranks = (flat_experts // self.num_local_experts).to(torch.int64)

        # flat length
        M = num_tokens * k

        # counts per rank (GPU bincount is typically faster)
        self._send_counts.zero_()
        self._send_counts.scatter_add_(0, dst_ranks, self._ones_full[:M])
        dist.all_to_all_single(self._recv_counts, self._send_counts)

        # Single global argsort is typically faster than per-bucket nonzero on GPU
        order = torch.argsort(dst_ranks)

        # Derive token/k via integer divmod
        sorted_tokens = order // k
        sorted_ks = (order % k).to(torch.int32)
        sorted_experts = flat_experts.gather(0, order).to(torch.int32)

        nsend = order.numel()
        # Use direct gather result as send buffer to avoid extra copy
        send_buf = dp_x.index_select(0, sorted_tokens)

        send_meta = self._send_meta_full[:nsend]
        send_meta[:, 0] = sorted_experts
        send_meta[:, 1] = self.rank_i32.expand(nsend)
        send_meta[:, 2] = sorted_tokens.to(torch.int32)
        send_meta[:, 3] = sorted_ks
        send_meta[:, 4].zero_()

        total_recv = int(self._recv_counts.sum().item())
        recv_buf = self._recv_buf_in_full[:total_recv]
        recv_meta = self._recv_meta_full[:total_recv]

        recv_sizes = self._recv_counts.tolist()
        send_sizes = self._send_counts.tolist()

        # Synchronous all_to_all tends to be more predictable here
        dist.all_to_all_single(recv_buf, send_buf, output_split_sizes=recv_sizes, input_split_sizes=send_sizes)
        dist.all_to_all_single(
            recv_meta.view(-1), send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_sizes],
            input_split_sizes=[c * self.META_DIM for c in send_sizes],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # Return flattened representation; aggregate later in combine to reduce intermediate reordering
        return recv_buf, recv_meta

    # ---------- combine ----------
    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, recv_meta: torch.Tensor, expert_y_flat: torch.Tensor):
        device = out_tokens.device
        cfg = self.cfg

        dst_ranks = recv_meta[:, 1].to(torch.int64)
        self._send_counts.zero_()
        if dst_ranks.numel() > 0:
            if dst_ranks.numel() <= self._ones_full.numel():
                ones = self._ones_full[: dst_ranks.numel()]
            else:
                ones = torch.ones_like(dst_ranks, dtype=torch.long, device=device)
            self._send_counts.scatter_add_(0, dst_ranks, ones)
        dist.all_to_all_single(self._recv_counts, self._send_counts)

        order = torch.argsort(dst_ranks) if dst_ranks.numel() > 1 else None
        if order is not None:
            send_buf = expert_y_flat.index_select(0, order)
            send_meta = recv_meta.index_select(0, order)
        else:
            send_buf = expert_y_flat
            send_meta = recv_meta

        total_recv = int(self._recv_counts.sum().item())
        recv_buf = self._recv_buf_out_full[:total_recv]
        recv_meta_out = self._recv_meta_full[:total_recv]

        recv_sizes = self._recv_counts.tolist()
        send_sizes = self._send_counts.tolist()

        dist.all_to_all_single(
            recv_buf,
            send_buf.contiguous(),
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
        )
        dist.all_to_all_single(
            recv_meta_out.view(-1),
            send_meta.contiguous().view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_sizes],
            input_split_sizes=[c * self.META_DIM for c in send_sizes],
        )
        recv_meta = recv_meta_out.view(-1, self.META_DIM)

        if recv_buf.numel() > 0:
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            w = weights[src_tokens, src_ks]
            contrib = recv_buf.to(torch.float32) * w.to(torch.float32).unsqueeze(1)
            out_tokens.index_add_(0, src_tokens, contrib.to(cfg.out_dtype))

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    with torch.inference_mode():
        ata = PyTorchAllToAll(cfg, rank, world_size)
        recv_buf, recv_meta = ata.dispatch(rank_data.x, rank_data.indices)

        scale = 1 + rank
        if cfg.out_dtype == recv_buf.dtype:
            expert_y = recv_buf * scale
        else:
            expert_y = recv_buf.to(cfg.out_dtype) * scale

        y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=rank_data.x.device)
        ata.combine(y, rank_data.weights, recv_meta, expert_y)
        return y[: rank_data.num_tokens]