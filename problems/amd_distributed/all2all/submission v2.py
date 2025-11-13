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
        # dynamic allocation path: no large persistent buffers

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        # 1) 平铺 (token, k)
        num_tokens = dp_x.shape[0]
        experts_per_token = indices.shape[1]
        t_ids = torch.arange(num_tokens, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, experts_per_token).reshape(-1)
        k_ids = torch.arange(experts_per_token, device=device, dtype=torch.int32).unsqueeze(0).expand(num_tokens, -1).reshape(-1)
        flat_experts = indices.reshape(-1).to(torch.int32)
        dst_ranks = (flat_experts // self.num_local_experts).to(torch.int64)

        # 2) 每个 rank 发送计数 + 交换接收计数
        send_counts_t = torch.bincount(dst_ranks, minlength=self.world_size).to(torch.long)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)

        # 3) 使用全局稳定排序构造连续发送缓冲
        order = torch.argsort(dst_ranks, stable=True)
        sorted_tokens = t_ids[order].to(torch.long)
        sorted_ks = k_ids[order]
        sorted_experts = flat_experts[order]
        send_buf = dp_x.index_select(0, sorted_tokens)

        send_meta = torch.stack([
            sorted_experts,
            torch.full_like(sorted_experts, self.rank, dtype=torch.int32),
            sorted_tokens.to(torch.int32),
            sorted_ks,
            torch.zeros_like(sorted_experts, dtype=torch.int32),
        ], dim=1).to(torch.int32)

        # 4) all2all（同步）
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
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

        # 1) 计算目的 rank 并进行 all2all（按 rank 稳定排序）
        flat_y = expert_y_flat
        flat_meta = recv_meta
        dst_ranks = flat_meta[:, 1].to(torch.int64)
        send_counts_t = torch.bincount(dst_ranks, minlength=self.world_size).to(torch.long)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)

        order = torch.argsort(dst_ranks, stable=True)
        send_buf = flat_y.index_select(0, order)
        send_meta = flat_meta.index_select(0, order)

        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 3. 使用 index_add_ 向量化回写并做加权求和
        if recv_buf.shape[0] > 0:
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            w = weights[src_tokens, src_ks].to(torch.float32)
            contrib = recv_buf.to(torch.float32) * w.unsqueeze(1)
            accum = torch.zeros(out_tokens.shape, dtype=torch.float32, device=device)
            accum.index_add_(0, src_tokens, contrib)
            out_tokens.add_(accum.to(out_tokens.dtype))

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    ata = PyTorchAllToAll(cfg, rank, world_size)

    with torch.inference_mode():
        recv_buf, recv_meta = ata.dispatch(rank_data.x, rank_data.indices)
        expert_y = recv_buf.to(cfg.out_dtype) * (1 + rank)
        y = torch.zeros(
            cfg.max_num_tokens,
            cfg.hidden_dim,
            dtype=cfg.out_dtype,
            device=rank_data.x.device,
        )
        ata.combine(y, rank_data.weights, recv_meta, expert_y)
        return y[: rank_data.num_tokens]
