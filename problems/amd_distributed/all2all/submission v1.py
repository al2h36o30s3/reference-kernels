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

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        # ---------1. Compute destination rank for each token-k pair (fully vectorized)-----------
        num_tokens = dp_x.shape[0]
        experts_per_token = indices.shape[1]

        t_ids = torch.arange(num_tokens, device=device, dtype=torch.int32).unsqueeze(1).expand(-1, experts_per_token).reshape(-1)
        k_ids = torch.arange(experts_per_token, device=device, dtype=torch.int32).unsqueeze(0).expand(num_tokens, -1).reshape(-1)
        flat_experts = indices.reshape(-1).to(torch.int32)
        dst_ranks = (flat_experts // self.num_local_experts).to(torch.int64)

        # Send counts per rank
        send_counts_t = torch.bincount(dst_ranks, minlength=self.world_size).to(torch.long)

        # Exchange receive counts
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)

        # ---------2. Build send buffer sorted by destination rank ----------
        order = torch.argsort(dst_ranks, stable=True)
        sorted_tokens = t_ids[order].to(torch.long)
        sorted_ks = k_ids[order].to(torch.int32)
        sorted_experts = flat_experts[order]

        send_buf = dp_x.index_select(0, sorted_tokens)

        send_meta = torch.stack([
            sorted_experts,  # global expert id
            torch.full_like(sorted_experts, self.rank, dtype=torch.int32),  # src rank
            sorted_tokens.to(torch.int32),  # src token id
            sorted_ks,  # src k
            torch.zeros_like(sorted_experts, dtype=torch.int32),  # pad
        ], dim=1).to(torch.int32)

        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # ---------3. all2all: send tokens and meta --------------
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

        # ---------4. Bucket received tokens by local expert (small outer loop; remove inner token loop)------------
        local_eids = (recv_meta[:, 0] % self.num_local_experts).to(torch.int64)
        expert_num_tokens = torch.bincount(local_eids, minlength=self.num_local_experts).to(torch.int32)

        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device
        )

        if recv_buf.numel() > 0:
            sort_idx = torch.argsort(local_eids, stable=True)
            sorted_buf = recv_buf.index_select(0, sort_idx)
            sorted_meta = recv_meta.index_select(0, sort_idx)

            counts_long = expert_num_tokens.to(torch.long)
            starts = torch.cumsum(torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts_long[:-1]]), dim=0)
            for e in range(self.num_local_experts):
                cnt = int(counts_long[e].item())
                if cnt == 0:
                    continue
                s = int(starts[e].item())
                expert_x[e, :cnt] = sorted_buf[s:s+cnt]
                expert_meta[e, :cnt] = sorted_meta[s:s+cnt]

        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, token_dim)
        expert_num_tokens: torch.Tensor,
    ):  # input
        device = out_tokens.device
        cfg = self.cfg

        # 1) Aggregate all local expert outputs to be returned (concatenate once)
        slices_y = []
        slices_meta = []
        counts = expert_num_tokens.to(torch.long)
        for e in range(self.num_local_experts):
            cnt = int(counts[e].item())
            if cnt == 0:
                continue
            slices_y.append(expert_y[e, :cnt])
            slices_meta.append(expert_meta[e, :cnt])

        if slices_y:
            flat_y = torch.cat(slices_y, dim=0)
            flat_meta = torch.cat(slices_meta, dim=0)
        else:
            flat_y = torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
            flat_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # 2) Compute destination ranks and perform all2all (sort by rank to form contiguous blocks)
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

        # 3) Use index_add_ for vectorized write-back and weighted sum
        if recv_buf.shape[0] > 0:
            src_tokens = recv_meta[:, 2].to(torch.long)
            src_ks = recv_meta[:, 3].to(torch.long)
            w = weights[src_tokens, src_ks].to(torch.float32)
            contrib = recv_buf.to(torch.float32) * w.unsqueeze(1)
            accum = torch.zeros_like(out_tokens, dtype=torch.float32)
            accum.index_add_(0, src_tokens, contrib)
            out_tokens.add_(accum.to(out_tokens.dtype))

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    ata = PyTorchAllToAll(cfg, rank, world_size)

    with torch.inference_mode():
        expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)
        expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
        y = torch.zeros(
            cfg.max_num_tokens,
            cfg.hidden_dim,
            dtype=cfg.out_dtype,
            device=rank_data.x.device,
        )
        ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)
        return y[: rank_data.num_tokens]
