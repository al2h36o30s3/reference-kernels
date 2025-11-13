from task import input_t, output_t
import torch
import torch.nn.functional as F

def custom_kernel(data: input_t) -> output_t:
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()

    output = F.linear(input, weight, bias).contiguous()  # [M, N]

    chunk_m = M // world_size
    output_reshaped = output.view(world_size, chunk_m, N).contiguous()

    rs_output = torch.empty((chunk_m, N), dtype=output.dtype, device=input.device)

    torch.distributed.reduce_scatter_tensor(rs_output, output_reshaped, op=torch.distributed.ReduceOp.SUM)

    return rs_output