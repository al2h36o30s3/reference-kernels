from task import input_t, output_t
import torch
import torch.nn.functional as F

def custom_kernel(data: input_t) -> output_t:
    """
    Optimized Gemm-ReduceScatter kernel for MI300X without unnecessary overlap.
    """
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()

    # ---- GEMM + bias fused ----
    # Use F.linear to automatically invoke a high-performance rocBLAS kernel
    output = F.linear(input, weight, bias)  # [M, N], contiguous

    # ---- ReduceScatter ----
    # Reshape directly to [world_size, M // world_size, N] to avoid slicing
    output = output.view(world_size, M // world_size, N).contiguous()

    rs_output = torch.empty((M // world_size, N),
                            dtype=output.dtype,
                            device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)

    return rs_output