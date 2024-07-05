import timeit

import numpy as np
import pytest
import torch

import neural_compressor.torch.algorithms.weight_only.modules as inc_modules

fn1 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_static
fn2 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_opt_np_numba
# fn1 = torch.compile(inc_modules.WeightOnlyLinear.pack_tensor_with_torch_static)
# fn1 = inc_modules.WeightOnlyLinear.pack_tensor_with_reshape
# fn1_opt = torch.compile(fn1)


# out_features = 1024
# in_features =1024
# bits = 4

# raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.int8)
# n_pack = 32 // 4
# compression_dtype: torch.dtype = torch.int32
# iters = 20

# raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.int8)
# from torch._inductor import config as inductor_config
# # inductor_config.cpp_wrapper = True
# re = fn1(raw_tensor, n_pack, bits, compression_dtype)

# # timeit
# time_res = timeit.timeit(lambda: fn1_opt(raw_tensor, n_pack, bits, compression_dtype), number=iters)
# # time_res = timeit.timeit(lambda: fn1(raw_tensor, n_pack, bits, compression_dtype), number=iters)
# time_ref = timeit.timeit(lambda: fn2(raw_tensor.numpy(), n_pack, bits), number=iters)
# print(f"ref : {time_ref},  res: {time_res}, speed up: {time_ref / time_res}")


# @pytest.mark.parametrize("out_features", [128, 1024, 5120, 13824])
# @pytest.mark.parametrize("in_features", [1024, 13824])
@pytest.mark.parametrize("out_features", [128])
@pytest.mark.parametrize("in_features", [1024])
def test_pack(in_features, out_features):
    bits = 4

    raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.int8)
    n_pack = 32 // 4
    compression_dtype: torch.dtype = torch.int32
    iters = 10
    raw_np = raw_tensor.numpy()
    time_ref = timeit.timeit(lambda: fn1(raw_tensor, n_pack, bits, compression_dtype), number=iters)
    time_res = timeit.timeit(lambda: fn2(raw_np, n_pack, bits), number=iters)

    print(f"ref : {time_ref},  res: {time_res}, speed up: {time_ref / time_res}")

    # print(f"ref_dur:{ref_dur}, res_dur:{res_dur} res_np")

    # assert np.array_equal(ref.numpy(), res), f"ref:{ref}, res:{res}"
    # assert torch.allclose(ref, torch.from_numpy(res)), f"ref:{ref}, res:{res}"


@torch.jit.script
def qdq_weight_sym_light(weight):
    """Quant and dequant tensor with sym schema.

    Args:
        weight : input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                For example: 4 bit
                    scale = amax / 8 if full_range else amax / 7
                    If True, scale = -scale if abs(min)> abs(max) else scale
                    Defaults to False.

    Returns:
        output: qdq weight
    """
    bits = 4
    # assert bits > 1, "symmetric scheme only supports bits > 1"
    maxq = torch.tensor(2 ** (bits - 1) - 1)
    minq = torch.tensor(-(2 ** (bits - 1)))
    max_val = torch.max(weight, 1)[0]
    min_val = torch.min(weight, 1)[0]
    wmax = torch.max(torch.abs(max_val), torch.abs(min_val))
    wmax = wmax
    tmp = wmax == 0
    wmax[tmp] = torch.tensor(1, dtype=wmax.dtype, device=wmax.device)
    scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    weight.div_(scale)
    weight.round_()
    weight.clamp_(minq, maxq)
    return weight, scale, None


def qdq_weight_sym_light_no_jit(weight):
    """Quant and dequant tensor with sym schema.

    Args:
        weight : input weight
        bits (int, optional): bits. Defaults to 4.
        quantile (float, optional): percentile of clip. Defaults to 1.0.
        return_int (bool, optional): Choose return fp32 or int8/uint8 data.
                                     Defaults to False.
        full_range (bool, optional): Choose sym range whether use -2**(bits-1).
                For example: 4 bit
                    scale = amax / 8 if full_range else amax / 7
                    If True, scale = -scale if abs(min)> abs(max) else scale
                    Defaults to False.

    Returns:
        output: qdq weight
    """
    bits = 4
    # assert bits > 1, "symmetric scheme only supports bits > 1"
    maxq = torch.tensor(2 ** (bits - 1) - 1)
    minq = torch.tensor(-(2 ** (bits - 1)))
    max_val = torch.max(weight, 1)[0]
    min_val = torch.min(weight, 1)[0]
    wmax = torch.max(torch.abs(max_val), torch.abs(min_val))
    wmax = wmax
    tmp = wmax == 0
    wmax[tmp] = torch.tensor(1, dtype=wmax.dtype, device=wmax.device)
    scale = wmax / maxq
    scale.unsqueeze_(dim=-1)
    weight.div_(scale)
    weight.round_()
    weight.clamp_(minq, maxq)
    return weight, scale, None


@pytest.mark.parametrize("out_features", [128, 1024, 5120, 13824])
@pytest.mark.parametrize("in_features", [128, 1024, 13824])
def test_quant(in_features, out_features):
    bits = 4

    raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.float32)
    n_pack = 32 // 4
    compression_dtype: torch.dtype = torch.int32
    iters = 100

    time_ref = timeit.timeit(lambda: qdq_weight_sym_light_no_jit(raw_tensor), number=iters)
    for i in range(10):
        qdq_weight_sym_light(raw_tensor)
    time_res = timeit.timeit(lambda: qdq_weight_sym_light(raw_tensor), number=iters)

    print(f"ref : {time_ref},  res: {time_res}, speed up: {time_ref / time_res}")

    # print(f"ref_dur:{ref_dur}, res_dur:{res_dur} res_np")

    # assert np.array_equal(ref.numpy(), res), f"ref:{ref}, res:{res}"
    # assert torch.allclose(ref, torch.from_numpy(res)), f"ref:{ref}, res:{res}"
