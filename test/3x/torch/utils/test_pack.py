import timeit

import numpy as np
import pytest
import torch

import neural_compressor.torch.algorithms.weight_only.modules as inc_modules

fn1 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_static
fn2 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_opt_np_numba
fn1 = torch.compile(inc_modules.WeightOnlyLinear.pack_tensor_with_torch_static)


@pytest.mark.parametrize("out_features", [128, 1024, 5120, 13824])
@pytest.mark.parametrize("in_features", [1024, 13824])
def test_pack(in_features, out_features):
    bits = 4

    raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.int8)
    n_pack = 32 // 4
    compression_dtype: torch.dtype = torch.int32
    iters = 20
    raw_np = raw_tensor.numpy()
    time_ref = timeit.timeit(lambda: fn1(raw_tensor, n_pack, bits, compression_dtype), number=iters)
    time_res = timeit.timeit(lambda: fn1(raw_tensor, n_pack, bits, compression_dtype), number=iters)

    print(f"ref : {time_ref},  res: {time_res}, speed up: {time_ref / time_res}")

    # print(f"ref_dur:{ref_dur}, res_dur:{res_dur} res_np")

    # assert np.array_equal(ref.numpy(), res), f"ref:{ref}, res:{res}"
    # assert torch.allclose(ref, torch.from_numpy(res)), f"ref:{ref}, res:{res}"
