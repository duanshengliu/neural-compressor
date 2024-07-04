import time

import numpy as np
import pytest
import torch

import neural_compressor.torch.algorithms.weight_only.modules as inc_modules

fn1 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_static
fn2 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_opt_np_numba

import timeit


@pytest.mark.parametrize("out_features", [1024, 5120, 13824])
@pytest.mark.parametrize("in_features", [1024, 13824])
# @pytest.mark.parametrize("out_features", [1024])
# @pytest.mark.parametrize("in_features", [1024])
def test_pack(in_features, out_features):
    bits = 4
    raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.int8)
    n_pack = 32 // 4
    compression_dtype: torch.dtype = torch.int32
    iters = 20
    raw_np = raw_tensor.numpy()

    # Time the function without Numba
    time_without_numba = timeit.timeit(lambda: fn1(raw_tensor, n_pack, bits, compression_dtype), number=iters)

    # Time the function with Numba
    # Ensure the function is compiled before timing
    fn2(raw_np, n_pack, bits, np.int32)
    time_with_numba = timeit.timeit(lambda: fn2(raw_np, n_pack, bits, np.int32), number=iters)

    print(
        f"Time w/ Numba: {time_without_numba},  W/o Numba: {time_with_numba}, speed up: {time_without_numba / time_with_numba}"
    )

    # print(f"ref_dur:{ref_dur}, res_dur:{res_dur} res_np")
    # assert np.array_equal(ref.numpy(), res), f"ref:{ref}, res:{res}"
    # assert torch.allclose(ref, torch.from_numpy(res)), f"ref:{ref}, res:{res}"
