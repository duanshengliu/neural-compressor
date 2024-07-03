import time

import pytest
import torch

import neural_compressor.torch.algorithms.weight_only.modules as inc_modules


@pytest.mark.parametrize("out_features", [5120, 13824])
@pytest.mark.parametrize("in_features", [13824])
def test_pack(in_features, out_features):
    bits = 4
    raw_tensor = torch.randint(0, 15, (out_features, in_features), dtype=torch.int8)
    n_pack = 32 // 4
    compression_dtype: torch.dtype = torch.int32
    iters = 100
    ref_start = time.time()
    for i in range(iters):
        ref = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_static(raw_tensor, n_pack, bits, compression_dtype)
    ref_dur = time.time() - ref_start
    res_start = time.time()
    for i in range(iters):
        res = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_opt(raw_tensor, n_pack, bits, compression_dtype)
    res_dur = time.time() - res_start

    raw_np = raw_tensor.numpy()
    res_np_start = time.time()
    import numpy as np

    for i in range(iters):
        res_np = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_opt_np(raw_np, n_pack, bits, np.int32)
    res_np_dur = time.time() - res_np_start

    raw_np = raw_tensor.numpy()
    res_np2_start = time.time()
    import numpy as np

    for i in range(iters):
        res_np2 = inc_modules.WeightOnlyLinear.pack_tensor_with_numpy_np_v2(raw_np, n_pack, bits, np.int32)
    res_np2_dur = time.time() - res_np2_start
    # assert np.array_equal(res_np, res_np2) Assert failed
    print(f"ref_dur:{ref_dur}, res_dur:{res_dur} res_np: {res_np_dur}, res_np2: {res_np2_dur}")
    assert torch.allclose(ref, res), f"ref:{ref}, res:{res}"
