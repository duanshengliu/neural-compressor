# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from neural_compressor.torch.quantization.quantize import quantize
from neural_compressor.torch.quantization.config import (
    RTNConfig,
    get_default_rtn_config,
    get_default_double_quant_config,
    GPTQConfig,
    get_default_gptq_config,
    StaticQuantConfig,
    get_default_static_config,
    SmoothQuantConfig,
    get_default_sq_config,
    HQQConfig,
    get_default_hqq_config,
)

# TODO(Yi): move config to config.py
from neural_compressor.torch.quantization.autotune import autotune, TuningConfig, get_all_config_set

### Quantization Function Registration ###
import neural_compressor.torch.quantization.algorithm_entry
