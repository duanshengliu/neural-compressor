import copy
import shutil

import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.weight_only.modules import WeightOnlyLinear
from neural_compressor.torch.quantization import (
    RTNConfig,
    convert,
    get_default_double_quant_config,
    get_default_rtn_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import accelerator, is_hpex_available

device = accelerator.current_device_name()

import neural_compressor.common.utils.utility as inc_utils
from neural_compressor.torch.algorithms.layer_wise import load_empty_model

model_name = "/mnt/disk4/modelHub/Llama-2-7b-chat-hf/"
# model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"


# self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
# # record label for comparison
# self.label = self.tiny_gptj(self.example_inputs)[0]
# test_default_config
def without_layer_wise():

    inc_utils.time_record.clear()
    #

    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
    )
    inc_utils.time_record.clear()
    model = copy.deepcopy(tiny_gptj)
    quant_config = get_default_rtn_config()
    model = prepare(model, quant_config)
    model = convert(model)
    inc_utils.summary_time_usage()


print("----------------------------------------------------------------------------")


def with_layer_wise():

    inc_utils.time_record.clear()
    inc_utils.summary_time_usage()
    model = load_empty_model(model_name)
    quant_config = RTNConfig(
        use_layer_wise=True,
        model_path=model_name,
    )
    model = prepare(model, quant_config)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    model = convert(model)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # out = model(self.example_inputs)[0]
    inc_utils.summary_time_usage()


for i in range(10):
    print("==============================")
    without_layer_wise()
    with_layer_wise()
