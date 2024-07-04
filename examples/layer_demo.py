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

# model_name = "/mnt/disk4/modelHub/Llama-2-7b-chat-hf/"

MODEL_NAME = (
    "/home/sdp/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a"
)
MODEL_NAME = "/home/sdp/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/"
# model_name = "hf-internal-testing/tiny-random-GPTJForCausalLM"


# self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
# # record label for comparison
# self.label = self.tiny_gptj(self.example_inputs)[0]
# test_default_config
def without_layer_wise(model_name):
    inc_utils.time_record.clear()
    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
    )
    model = tiny_gptj
    quant_config = RTNConfig(group_size=128)
    model = prepare(model, quant_config)
    model = convert(model)
    inc_utils.summary_time_usage()
    del model
    del tiny_gptj
    gc.collect()


def prof_without_layer_wise(model_name):
    inc_utils.time_record.clear()
    #

    with torch.autograd.profiler.profile(with_stack=True, profile_memory=True, with_modules=True, use_cpu=True) as prof:
        tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
        )
        inc_utils.time_record.clear()
        # model = copy.deepcopy(tiny_gptj)
        model = tiny_gptj
        quant_config = RTNConfig(group_size=128)
        model = prepare(model, quant_config)
        model = convert(model)
        inc_utils.summary_time_usage()
        del model
    import pdb

    pdb.set_trace()
    prof.export_chrome_trace("trace_no_layer_wise3.json")


print("----------------------------------------------------------------------------")


@inc_utils.dump_elapsed_time()
def with_layer_wise(model_name):
    inc_utils.time_record.clear()
    inc_utils.summary_time_usage()
    model = load_empty_model(model_name)
    quant_config = RTNConfig(
        use_layer_wise=True,
        model_path=model_name,
    )
    # with torch.autograd.profiler.profile(with_stack=True, profile_memory=True, with_modules=True, use_cpu=True) as prof:
    model = prepare(model, quant_config)
    model = convert(model)
    model = prepare(model, quant_config)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    model = convert(model)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # out = model(self.example_inputs)[0]
    inc_utils.summary_time_usage()
    del model
    gc.collect()
    # prof.export_chrome_trace("trace_layer_wise.json")

    # with torch.autograd.profiler.profile(with_stack=True, profile_memory=True, with_modules=True, use_cpu=True) as prof:
    #     model = prepare(model, quant_config)
    #     model = convert(model)
    #     model = prepare(model, quant_config)
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    #     model = convert(model)
    #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     # out = model(self.example_inputs)[0]
    #     inc_utils.summary_time_usage()
    # prof.export_chrome_trace("trace_layer_wise.json")


import gc


def multi_rounds_no_layer(counts, model_name):
    for i in range(counts):
        print("==============================")
        without_layer_wise(model_name)
        gc.collect()
        # with_layer_wise()


def multi_rounds(counts, model_name):
    for i in range(counts):
        print("==============================")
        # without_layer_wise()
        # gc.collect()
        with_layer_wise(model_name)


import pdb


@inc_utils.dump_elapsed_time()
def profiling_layer_wise(model_name):
    inc_utils.time_record.clear()
    inc_utils.summary_time_usage()
    model = load_empty_model(model_name)
    quant_config = RTNConfig(
        use_layer_wise=True,
        model_path=model_name,
    )

    with torch.autograd.profiler.profile(with_stack=True, profile_memory=True, with_modules=True, use_cpu=True) as prof:
        model = prepare(model, quant_config)
        model = convert(model)
        model = prepare(model, quant_config)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>")
        model = convert(model)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # out = model(self.example_inputs)[0]
        inc_utils.summary_time_usage()
    pdb.set_trace()
    prof.export_chrome_trace("trace_layer_wise2.json")


import click


@click.command()
@click.option("--funcname", default="multi_rounds", help="Number of greetings.")
@click.option("--counts", default=10, help="Number of rounds.")
@click.option("--model_name", default=None, help="Model name.")
def main(funcname, counts, model_name):
    if model_name is None:
        model_name = MODEL_NAME
    if funcname == "multi_rounds":
        multi_rounds(counts, model_name)
    elif funcname == "multi_rounds_no_layer":
        multi_rounds_no_layer(counts, model_name)
    elif funcname == "profiling_layer_wise":
        profiling_layer_wise(model_name)
    elif funcname == "prof_without_layer_wise":
        prof_without_layer_wise(model_name)
    else:
        raise ValueError(f"Unknown function name: {funcname}")


if __name__ == "__main__":
    main()

"""
W/o Layer wise
Time Statistics:
RTN Quantizer prepare:
  Average time: 0.01 ms
  Standard deviation: 0.00 ms
quant_tensor:
  Average time: 7199.18 ms
  Standard deviation: 646.28 ms
WOQ Packing:
  Average time: 29929.56 ms
  Standard deviation: 538.91 ms
RTN Quantizer Convert:
  Average time: 37322.53 ms
  Standard deviation: 925.98 ms


W/ Layer-wise
  Time Statistics:
RTN Quantizer prepare:
  Average time: 0.00 ms
  Standard deviation: 0.00 ms
_save_one_module:
  Average time: 1267.46 ms
  Standard deviation: 219.51 ms
load_module:
  Average time: 3707.16 ms
  Standard deviation: 421.02 ms
quant_tensor:
  Average time: 4567.36 ms
  Standard deviation: 180.32 ms
WOQ Packing:
  Average time: 26630.09 ms
  Standard deviation: 1530.63 ms
RTN Quantizer Convert:
  Average time: 36439.28 ms
  Standard deviation: 1661.89 ms


layer- llama-2-7b-chat-hf
(llm) (base) sdp@:inc$ python parse_log.py _10time_7b_with_5cores_with_layer
Time Statistics:
RTN Quantizer prepare:
  Average time: 0.01 ms
  Standard deviation: 0.01 ms
_save_one_module:
  Average time: 5029.02 ms
  Standard deviation: 438.29 ms
load_module:
  Average time: 13411.94 ms
  Standard deviation: 816.06 ms
quant_tensor:
  Average time: 18924.40 ms
  Standard deviation: 200.50 ms
WOQ Packing:
  Average time: 143808.35 ms
  Standard deviation: 1820.23 ms
RTN Quantizer Convert:
  Average time: 182878.02 ms
  Standard deviation: 2349.06 ms




W/ Layer-wise
(Pdb) print(prof.key_averages().table(sort_by="cpu_time_total"))
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            new_module.pack        61.78%        1.370s        68.51%        1.519s      72.346ms           0 b    -736.64 Mb            21  
                                load_module        19.96%     442.612ms        19.98%     443.099ms      21.100ms     189.00 Mb     189.00 Mb            21  
                               quant_tensor         0.27%       5.941ms        11.30%     250.561ms      11.931ms     -10.36 Mb    -245.18 Mb            21  
                                aten::copy_         5.41%     120.037ms         5.42%     120.151ms     255.098us      13.41 Mb      12.18 Mb           471  
                                  aten::max         3.48%      77.154ms         3.54%      78.548ms       1.870ms      66.66 Mb      49.97 Mb            42  
                                  aten::min         2.87%      63.542ms         2.87%      63.629ms       3.030ms      50.20 Mb      50.20 Mb            21  
                                   aten::to         0.11%       2.362ms         2.78%      61.743ms      76.226us     543.88 Mb      13.85 Mb           810  
                             aten::_to_copy         0.05%       1.019ms         2.77%      61.461ms     280.644us     543.88 Mb      30.35 Mb           219  
                           aten::contiguous         0.07%       1.442ms         2.58%      57.208ms     302.688us     163.39 Mb       7.18 Mb           189  
                                aten::clone         0.08%       1.799ms         2.57%      57.050ms     301.852us     163.37 Mb     -14.50 Mb           189  
                               aten::clamp_         2.53%      56.204ms         2.55%      56.491ms       2.690ms          20 b        -196 b            21  
                                 aten::add_         0.85%      18.772ms         0.86%      19.028ms     906.095us           8 b         -92 b            21  
                                 aten::div_         0.80%      17.819ms         0.80%      17.819ms     848.524us           0 b           0 b            21  
                               aten::round_         0.79%      17.629ms         0.79%      17.629ms     839.476us           0 b           0 b            21  
                                  aten::abs         0.14%       3.127ms         0.23%       5.115ms      32.580us     124.88 Mb      76.12 Mb           157  

W/o Layer-wise
(Pdb)  print(prof.key_averages().table(sort_by="cpu_time_total"))
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            new_module.pack        30.19%        1.349s        33.69%        1.506s      71.706ms           0 b    -700.84 Mb            21  
                                  aten::cos        17.37%     776.138ms        17.37%     776.138ms      27.719ms       1.75 Gb       1.75 Gb            28  
                                  aten::sin        16.81%     751.101ms        16.81%     751.101ms      26.825ms       1.75 Gb       1.75 Gb            28  
                                  aten::cat        15.34%     685.428ms        15.36%     686.630ms      24.523ms       1.75 Gb       1.75 Gb            28  
                               quant_tensor         0.25%      11.047ms         9.01%     402.742ms      19.178ms     420.00 Kb     -58.31 Mb            21  
                                  aten::mul         6.70%     299.314ms         6.71%     300.009ms       3.896ms     900.19 Mb     900.19 Mb            77  
                                aten::outer         0.01%     564.000us         6.70%     299.448ms      10.695ms     896.00 Mb           0 b            28  
                                 aten::div_         4.16%     185.969ms         4.16%     185.969ms       8.856ms           0 b           0 b            21  
                                aten::copy_         2.80%     125.160ms         2.80%     125.251ms     131.843us      11.68 Mb      10.54 Mb           950  
                                   aten::to         0.16%       6.953ms         1.70%      75.981ms      20.176us     551.61 Mb      10.45 Mb          3766  
                             aten::_to_copy         0.13%       5.747ms         1.66%      74.399ms     106.589us     551.61 Mb      60.43 Mb           698  
                                  aten::max         1.59%      70.996ms         1.61%      72.027ms       1.715ms      16.71 Mb      13.02 Mb            42  
                               aten::clamp_         1.33%      59.596ms         1.35%      60.325ms       2.873ms          16 b        -176 b            21  
                           aten::contiguous         0.09%       4.030ms         1.25%      55.880ms     295.661us     141.37 Mb      10.35 Mb           189  
                                aten::clone         0.03%       1.542ms         1.24%      55.562ms     293.979us     141.34 Mb     -11.54 Mb           189  
                                  aten::min         1.06%      47.450ms         1.06%      47.548ms       2.264ms      12.55 Mb      12.55 Mb            21  
                               aten::arange         0.28%      12.622ms         0.53%      23.673ms     211.366us      56.03 Mb      13.00 Mb           112  
                                 aten::add_         0.45%      20.061ms         0.46%      20.426ms     972.667us         -24 b        -120 b            21  
                               aten::round_         0.41%      18.335ms         0.41%      18.335ms     873.095us           0 b           0 b            21  
                               aten::detach         0.10%       4.540ms         0.19%       8.582ms       2.090us           0 b           0 b          4107  
                                aten::empty         0.10%       4.566ms         0.10%       4.566ms       3.462us       6.79 Gb       6.79 Gb          1319  
                                     detach         0.10%       4.564ms         0.10%       4.564ms       1.111us           0 b           0 b          4107  
                              aten::type_as         0.00%     218.000us         0.10%       4.470ms     159.643us      14.00 Mb     512.00 Kb            28  
                                aten::zeros         0.02%     848.000us         0.10%       4.387ms      52.226us      69.68 Mb    -149.00 Kb            84  

|| RTN Quantizer prepare, total time: 0.01 ms
2024-07-03 01:41:28 [INFO][utility.py:188] || load_module, total time: 3956.42 ms
2024-07-03 01:41:28 [INFO][utility.py:188] || quant_tensor, total time: 1898.1400000000003 ms
2024-07-03 01:41:28 [INFO][utility.py:188] || WOQ Packing, total time: 13792.359999999997 ms
2024-07-03 01:41:28 [INFO][utility.py:188] || RTN Quantizer Convert, total time: 19806.59 ms

2024-07-03 01:41:58 [INFO][utility.py:186] ==--------------Time usage summary--------------==
2024-07-03 01:41:58 [INFO][utility.py:188] || RTN Quantizer prepare, total time: 0.02 ms
2024-07-03 01:41:58 [INFO][utility.py:188] || load_module, total time: 2678.640000000001 ms
2024-07-03 01:41:58 [INFO][utility.py:188] || quant_tensor, total time: 1967.8199999999995 ms
2024-07-03 01:41:58 [INFO][utility.py:188] || WOQ Packing, total time: 11777.710000000001 ms
2024-07-03 01:41:58 [INFO][utility.py:188] || RTN Quantizer Convert, total time: 16573.219999999998 ms
2024-07-03 01:41:58 [INFO][utility.py:189] ==----------------------------------------------==

"""
