Python Launcher
===========================

Neural Coder can be used as a Python **Launcher**. Users can run the Python model code as it is with automatic enabling of Deep Learning optimizations by using Neural Coder's inline Python **Launcher** design.

## Quick-Start

Example: Let's say you are running an NLP model using ```run_glue.py``` from HuggingFace transformers [examples](https://github.com/huggingface/transformers/blob/v4.21-release/examples/pytorch/text-classification/run_glue.py).

Pre-requisites:
```bash
pip install transformers==4.21.0 torch datasets 
```

Generally we run this code with a Python command line like this:
```bash
python run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result
```

With Neural Coder's **Launcher**, users can easily enjoy Deep Learning optimizations (e.g. default - INT8 dynamic quantization by Intel® Neural Compressor for PyTorch models) by simply adding an inline prefix
```bash
-m neural_coder
```
to the Python command line, and everything else remains the same:
```bash
python -m neural_coder run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result
```

This will run ```run_glue.py``` with the Deep Learning optimization automatically enabled, while everything else (e.g. your input arguments for the code itself) remains the same as the original code. You can also check out the optimized code ```run_glue_optimized.py``` auto-generated by the **Launcher** under the same folder if you want to learn the code enabling.

Note: Any modification on the optimized code ```run_glue_optimized.py``` will be overwritten every time you run Neural Coder **Launcher** on ```run_glue.py```, so please make any modification on the original code ```run_glue.py``` instead of the optimized one. The optimized code is only saved for your reference.

## Launcher Arguments (Optional)

Users can specify which Deep Learning optimization they want to conduct using ```--opt``` argument. The list of supported Deep Learning optimization features can be found [here](SupportMatrix.md).

Note that if specifically optimizing with INT8 quantization by Intel® Neural Compressor, to choose a quantization approach (strategy), ```--approach``` argument can be specified with either ```static```, ```static_ipex``` or ```dynamic```. For example, to run INT8 static quantization by Intel® Neural Compressor:
```bash
python -m neural_coder --approach static run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result
```