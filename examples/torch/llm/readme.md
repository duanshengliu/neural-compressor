# Run

## Run FP32 model
``` python
python run_llm.py --model {model_name_or_path} --tasks lambada_openai --batch_size 32  --accuracy
```

## Run FP8 model
``` python
python run_llm.py --model {model_name_or_path} --approach [dynamic|static] --precision [fp8_e4m3|fp8_e5m2] --tasks lambada_openai --batch_size 32  --accuracy --quantize --to_graph
```