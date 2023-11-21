import argparse
import time
import json
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import habana_frameworks.torch.hpex
import lm_eval.tasks
import lm_eval.evaluator
torch.set_grad_enabled(False)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="facebook/opt-125m"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--to_graph", action="store_true")
parser.add_argument("--approach", type=str, default=None, 
                    help="Select from ['dynamic', 'static' 'cast']")
parser.add_argument("--precision", type=str, default='fp8_e4m3', 
                    help="Select from ['fp8_e4m3', 'fp8_e5m2', 'bf16', 'fp16'], \
                        ['bf16', 'fp16'] only work with cast approach")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=100, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai"], type=str, \
                    choices=["winogrande", "copa", "piqa", "rte", "hellaswag", \
                    "openbookqa", "lambada_openai", "lambada_standard", "wikitext"],
                    help="tasks list for accuracy validation")
parser.add_argument("--limit", default=None, type=int,
                    help="the sample num of evaluation.")
parser.add_argument("--max_new_tokens", default=100, type=int,
                    help="calibration iters.")
parser.add_argument('--buckets', type=int, nargs='+', \
                    help="Input length buckets to use with static_shapes", default=[129])

args = parser.parse_args()


if args.approach is None:
    import habana_frameworks.torch.core as htcore
    htcore.hpu_set_env()

# model
if re.search("llama", args.model.lower()):
    from models.modeling_llama import LlamaForCausalLM
    user_model = LlamaForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        device_map='hpu',
    )
elif re.search("chatglm", args.model.lower()):
    from models.modeling_chatglm import ChatGLMForConditionalGeneration
    user_model = ChatGLMForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        device_map='hpu',
    )
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map='hpu',
    )
# tokenizer
if re.search("baichuan", args.model.lower()):
    from models.tokenization_baichuan import BaichuanTokenizer
    tokenizer = BaichuanTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=args.trust_remote_code
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=args.trust_remote_code
    )

if args.approach is None:
    from habana_frameworks.torch.core.quantization import _mark_params_as_const, _check_params_as_const
    _mark_params_as_const(user_model)
    _check_params_as_const(user_model)
    import habana_frameworks.torch.core as htcore
    htcore.hpu_initialize(user_model)

user_model.eval()

if args.approach in ["dynamic", "static"]:
    print("device:", next(user_model.parameters()).device)
    from neural_compressor.torch.quantization import get_fp8_e5m2_qconfig, get_fp8_e4m3_qconfig
    if args.precision == "fp8_e4m3":
        dtype = torch.float8_e4m3fn
        qconfig = get_fp8_e4m3_qconfig()
    else:
        dtype = torch.float8_e5m2
        qconfig = get_fp8_e5m2_qconfig()


    from neural_compressor.torch.quantization.fp8 import quantize_dynamic, quantize
    if args.approach == "dynamic":
        user_model = quantize_dynamic(user_model, dtype, inplace=True)
    elif args.approach == "static":
        # dataset
        from datasets import load_dataset
        calib_dataset = load_dataset(args.dataset, split="train").select(range(100))
        calib_dataset = calib_dataset.shuffle(seed=42)
        calib_data = []
        for examples in calib_dataset:
            calib_data.append(
                tokenizer(examples["text"], return_tensors="pt", max_length=128)
            )

        def calib_func(model):
            for i, calib_input in enumerate(calib_data):
                if i >= args.calib_iters:
                    break
                model(
                    input_ids=calib_input["input_ids"].to('hpu'),
                    attention_mask=calib_input["attention_mask"].to('hpu'),
                )

        user_model = quantize(user_model, qconfig, calib_func=calib_func, inplace=True)
    print(user_model)

if args.to_graph:
    import habana_frameworks.torch.hpu.graphs as htgraphs
    user_model = htgraphs.wrap_in_hpu_graph(user_model)

if args.generate:
    input_prompt = "DeepSpeed is a machine learning framework"
    print("Prompt sentence:", input_prompt)
    generation_config = {
        "min_new_tokens": args.max_new_tokens, "max_new_tokens": args.max_new_tokens,
        # "do_sample": False, "temperature": 0.9, "num_beams": 4,
    }
    input_tokens = tokenizer(input_prompt, return_tensors="pt").to('hpu')
    eval_start = time.perf_counter()
    if args.approach == "cast":
        from neural_compressor.torch.amp import autocast
        from neural_compressor.torch.dtype import float8_e4m3, float8_e5m2
        if args.precision == "fp8_e4m3":
            dtype = float8_e4m3
        elif args.precision == "fp8_e5m2":
            dtype = float8_e5m2
        elif args.precision == "fp16":
            dtype = torch.float16
        elif args.precision == "bf16":
            dtype = torch.bfloat16
        with autocast('hpu', dtype=dtype):
            outputs = user_model.generate(**input_tokens, **generation_config)
    else:
        outputs = user_model.generate(**input_tokens, **generation_config)

    output_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    eval_end = time.perf_counter()
    print("Generated sentence:", output_sentence)
    print("Duration:", eval_end - eval_start)

if args.accuracy:

    class HabanaModelAdapter(lm_eval.base.BaseLM):
        def __init__(self, tokenizer, model, args, options):
            super().__init__()
            self.tokenizer = tokenizer
            self.model = model.eval()
            self._batch_size = args.batch_size
            self.buckets = list(sorted(args.buckets))
            self.options = options
            self._device = "hpu"
            torch.set_grad_enabled(False)

        @property
        def eot_token_id(self):
            return self.model.config.eos_token_id

        @property
        def max_length(self):
            return self.buckets[-1]

        @property
        def max_gen_toks(self):
            raise NotImplementedError()

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            # We need to do padding ourselves, otherwise we'll end up with recompilations
            # Returning 'cpu' to keep tensors on CPU in lm_eval code
            return 'hpu'

        def tok_encode(self, string):
            if re.search("chatglm3", args.model.lower()) or re.search("llama", args.model.lower()) :
                string = string.lstrip()
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_generate(self, context, max_length, eos_token_id):
            raise NotImplementedError()

        def find_bucket(self, length):
            return [b for b in self.buckets if b >= length][0]

        def _model_call(self, inps):
            seq_length = inps.shape[-1]
            bucket_length = self.find_bucket(seq_length)
            padding_length = bucket_length - seq_length
            if True:
                import torch.nn.functional as F
                inps = F.pad(inps, (0, padding_length), value=self.model.config.pad_token_id)

            logits = self.model(inps.to(self._device))['logits']
            if True and padding_length > 0:
                logits = logits[:, :-padding_length, :]
            logits = logits.to(torch.float32)
            return logits

    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    options = None
    lm = HabanaModelAdapter(tokenizer, user_model, args, options)

    eval_start = time.perf_counter()
    if args.approach == "cast":
        from neural_compressor.torch.amp import autocast
        from neural_compressor.torch.dtype import float8_e4m3, float8_e5m2
        if args.precision == "fp8_e4m3":
            dtype = float8_e4m3
        elif args.precision == "fp8_e5m2":
            dtype = float8_e5m2
        elif args.precision == "fp16":
            dtype = torch.float16
        elif args.precision == "bf16":
            dtype = torch.bfloat16
        with autocast('hpu', dtype=dtype):
            results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit)
    else:
        results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit)
    print(lm_eval.evaluator.make_table(results)) 
    eval_end = time.perf_counter()
    print("Duration:", eval_end - eval_start)