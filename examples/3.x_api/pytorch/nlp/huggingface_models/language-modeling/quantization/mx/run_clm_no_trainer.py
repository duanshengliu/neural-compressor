import argparse
import time
import json
import sys
import torch
from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()
if "hpu" in device:
    import habana_frameworks.torch.hpex
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.hpu import memory_stats
    import numpy as np
    import glob
    htcore.hpu_set_env()
    torch.device('hpu')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--quantize", action="store_true")
parser.add_argument('--buckets', type=int, nargs='+', \
                    help="Input length buckets to use with static_shapes", default=[256, 512])
# dynamic only now
parser.add_argument("--w_dtype", type=str, default="int8", 
                    choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", 
                                                "fp6_e2m3", "fp4", "float16", "bfloat12"],
                    help="weight data type")
parser.add_argument("--act_dtype", type=str, default="int8", 
                    choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", 
                                                "fp6_e2m3", "fp4", "float16", "bfloat12"],
                    help="input activation data type")
parser.add_argument("--woq", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--iters", default=100, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", type=str, default=["lambada_openai"],
                    help="tasks list for accuracy validation")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")
parser.add_argument("--to_graph", action="store_true")

args = parser.parse_args()


def itrex_bootstrap_stderr(f, xs, iters):
    from lm_eval.metrics import _bootstrap_internal, sample_stddev
    res = []
    chunk_size = min(1000, iters)
    it = _bootstrap_internal(f, chunk_size)
    for i in range(iters // chunk_size):
        bootstrap = it((i, xs))
        res.extend(bootstrap)
    return sample_stddev(res)


def eval_func(user_model, tokenizer, args):
    import os
    import re
    import time
    import json
    import torch
    import habana_frameworks.torch.hpex
    import torch.nn.functional as F
    import lm_eval
    import lm_eval.tasks
    import lm_eval.evaluator

    # to avoid out-of-memory caused by Popen for large language models.
    lm_eval.metrics.bootstrap_stderr = itrex_bootstrap_stderr

    class HabanaModelAdapter(lm_eval.base.BaseLM):
        def __init__(self, tokenizer, model, args, options):
            super().__init__()
            self.tokenizer = tokenizer
            self.model = model.eval()
            self._batch_size = args.batch_size
            self.buckets = args.buckets
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
            return 'cpu' # 'hpu'

        def tok_encode(self, string):
            if (
                re.search("chatglm3", args.model.lower()) or
                re.search("llama", args.model.lower()) or
                re.search("mistral", args.model.lower())
            ):
                string = string.lstrip()
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_generate(self, context, max_length, eos_token_id):
            raise NotImplementedError()

        def find_bucket(self, length):
            return [b for b in self.buckets if b >= length][0]

        def _model_call(self, inputs):
            seq_length = inputs.shape[-1]
            padding_length = 0
            bucket_length = self.find_bucket(seq_length)
            padding_length = bucket_length - seq_length
            inputs = F.pad(inputs, (0, padding_length), value=self.model.config.pad_token_id)
            logits = self.model(inputs.to(self._device))["logits"].cpu()

            if padding_length > 0:
                logits = logits[:, :-padding_length, :]
            logits = logits.to(torch.float32)
            return logits

    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    options = None
    lm = HabanaModelAdapter(tokenizer, user_model, args, options)

    eval_start = time.perf_counter()
    results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=10)
    print(lm_eval.evaluator.make_table(results))
    eval_end = time.perf_counter()
    print("Duration:", eval_end - eval_start)
    results['args'] = vars(args)
    results['duration'] = eval_end - eval_start

    # make sure that result is dumped only once during multi-cards evaluation
    local_rank = int(os.getenv('LOCAL_RANK', '-1'))
    if local_rank in [-1, 0]:
        dumped = json.dumps(results, indent=2)
        accu_dict = {}
        for task_name in args.tasks:
            if task_name == "wikitext":
                print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]), flush=True)
            else:
                print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]), flush=True)


def get_user_model():
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    user_model.eval()
    return user_model, tokenizer

user_model, tokenizer = get_user_model()
if args.quantize:
    from neural_compressor.torch.quantization import MXQuantConfig, quantize
    quant_config = MXQuantConfig(w_dtype=args.w_dtype, act_dtype=args.act_dtype, weight_only=args.woq)
    user_model = quantize(model=user_model, quant_config=quant_config)

if "hpu" in device:
    if args.to_graph:
        # inference optimization
        import habana_frameworks.torch.hpu.graphs as htgraphs
        user_model = htgraphs.wrap_in_hpu_graph(user_model)

if args.accuracy:
    eval_func(user_model, tokenizer=tokenizer, args=args)
    #import pdb;pdb.set_trace()
    #from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
    #eval_args = LMEvalParser(
    #    model="hf",
    #    user_model=user_model,
    #    tokenizer=tokenizer,
    #    batch_size=args.batch_size,
    #    tasks=','.join(args.tasks),
    #    device=device,
    #    limit=10,
    #)
    #results = evaluate(eval_args)
    #dumped = json.dumps(results, indent=2)
    #if args.save_accuracy_path:
    #    with open(args.save_accuracy_path, "w") as f:
    #        f.write(dumped)

    #eval_acc = 0
    #for task_name in args.tasks:
    #    if task_name == "wikitext":
    #        print("Accuracy for %s is: %s" %
    #              (task_name, results["results"][task_name]["word_perplexity,none"]))
    #        eval_acc += results["results"][task_name]["word_perplexity,none"]
    #    else:
    #        print("Accuracy for %s is: %s" %
    #              (task_name, results["results"][task_name]["acc,none"]))
    #        eval_acc += results["results"][task_name]["acc,none"]

if args.performance:
    eval_start = time.perf_counter()
    input_prompt = "Intel is a company which"
    input_tokens = torch.ones((1, 128), dtype=torch.long).to('hpu')
    generation_config = {"min_new_tokens": 100, "max_new_tokens": 100}
    outputs = user_model.generate(input_tokens, **generation_config)
    print("Duration of generating 100 tokens :", time.perf_counter() - eval_start)
