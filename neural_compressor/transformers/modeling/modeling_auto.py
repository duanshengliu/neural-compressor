# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import re
import types
from threading import Thread
from typing import Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from packaging import version
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import load_state_dict
from transformers.utils import has_file, is_accelerate_available, is_bitsandbytes_available, is_safetensors_available

from neural_compressor.model.torch_model import PyTorchFXModel
from neural_compressor.torch.algorithms.weight_only.modules import INCWeightOnlyLinear as WeightOnlyLinear
from neural_compressor.transformers.utils.utility import (
    _neural_compressor_version,
    is_intel_gpu_available,
    is_ipex_available,
)

from ..quantization.utils import (
    convert_dtype_str2torch,
    convert_dtype_torch2str,
    convert_to_quantized_model,
    replace_linear,
)
from ..utils import AutoRoundConfig, AwqConfig, GPTQConfig, LazyImport, RtnConfig, TeqConfig, logger
from ..utils.utility import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, CpuInfo

if is_ipex_available() and is_intel_gpu_available():
    # pylint: disable=E0401
    from intel_extension_for_pytorch.nn.utils._quantize_convert import WeightOnlyQuantizedLinear

torch = LazyImport("torch")


def recover_export_model(model, current_key_name=None):
    """Private method that wraps the recursion for module replacement.

    Return optimum format model.
    """
    from ..llm.quantization.nn.modules import QuantizedLinearQBits

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, QuantizedLinearQBits):
            (
                groupsize,
                in_features,
                out_features,
                desc_act,
                g_idx,
                weight_dtype,
                bits,
                scales_dtype,
                scales,
                zp,
                zeros,
                int_weight,
            ) = module.recover_qparms()
            dtype = "int4" if weight_dtype == "int4_clip" else weight_dtype
            use_optimum_format = False if weight_dtype in ["fp4_e2m1", "fp4", "nf4"] else True
            model._modules[name] = WeightOnlyLinear(
                in_features,
                out_features,
                dtype=dtype,
                bits=bits,
                group_size=groupsize,
                zp=zp,
                bias=module.bias is not None,
                scale_dtype=scales_dtype,
                g_idx=desc_act,
                use_optimum_format=use_optimum_format,
            )

            # Setting g_idx is invalid when use_optimum_format is True, so set it again when g_idx is not None.
            # https://github.com/intel/neural-compressor/blob/v2.5.dev2/neural_compressor/adaptor/torch_utils/
            # model_wrapper.py#L343
            model._modules[name].pack(
                int_weight.contiguous(),
                scales.contiguous(),
                zeros.contiguous() if zeros is not None else None,
                module.bias.contiguous() if module.bias is not None else None,
            )
            if g_idx is not None:
                model._modules[name].g_idx = g_idx.contiguous()

        if len(list(module.children())) > 0:  # pylint: disable=E1101
            _ = recover_export_model(module, current_key_name)
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model


def build_woq_model(model, quantization_config):
    from neural_compressor.adaptor.torch_utils.util import set_module

    weight_dtype = quantization_config.weight_dtype
    for n, m in model.named_modules():
        if n in quantization_config.llm_int8_skip_modules:
            continue
        if isinstance(m, torch.nn.Linear):
            zp = getattr(
                quantization_config,
                "zero_point",
                not getattr(quantization_config, "sym", False),
            )
            dtype = "int4" if weight_dtype == "int4_clip" else weight_dtype
            use_optimum_format = False if weight_dtype in ["nf4", "fp4", "fp4_e2m1"] else True
            with init_empty_weights():
                new_module = WeightOnlyLinear(
                    m.in_features,
                    m.out_features,
                    dtype=dtype,
                    bits=quantization_config.bits,
                    group_size=quantization_config.group_size,
                    zp=zp,
                    bias=m.bias is not None,
                    g_idx=True,
                    use_optimum_format=use_optimum_format,
                )
            set_module(model, n, new_module)
    return model


def convert_model_to_public(model):
    # reorder weight and scales if they have been transposed
    if model.device == "xpu" or (isinstance(model.device, torch.device) and model.device.type == "xpu"):
        for name, module in model.named_modules():
            if isinstance(module, WeightOnlyQuantizedLinear) and not module.use_optimum_format:
                if module.weight_transposed:
                    module.qweight.data = module.qweight.t_().contiguous()
                    module.scales.data = module.scales.t_().contiguous()
                    module.weight_transposed = False
    elif model.quantization_config.use_ipex:
        pass
    elif model.quantization_config.weight_dtype not in [
        "fp8_e5m2",
        "fp8_e4m3",
    ]:
        model = recover_export_model(model)


def save_low_bit(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):

    assert hasattr(self, "quantization_config"), "Detected this model is not a low-bit model."

    if os.path.isfile(save_directory):
        logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
        return
    if isinstance(self, PyTorchFXModel):
        self.quantization_config.save_pretrained(save_directory, **kwargs)
        self.model.config.quantization_config = self.quantization_config
        self.model.config.save_pretrained(save_directory)
        weights_file = os.path.join(os.path.abspath(os.path.expanduser(save_directory)), WEIGHTS_NAME)
        torch.save(self.quantized_state_dict(), weights_file)
        return

    convert_model_to_public(self)
    os.makedirs(save_directory, exist_ok=True)
    # use transformers original `save_pretrained` function
    del self.save_pretrained

    self.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    if self.quantization_config.use_ipex:

        def save_linear_parameters(model, save_directory):
            # only can save to pytorch model.bin due to ipex.
            weights_file = os.path.join(os.path.abspath(os.path.expanduser(save_directory)), SAFE_WEIGHTS_NAME)
            os.remove(weights_file)
            weights_file = os.path.join(os.path.abspath(os.path.expanduser(save_directory)), WEIGHTS_NAME)
            linear_parameters = {}
            from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear as ipex_cpu_linear

            for name, module in model.named_modules():
                if isinstance(module, ipex_cpu_linear):
                    linear_parameters[name + ".ipex_scales"] = module._op_context.get_scales().contiguous()
                    linear_parameters[name + ".ipex_weight"] = module._op_context.to_public(
                        module._op_context.get_weight()
                    ).contiguous()
                    linear_parameters[name + ".ipex_zeros"] = module._op_context.get_zero_points().contiguous()
                    if module._op_context.get_bias() is not None:
                        linear_parameters[name + ".ipex_bias"] = module._op_context.get_bias().contiguous()
                    if module._op_context.get_g_idx() is not None:
                        linear_parameters[name + ".ipex_g_idx"] = module._op_context.get_g_idx().contiguous()
            others_parameters = model.state_dict()
            linear_parameters.update(others_parameters)

            torch.save(linear_parameters, weights_file)

        save_linear_parameters(self, save_directory)
    self.save_pretrained = types.MethodType(save_low_bit, self)
    # We conveniently save all the keys of the model to have them on hand,
    # so that when using 'low_cpumem load',
    # it's not necessary to load the entire model to extract its keys
    # and we can avoid gc not triggered potentially.
    all_checkpoint_keys = {"all_checkpoint_keys": list(self.state_dict().keys())}
    json_file_path = os.path.join(save_directory, "all_checkpoint_keys.json")
    with open(json_file_path, "w") as json_file:
        json.dump(all_checkpoint_keys, json_file)
    if push_to_hub:
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            logger.warning.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token
        commit_message = kwargs.pop("commit_message", None)
        repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
        repo_id = self._create_repo(repo_id, **kwargs)
        files_timestamps = self._get_files_timestamps(save_directory)
        self._upload_modified_files(
            save_directory,
            repo_id,
            files_timestamps,
            commit_message=commit_message,
            token=kwargs.get("token"),
        )
    self.quantization_config.save_pretrained(save_directory, **kwargs)


class _BaseQBitsAutoModelClass:
    ORIG_MODEL = None
    model_type_list = [
        "llama",
        "gptj",
        "mpt",
        "opt",
        "gptneox",
        "dolly",
        "polyglot",
        "starcoder",
        "falcon",
        "bloom",
        "chatglm2",
        "chatglm",
        "baichuan",
        "mistral",
        "qwen",
        "phi",
        "whisper",
        "qwen2",
        "gemma",
        "phi3",
        "tinyllama",
    ]

    model_type_list_for_gptq = [
        "llama",
        "gptj",
        "mpt",
        "falcon",
        "chatglm2",
        "chatglm",
        "baichuan",
        "mistral",
        "qwen",
        "phi",
    ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):

        device_map = kwargs.get("device_map", "cpu")
        use_cpu = True if device_map == torch.device("cpu") or device_map == "cpu" else False
        use_xpu = True if device_map == torch.device("xpu") or device_map == "xpu" else False

        config = kwargs.pop("config", None)
        model_hub = kwargs.pop("model_hub", "huggingface")

        quantization_config = kwargs.pop("quantization_config", None)
        if not isinstance(config, PretrainedConfig):
            if model_hub == "modelscope":
                import modelscope  # pylint: disable=E0401

                config = modelscope.AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            else:
                config, _ = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    return_unused_kwargs=True,
                    **kwargs,
                )

        if hasattr(config, "quantization_config"):
            if config.quantization_config is None:
                logger.warning(
                    "Quantization_config loading failed. If you want to load saved "
                    "low bit model, please check your quantizate_config.json."
                )

            else:
                logger.info("quantization_config: {}".format(config.quantization_config))
                try:
                    model = cls.load_low_bit(
                        pretrained_model_name_or_path,
                        *model_args,
                        config=config,
                        **kwargs,
                    )
                    logger.info("Saved low bit model loading successfully. Other input args " "will be ignored.")
                    return model
                except Exception as e:
                    logger.error(e)
                    logger.error("Saved low bit model loading failed, please check your model.")
                    exit(0)

        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        if load_in_8bit or load_in_4bit:
            if is_accelerate_available() and is_bitsandbytes_available() and not use_cpu and not use_xpu:
                model = cls.ORIG_MODEL.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    quantization_config=quantization_config,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    **kwargs,
                )
                logger.info("WeightOnlyQuant bitsandbytes done.")
                return model
            logger.info("{} device is used.".format(device_map))
            if load_in_8bit or load_in_4bit or quantization_config is not None:
                torch_dtype = kwargs.get("torch_dtype", torch.float16 if use_xpu else torch.float32)
                if use_xpu:
                    assert (
                        torch_dtype == torch.float16
                    ), "Intel GPU only support torch.float16 now, will support other dtype in future!"
                    kwargs["torch_dtype"] = torch_dtype
            if load_in_4bit:
                if quantization_config is None:
                    quantization_config = RtnConfig(
                        bits=4,
                        compute_dtype=(
                            torch.float32
                            if (use_cpu and not CpuInfo().bf16 and torch_dtype == torch.bfloat16)
                            else convert_dtype_torch2str(torch_dtype)
                        ),
                        weight_dtype="int4_clip" if use_cpu else "int4_fullrange",
                    )
                else:
                    assert (
                        "4" in quantization_config.weight_dtype
                        and convert_dtype_str2torch(quantization_config.compute_dtype) == torch_dtype
                    ), "Quantization_config.weight_dtype should be 'nf4' , 'int4', 'int4_fullrange', 'int4_clip', "
                    f"'fp4', 'fp4_e2m1' and compute_dtype should be {torch_dtype}."
            elif load_in_8bit:
                if quantization_config is None:
                    quantization_config = RtnConfig(
                        bits=8,
                        compute_dtype=(
                            torch.float32
                            if (use_cpu and not CpuInfo().bf16 and torch_dtype == torch.bfloat16)
                            else convert_dtype_torch2str(torch_dtype)
                        ),
                        weight_dtype="int8",
                    )
                else:
                    assert (
                        quantization_config.weight_dtype == "int8" and quantization_config.compute_dtype == torch_dtype
                    ), f"Quantization_config.weight_dtype should be 'int8' and compute_dtype should be {torch_dtype}."

        if isinstance(
            quantization_config,
            (RtnConfig, AwqConfig, TeqConfig, GPTQConfig, AutoRoundConfig),
        ):
            logger.info("Applying Weight Only Quantization.")
            if use_xpu:
                # TODO: if low_cpu_mem_uasge is True, gptj will have accuracy issue on CPU device.
                kwargs["low_cpu_mem_usage"] = True
                kwargs["device_map"] = "cpu"
                try:
                    model = cls.ORIG_MODEL.from_pretrained(
                        pretrained_model_name_or_path,
                        *model_args,
                        config=config,
                        **kwargs,
                    )
                    model.config.update({"low_cpu_mem_usage": True})
                except NotImplementedError:
                    logger.info(
                        "Failed to load models with `low_cpu_mem_usage` specified, "
                        "will fall to traditional load method with higher memory consumption."
                    )
                    kwargs["low_cpu_mem_usage"] = False
                    config.torchscript = True if quantization_config.quant_method.value in ["teq", "awq"] else False
                    model = cls.ORIG_MODEL.from_pretrained(
                        pretrained_model_name_or_path,
                        *model_args,
                        config=config,
                        **kwargs,
                    )
                    model.config.update({"low_cpu_mem_usage": False})
                    quantization_config.post_init_xpu()
            else:
                kwargs["low_cpu_mem_usage"] = True
                config.torchscript = True if quantization_config.quant_method.value in ["teq", "awq"] else False
                model = cls.ORIG_MODEL.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **kwargs,
                )
                model.config.update({"low_cpu_mem_usage": True})
                quantization_config.post_init_cpu()
            model.eval()

            if use_xpu:
                import intel_extension_for_pytorch

                assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"
                quantization_config.update(**{"device": "xpu"})
                quantization_config.post_init_xpu()
            if (
                not torch.cuda.is_available() or device_map == "cpu" or device_map == torch.device("cpu")
            ) and model.config.model_type == "chatglm":
                model = model.float()
            model = convert_to_quantized_model(model, quantization_config, device=device_map)
            quantization_config.remove_redundant_parameters()
            model.config.quantization_config = quantization_config
        else:
            model = cls.ORIG_MODEL.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
            if (
                not torch.cuda.is_available() or device_map == "cpu" or device_map == torch.device("cpu")
            ) and model.config.model_type == "chatglm":
                model = model.float()

            model.eval()

        # add quantization_config and save_low_bit to pretrained model dynamically
        model.device_map = device_map
        model.quantization_config = quantization_config

        model.save_pretrained = types.MethodType(save_low_bit, model)
        logger.info("WeightOnlyQuant done.")
        return model

    @classmethod
    def load_low_bit(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a low bit optimized model (including INT4, INT5 and INT8) from a saved ckpt.

        :param pretrained_model_name_or_path: str value, Path to load the optimized model ckpt.
        # :param optimize_model: boolean value, Whether to further optimize the low_bit llm model.
        #                        Default to be True.
        :return: a model instance
        """
        from accelerate.big_modeling import init_empty_weights
        from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
        from transformers.generation.configuration_utils import GenerationConfig
        from transformers.modeling_utils import _add_variant, get_checkpoint_shard_files, no_init_weights
        from transformers.models.auto.auto_factory import _get_model_class
        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.utils import ContextManagers, cached_file, download_url, extract_commit_hash, is_remote_url

        # Autofactory
        kwargs_orig = copy.deepcopy(kwargs)
        # modules_to_not_convert = kwargs.pop("modules_to_not_convert", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        # Maybe needed when extract_local_archive_file
        subfolder = kwargs.pop("subfolder", "")
        variant = kwargs.pop("variant", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        torch_dtype = kwargs.pop("torch_dtype", "auto")
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        resume_download = kwargs.pop("resume_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        token = kwargs.pop("token", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        revision = kwargs.pop("revision", "main")
        commit_hash = kwargs.pop("_commit_hash", None)
        _fast_init = kwargs.pop("_fast_init", True)
        device_map = kwargs.pop("device_map", "auto")
        use_safetensors = kwargs.pop("use_safetensors", None)
        kwarg_attn_imp = kwargs.pop("attn_implementation", None)

        # lm-eval device map is dictionary
        device_map = device_map[""] if isinstance(device_map, dict) and "" in device_map else device_map

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        if use_auth_token is not None:
            logger.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
                "Please use `token` instead."
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        use_cpu = True if device_map == torch.device("cpu") or device_map == "cpu" else False
        use_xpu = True if device_map == torch.device("xpu") or device_map == "xpu" else False

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        config = kwargs.pop("config", None)
        if kwarg_attn_imp is not None and config._attn_implementation != kwarg_attn_imp:
            config._attn_implementation = kwarg_attn_imp

        quantization_config = config.quantization_config

        if quantization_config["quant_method"] == "rtn":
            quantization_config = RtnConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "awq":
            quantization_config = AwqConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "teq":
            quantization_config = TeqConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "gptq":
            quantization_config = GPTQConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "autoround":
            quantization_config = AutoRoundConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "static":
            quantization_config = StaticQuantConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "dynamic":
            quantization_config = DynamicQuantConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "qat":
            quantization_config = QuantAwareTrainingConfig.from_dict(quantization_config)
        elif quantization_config["quant_method"] == "sq":
            quantization_config = SmoothQuantConfig.from_dict(quantization_config)
        assert quantization_config is not None, "Detect this model is not a low-bit model."

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent)
                # to get the commit hash as soon as possible.
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    "config.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        has_remote_code = hasattr(config, "auto_map") and cls.ORIG_MODEL.__name__ in config.auto_map

        has_local_code = type(config) in cls.ORIG_MODEL._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code,
            pretrained_model_name_or_path,
            has_local_code,
            has_remote_code,
        )
        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.ORIG_MODEL.__name__]
            model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs_orig)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.ORIG_MODEL.__name__)
            else:
                cls.ORIG_MODEL.register(config.__class__, model_class, exist_ok=True)
        elif type(config) in cls.ORIG_MODEL._model_mapping.keys():
            model_class = _get_model_class(config, cls.ORIG_MODEL._model_mapping)

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_NAME, variant),
                    )
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_NAME, variant),
                    )
                # only for inc sq
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant("quantized_model.pt", variant),
                    )
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant("quantized_model.pt", variant),
                    )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                    )
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                    )
                    is_sharded = True
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    )
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_NAME, variant),
                    )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                    )
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                    )
                    is_sharded = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                if use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)
                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or "
                                f"{_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                            )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path,
                                filename,
                                **cached_file_kwargs,
                            )
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True

                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                        }
                        if variant is not None and has_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_NAME,
                            **has_file_kwargs,
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                f" {variant}. Use `variant=None` to load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(WEIGHTS_NAME, variant)}."
                            )
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                    ) from e
            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict,
        #    by checking its first weights entry that is of a floating type
        #    - we assume all floating dtype weights are of the same dtype
        # we also may have config.torch_dtype available, but we won't rely on it till v5
        # Pretrained Model
        if quantization_config.quant_method in ["static", "dynamic", "qat"]:
            model = model_class(config, *model_args, **kwargs)
            from neural_compressor.utils.pytorch import load

            weights_file = os.path.join(
                os.path.abspath(os.path.expanduser(pretrained_model_name_or_path)),
                WEIGHTS_NAME,
            )
            q_model = load(weights_file, model, dataloader=None)
            del model
            return q_model
        if quantization_config.quant_method in ["sq"]:
            print("Loading SmoothQuant model from: ", pretrained_model_name_or_path)
            from intel_extension_for_transformers.transformers.llm.quantization.sq_utils import TSModelCausalLMForITREX

            q_model = torch.jit.load(os.path.join(pretrained_model_name_or_path, "quantized_model.pt"))
            origin_model_type = config.model_type
            if origin_model_type in ["chatglm", "qwen", "baichuan"]:
                config.model_type = "qwen2"
            q_model = TSModelCausalLMForITREX(q_model, config=config)
            q_model.config.model_type = origin_model_type
            return q_model
        dtype_orig = None
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    if (
                        hasattr(config, "torch_dtype")
                        and config.torch_dtype is not None
                        and config.torch_dtype != "auto"
                    ):
                        torch_dtype = config.torch_dtype
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        else:
                            torch_dtype = torch.float32
                else:
                    assert False, f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}'

            dtype_orig = model_class._set_default_torch_dtype(torch_dtype)
        if quantization_config.compute_dtype is None:
            if use_xpu:
                quantization_config.compute_dtype = (
                    "fp16"
                    if (torch_dtype is None or torch_dtype == torch.bfloat16)
                    else convert_dtype_torch2str(torch_dtype)
                )
            else:
                quantization_config.compute_dtype = (
                    "fp32"
                    if (
                        torch_dtype is None
                        or (not CpuInfo().bf16 and torch_dtype == torch.bfloat16)
                        or (torch_dtype == torch.float16)
                    )
                    else convert_dtype_torch2str(torch_dtype)
                )
        else:
            if (not CpuInfo().bf16 and quantization_config.compute_dtype == "bf16") or (
                use_cpu and quantization_config.compute_dtype == "fp16"
            ):
                quantization_config.compute_dtype = "fp32"

        if quantization_config.scale_dtype is None:
            quantization_config.scale_dtype = "fp32"
        if quantization_config.scale_dtype not in ["fp32", "fp16", "bf16"]:
            logger.warning("scale_dtype only supports fp32, bf16, fp16.")
            quantization_config.scale_dtype = "fp32"
            logger.warning("fp32 scale_dtype is used, please change the config.json if you don't want to use it.")

        # weight dtype is higher priority than bits in config.json when both existed.
        if quantization_config.weight_dtype is None:
            if quantization_config.bits == 4:
                if use_xpu:
                    quantization_config.weight_dtype = "int4_fullrange"
                else:
                    quantization_config.weight_dtype = "int4_clip"
                logger.info(
                    "{} quantization weight_dtype is used due to bits is 4 in config.json.".format(
                        quantization_config.weight_dtype
                    )
                )
            elif quantization_config.bits == 8:
                quantization_config.weight_dtype = "int8"
                logger.info(
                    "{} quantization weight_dtype is used due to bits is 8 in config.json.".format(
                        quantization_config.weight_dtype
                    )
                )
            else:
                logger.warning("bits number only supports 4, 8.")
                quantization_config.weight_dtype = "int4_clip"
                logger.warning(
                    "int4_clip weight_dtype is used, please change the config.json if you don't want to use it."
                )
        else:
            if quantization_config.weight_dtype not in [
                "int4_fullrange",
                "int4_clip",
                "int8",
                "fp8_e5m2",
                "fp8_e4m3",
                "nf4",
                "fp4_e2m1_bnb",
                "fp4_e2m1",
            ]:
                logger.warning("Please provide the correct bits number or weight_dtype in config.json.")
                raise ValueError(
                    "weight_dtype must be a string in "
                    "'int8', 'int4', 'int4_fullrange', 'int4_clip', 'nf4', "
                    "'fp4', 'fp4_e2m1', 'fp8', 'fp8_e5m2, fp8_e4m3'"
                )
            else:
                logger.info("{} quantization weight_dtype is used.".format(quantization_config.weight_dtype))

        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(init_empty_weights())

        with ContextManagers(init_contexts):
            model = model_class(config, *model_args, **kwargs)

        if quantization_config.weight_dtype not in [
            "fp8_e5m2",
            "fp8_e4m3",
        ]:
            model = build_woq_model(model, quantization_config)
        else:
            model = replace_linear(
                model,
                quantization_config=quantization_config,
                device="cpu" if device_map == "auto" else device_map,
                empty_weights=True,
            )

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_archive_file)
            loaded_state_dict_keys = list(state_dict.keys())

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        if is_ipex_available() and quantization_config.use_ipex:
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear as ipex_linear

            def replace_ipex_cpu_woq_linear(model, current_name=[]):
                for name, module in model.named_children():
                    current_name.append(name)
                    if isinstance(module, WeightOnlyLinear):
                        weight_dtype = {
                            4: ipex.quantization.WoqWeightDtype.INT4,
                            8: ipex.quantization.WoqWeightDtype.INT8,
                        }
                        compute_dtype = {
                            "fp32": ipex.quantization.WoqLowpMode.NONE,  # follow the activation datatype.
                            "bf16": ipex.quantization.WoqLowpMode.BF16,
                            "fp16": ipex.quantization.WoqLowpMode.FP16,
                            "int8": ipex.quantization.WoqLowpMode.INT8,
                        }

                        ipex_qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                            weight_dtype=weight_dtype[quantization_config.bits],
                            lowp_mode=compute_dtype[quantization_config.compute_dtype],
                            act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                            group_size=quantization_config.group_size,
                        )
                        tmp_linear = torch.nn.Linear(
                            module.in_features,
                            module.out_features,
                            True if hasattr(module, "bias") else False,
                        )
                        tmp_linear.qconfig = ipex_qconfig_mapping.global_qconfig
                        target_linear = ipex_linear.from_float_and_int4_weight(
                            mod=tmp_linear,
                            qweight=state_dict.pop(".".join(current_name) + ".ipex_weight"),
                            scales=state_dict.pop(".".join(current_name) + ".ipex_scales"),
                            zero_points=state_dict.pop(".".join(current_name) + ".ipex_zeros"),
                            bias=(
                                state_dict.pop(".".join(current_name) + ".ipex_bias")
                                if ".".join(current_name) + ".ipex_bias" in state_dict
                                else None
                            ),
                            group_size=quantization_config.group_size,
                            g_idx=(
                                state_dict.pop(".".join(current_name) + ".ipex_g_idx")
                                if ".".join(current_name) + ".ipex_g_idx" in state_dict
                                else None
                            ),
                        )
                        setattr(model, name, target_linear)
                    else:
                        replace_ipex_cpu_woq_linear(module, current_name)
                    current_name.pop()

            replace_ipex_cpu_woq_linear(model)
            model.load_state_dict(state_dict, strict=False, assign=True)
        else:
            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = model_class._load_pretrained_model(
                model,
                None,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                sharded_metadata=sharded_metadata,
                _fast_init=_fast_init,
                low_cpu_mem_usage=True,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                keep_in_fp32_modules=[],
            )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if (
            quantization_config.weight_dtype
            not in [
                "fp8_e5m2",
                "fp8_e4m3",
            ]
            and not quantization_config.use_ipex
        ):
            model = replace_linear(
                model,
                quantization_config=quantization_config,
                device="cpu" if device_map == "auto" else device_map,
                empty_weights=True,
            )

        if (not use_xpu and torch_dtype == torch.float16) or (
            not use_xpu and not CpuInfo().bf16 and torch_dtype == torch.bfloat16
        ):
            model.to(dtype=torch.float32)

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except (OSError, TypeError):
                pass
        for param in model.parameters():
            param.requires_grad_(False)
        if device_map == "xpu":
            model = model.to("xpu")
        model.quantization_config = quantization_config
        model.save_pretrained = types.MethodType(save_low_bit, model)
        return model


class AutoModelForCausalLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForCausalLM


class AutoModel(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModel


class AutoModelForSeq2SeqLM(_BaseQBitsAutoModelClass):
    ORIG_MODEL = transformers.AutoModelForSeq2SeqLM