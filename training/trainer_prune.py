# coding=utf-8
# Copyright (C) 2024 Xiaomi Corporation. All rights reserved.
#
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import importlib.metadata
import inspect
import logging
import os
import warnings
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

from tqdm import tqdm

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import gc

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import (Dataset, IterableDataset, RandomSampler,
                              SequentialSampler)
from transformers import Trainer, __version__
from transformers.data.data_collator import DataCollator
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (DefaultFlowCallback,
                                           ProgressCallback, TrainerCallback,
                                           TrainerState)
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
                                        enable_full_determinism,
                                        find_executable_batch_size,
                                        get_last_checkpoint, set_seed)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (is_accelerate_available, is_apex_available,
                                is_datasets_available, is_in_notebook,
                                is_peft_available, is_safetensors_available,
                                is_sagemaker_mp_enabled,
                                is_torch_xla_available)

from pruning.module_dict import *
from pruning.prune import pruning_schedule, pruning_structured
from pruning.scorer import ActsScorer, TaylorScorer

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = \
        version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (smp_forward_backward,
                                               smp_forward_only, smp_gather,
                                               smp_nested_concat)
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate import __version__ as accelerate_version
    from accelerate import skip_first_batches
    from accelerate.utils import (DistributedDataParallelKwargs,
                                  DistributedType, GradientAccumulationPlugin,
                                  load_fsdp_model, load_fsdp_optimizer,
                                  save_fsdp_model, save_fsdp_optimizer)

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}

if TYPE_CHECKING:
    import optuna

logging.basicConfig(format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class TrainerPrune(Trainer):
    # Those are used as methods of the Trainer in examples.
    from transformers.trainer_pt_utils import (_get_learning_rate, log_metrics,
                                               metrics_format, save_metrics,
                                               save_state)

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        pruning_args: Any = None,
        model_args: Any = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        config: Any = None,
    ):
        super().__init__(model,
                         args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers,
                         preprocess_logits_for_metrics,)
        self.model_args = model_args
        self.pruning_args = pruning_args
        self.config = config

        def freeze_params(module):
            # print("freezing all params in", module)
            for param in module.parameters():
                param.requires_grad = False

        def unfreeze_params(module):
            # print("unfreezing params in", module)
            for param in module.parameters():
                param.requires_grad = True
        
        def unfreeze_params_by_name(target_name_list):
            for name, param in self.model.named_parameters():
                for target_name in target_name_list:
                    if target_name in name:
                        param.requires_grad = True
                        break

    def calculate_parameters(self):
        model = self.model
        n_params = 0
        for n, p in model.named_parameters():
            if self.is_world_process_zero():
                padding = "-"*(60-len(n))
                logger.info("  %s %s %d" % (n, padding, p.nelement()))
            n_params += p.nelement()
        print("  Total parameters: %.2fB" % (n_params/1e9))
        return n_params

    def eval_acts(self, model):
        num_processes_bak = self.accelerator.state.num_processes
        self.accelerator.state.num_processes = 1  # HACK: avoid split dataloader into n_gpu shards
        eval_dataloader = self.get_eval_dataloader()
        model.eval()
        mha_scores, mlp_scores = None, None
        acts_scorer = ActsScorer(self.config.model_type, self.config.num_hidden_layers)
        # region a pseudo eval here
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Prune Eval")):
            batch["output_pruning_acts"] = True
            batch["output_hidden_states"] = True
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) and torch.no_grad():
                outputs = model(**batch)
            acts_scorer.accumulate(outputs["mha_pruning_acts"], outputs["mlp_pruning_acts"])
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
            if step > self.pruning_args.prune_eval_samples:
                break
        # endregion
        acts_scorer.compute_score()
        mha_scores, mlp_scores = acts_scorer.get_score()
        self.accelerator.free_memory()
        self.accelerator.state.num_processes = num_processes_bak
        return mha_scores, mlp_scores

    def eval_taylor(self, model):
        num_processes_bak = self.accelerator.state.num_processes
        self.accelerator.state.num_processes = 1  # HACK: avoid split dataloader into n_gpu shards
        eval_dataloader = self.get_eval_dataloader()
        model.train()
        mha_scores, mlp_scores = None, None
        taylor_scorer = TaylorScorer(model, self.config.model_type, self.config.num_hidden_layers)
        # region a pseudo eval here
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Prune Eval")):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            taylor_scorer.accumulate(model)
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
        # endregion
        taylor_scorer.compute_score(model)
        mha_scores, mlp_scores = taylor_scorer.get_score()
        self.accelerator.free_memory()
        self.accelerator.state.num_processes = num_processes_bak
        return mha_scores, mlp_scores

    def eval_and_prune(self, model, args):
        if self.is_world_process_zero():
            prune_eval_fn = {
                "acts": self.eval_acts,
                "taylor": self.eval_taylor,
            }
            mha_scores, mlp_scores = prune_eval_fn[self.pruning_args.prune_mode](model)

            pruning_config, all_layer_mha_keep_indices, all_layer_mlp_keep_indices = pruning_schedule(
                self.pruning_args.current_shot - 1, self.pruning_args.prune_shots,
                self.current_layer, self.pruning_args.target_layer,
                self.current_mha_dim, self.pruning_args.target_mha_dim,
                self.config.num_attention_heads,
                self.current_mlp_dim, self.pruning_args.target_mlp_dim,
                mha_scores, mlp_scores,
                self.pruning_args.prune_head_num,
                self.pruning_args.prune_head_dim,
            )
            del mha_scores, mlp_scores
            self.accelerator.free_memory()
        else:
            pruning_config, all_layer_mha_keep_indices, all_layer_mlp_keep_indices = None, None, None

        if is_torch_xla_available():
            xm.rendezvous("pruning_schedule on device 0")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        broadcast_pruning_obj = [
            pruning_config, all_layer_mha_keep_indices, all_layer_mlp_keep_indices]
        dist.broadcast_object_list(broadcast_pruning_obj, src=0)
        pruning_config, all_layer_mha_keep_indices, all_layer_mlp_keep_indices = broadcast_pruning_obj

        next_layer = pruning_config["next_layer"]
        next_head_num = pruning_config["next_head_num"]
        next_head_dim = pruning_config["next_head_dim"]
        next_mha_dim = pruning_config["next_mha_dim"]
        next_mlp_dim = pruning_config["next_mlp_dim"]
        model = pruning_structured(
            model,
            model_type=self.config.model_type,
            next_layer=next_layer,
            next_head_num=next_head_num,
            next_head_dim=next_head_dim,
            next_mha_dim=next_mha_dim,
            next_mlp_dim=next_mlp_dim,
            all_layer_mha_keep_indices=all_layer_mha_keep_indices,
            all_layer_mlp_keep_indices=all_layer_mlp_keep_indices,
        )
        self.current_layer, self.current_mha_dim, self.current_mlp_dim = \
            next_layer, next_mha_dim, next_mlp_dim

        # update config
        self.config.num_hidden_layers = self.current_layer
        self.config.num_attention_heads = next_head_num
        self.config.num_key_value_heads = next_head_num  # TODO: support GQA 
        self.config.mha_inter_size = self.current_mha_dim
        self.config.intermediate_size = self.current_mlp_dim
        logger.info(f"Config updated after pruning.\n{self.config}")
        model.zero_grad(set_to_none=True)
        model.update_config(self.config)

        self.calculate_parameters()

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                "train() received got unexpected keyword arguments: "
                f"{', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) \
                if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # region prune
        self.current_layer, self.current_mha_dim, self.current_mlp_dim = \
            self.config.num_hidden_layers, self.config.mha_inter_size, self.config.intermediate_size
        if self.pruning_args.current_shot <= self.pruning_args.prune_shots:
            # region place model on cuda
            need_grad = self.pruning_args.prune_mode == "taylor"
            print("Dtype:", self.model.dtype)
            self.model.to(torch.bfloat16)
            print("Dtype:", self.model.dtype)
            model = self._wrap_model(self.model, training=need_grad)
            if len(self.accelerator._models) == 0 and model is self.model:
                self.accelerator.prepare_model(model, evaluation_mode=not need_grad)
                self.model = model
            # endregion

            self.eval_and_prune(self.model, args)
            self.accelerator.free_memory()
            # updated self.model
            self.model.to(torch.float32)

            model_reloaded = True
        # endregion

        # If model was re-initialized or pruned, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
