#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
import evaluate
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.37.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


from transformers.integrations import WandbCallback
class AddPerplexityWandbCallback(WandbCallback):
    # def __init__(self):
    #     super().__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return

        if state.is_world_process_zero:
            if "loss" in logs:
                self._wandb.log({"train/perplexity": math.exp(logs["loss"])})

            if "eval_loss" in logs:
                self._wandb.log({"eval/perplexity": math.exp(logs["eval_loss"])})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    current_time = time.time()
    local_time = time.localtime(current_time)
    time_stamp = f'{local_time.tm_mon}-{local_time.tm_mday}-{local_time.tm_hour}-{local_time.tm_min}'
    training_args.run_name = f'{training_args.run_name}-{time_stamp}'

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if model_args.model_name_or_path:
    #     torch_dtype = (
    #         model_args.torch_dtype
    #         if model_args.torch_dtype in ["auto", None]
    #         else getattr(torch, model_args.torch_dtype)
    #     )
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         token=model_args.token,
    #         trust_remote_code=model_args.trust_remote_code,
    #         torch_dtype=torch_dtype,
    #         low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    #     )
    # else:
    #     model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
    #     n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    #     logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    from configuration_gpt_neo import GPTNeoConfig
    from modeling_gpt_neo import GPTNeoForCausalLM
    # import json
    # config = GPTNeoConfig()
    # with open('gpt_neo_125m_config.json') as c_f:
    #     c_dic = json.load(c_f)
    # config.update(c_dic)
    # Initializing a model (with random weights)
    model = GPTNeoForCausalLM(config)

    # torch_dtype = (
    #     model_args.torch_dtype
    #     if model_args.torch_dtype in ["auto", None]
    #     else getattr(torch, model_args.torch_dtype)
    # )
    # model = GPTNeoForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     token=model_args.token,
    #     trust_remote_code=model_args.trust_remote_code,
    #     torch_dtype=torch_dtype,
    #     low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    # )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]

        print('\n------------ !!! ------------ ')
        num_processes = 4
        num_steps_per = train_dataset.num_rows // training_args.per_device_train_batch_size // num_processes
        print(f'num_steps_per: {num_steps_per}; save_steps {training_args.save_steps}\n ')

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        callbacks=[AddPerplexityWandbCallback],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)




    # # ------------------------ accelerate ------------------------
    # from argparse import Namespace
    # from pathlib import Path

    # from accelerate import Accelerator, DistributedType
    # from accelerate.utils import ProjectConfiguration
    # from huggingface_hub import Repository
    # from torch.optim import AdamW
    # from torch.utils.data.dataloader import DataLoader

    # from transformers import get_scheduler

    # def get_grouped_params(model, args, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    #     params_with_wd, params_without_wd = [], []
    #     for n, p in model.named_parameters():
    #         if any(nd in n for nd in no_decay):
    #             params_without_wd.append(p)
    #         else:
    #             params_with_wd.append(p)
    #     return [
    #         {"params": params_with_wd, "weight_decay": args.weight_decay},
    #         {"params": params_without_wd, "weight_decay": 0.0},
    #     ]

    # def get_lr():
    #     return optimizer.param_groups[0]["lr"]

    # def setup_logging(args):
    #     # project_name = args.model_ckpt.split("/")[-1]
    #     project_name = 'gpt-125m-wikitext'
    #     logger = logging.getLogger(__name__)
    #     log_dir = Path(args.output_dir) / "log/"
    #     log_dir.mkdir(exist_ok=True)
    #     filename = f"debug_{accelerator.process_index}.log"
    #     logging.basicConfig(
    #         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #         datefmt="%m/%d/%Y %H:%M:%S",
    #         level=logging.INFO,
    #         handlers=[logging.FileHandler(log_dir / filename), logging.StreamHandler()],
    #     )
    #     if accelerator.is_main_process:  # we only want to setup logging once
    #         current_time = time.time()
    #         local_time = time.localtime(current_time)
    #         time_stamp = f'{local_time.tm_mon}-{local_time.tm_mday}-{local_time.tm_hour}-{local_time.tm_min}'
    #         accelerator.init_trackers(project_name, init_kwargs={"wandb": {"name": f"base-{time_stamp}"}})
    #         run_name = accelerator.trackers[0].run.name
    #         logger.setLevel(logging.INFO)
    #         datasets.utils.logging.set_verbosity_info()
    #         transformers.utils.logging.set_verbosity_info()
    #     else:
    #         run_name = ""
    #         logger.setLevel(logging.ERROR)
    #         datasets.utils.logging.set_verbosity_error()
    #         transformers.utils.logging.set_verbosity_error()
    #     return logger, run_name

    # def log_metrics(step, metrics):
    #     if accelerator.is_main_process and step % training_args.logging_steps == 0:
    #         logger.info(f"Step {step}: {metrics}")
    #         accelerator.log(metrics, step)

    # def evaluate_fn(args):
    #     model.eval()
    #     losses = []
    #     for step, batch in enumerate(eval_dataloader):
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #         loss = outputs.loss.repeat(args.per_device_eval_batch_size)
    #         losses.append(accelerator.gather(loss))
    #         # if args.eval_steps > 0 and step >= args.eval_steps:
    #         #     break
    #     losses = torch.cat(losses)
    #     # loss = losses[: eval_dataloader.dataset.current_size].mean()
    #     loss = losses.mean()
    #     try:
    #         perplexity = torch.exp(loss)
    #     except OverflowError:
    #         perplexity = float("inf")
    #     return loss.item(), perplexity.item()

    # # Accelerator
    # config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir="log")
    # accelerator = Accelerator(log_with=["wandb"], project_config=config)
    # acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    # # training_args = Namespace(**vars(training_args), **acc_state)
    # samples_per_step = accelerator.state.num_processes * training_args.per_device_train_batch_size
    # set_seed(training_args.seed)

    # # Clone model repository
    # # if accelerator.is_main_process:
    # #     hf_repo = Repository(training_args.output_dir, clone_from=training_args.model_ckpt)

    # # Logging
    # logger, run_name = setup_logging(training_args)
    # logger.info(accelerator.state)

    # # # Checkout new branch on repo
    # # if accelerator.is_main_process:
    # #     hf_repo.git_checkout(run_name, create_branch_ok=True)

    # # # Load model and tokenizer
    # # model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    # # if args.gradient_checkpointing:
    # #     model.gradient_checkpointing_enable()
    # # tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

    # # Load dataset and dataloader
    # data_collator = default_data_collator
    # train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)

    # # Prepare the optimizer and learning rate scheduler
    # optimizer = AdamW(get_grouped_params(model, training_args), lr=training_args.learning_rate)
    # training_args.max_steps = int(training_args.num_train_epochs) * len(train_dataloader) // accelerator.num_processes
    # training_args.warmup_steps = int(training_args.max_steps * training_args.warmup_ratio)
    # lr_scheduler = get_scheduler(
    #     name=training_args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=training_args.warmup_steps,
    #     num_training_steps=training_args.max_steps,
    # )
    # accelerator.register_for_checkpointing(lr_scheduler)

    # # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader
    # )

    # if accelerator.is_main_process:
    #     print('\n------------ !!! ------------ ')
    #     print(f'num_batches: {len(train_dataloader)}; max_steps: {training_args.max_steps} \n ')

    # # load in the weights and states from a previous save
    # # if training_args.resume_from_checkpoint:
    # #     if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
    # #         accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
    # #         accelerator.load_state(training_args.resume_from_checkpoint)
    # #         path = os.path.basename(training_args.resume_from_checkpoint)
    # #     else:
    # #         # Get the most recent checkpoint
    # #         dirs = [f.name for f in os.scandir(training_args.output_dir) if f.is_dir() and "step" in str(f)]
    # #         dirs.sort(key=os.path.getctime)
    # #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
    # #     # Extract the step of the checkpoint to continue from there
    # #     training_difference = os.path.splitext(path)[0]
    # #     resume_step = int(training_difference.replace("step_", ""))

    # # Train model
    # completed_steps = 0
    # starting_epoch = 0
    # for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
    #     model.train()
    #     t_start = time.time()
    #     loss_tracking = 0
    #     for step, batch in enumerate(train_dataloader, start=1):
    #         if training_args.resume_from_checkpoint and step < resume_step:
    #             continue  # we need to skip steps until we reach the resumed step
            
    #         input_ids = batch['input_ids']
    #         attention_mask = batch['attention_mask']
    #         labels = batch['labels']
    #         loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss
    #         avg_loss = accelerator.gather(loss.repeat(training_args.per_device_train_batch_size)).mean()
    #         loss_tracking += avg_loss.item() / training_args.gradient_accumulation_steps
    #         log_metrics(completed_steps, {"train/samples": completed_steps * samples_per_step, "train/loss_per_step": loss.item()})
    #         loss = loss / training_args.gradient_accumulation_steps
    #         if step % training_args.gradient_accumulation_steps != 0:
    #             # Prevent backward from doing gradient all_reduce in every step
    #             if accelerator.distributed_type == DistributedType.MULTI_GPU:
    #                 with model.no_sync():
    #                     accelerator.backward(loss)
    #             else:
    #                 accelerator.backward(loss)
    #         else:
    #             lr = get_lr()
    #             accelerator.backward(loss)
    #             accelerator.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             elapsed_time = time.time() - t_start
    #             # tflops = compute_tflops(elapsed_time, accelerator, training_args)
    #             tflops = 0
    #             log_metrics(
    #                 completed_steps,
    #                 {
    #                     "train/steps": completed_steps,
    #                     "train/loss": loss_tracking,
    #                     "train/lr": lr,
    #                     "train/time_per_iteration": elapsed_time,
    #                 },
    #             )
    #             t_start = time.time()
    #             loss_tracking = 0
    #             completed_steps += 1
    #         # if step % training_args.save_steps == 0:
    #         if completed_steps % training_args.eval_steps == 0:            
    #             logger.info("Evaluating and saving model checkpoint")
    #             eval_loss, perplexity = evaluate_fn(training_args)
    #             log_metrics(completed_steps, {"eval/loss": eval_loss, "eval/perplexity": perplexity})
    #             # accelerator.wait_for_everyone()
    #             # save_dir = os.path.join(training_args.output_dir, f"step_{step}")
    #             # accelerator.save_state(save_dir)
    #             # if accelerator.is_main_process:
    #             #     hf_repo.push_to_hub(commit_message=f"step {step}")
    #             model.train()
    #         if completed_steps >= training_args.max_steps:
    #             break

    # # Evaluate and save the last checkpoint
    # logger.info("Evaluating and saving model after training")
    # # eval_loss, perplexity = evaluate(training_args)
    # # log_metrics(step, {"loss/eval": eval_loss, "perplexity": perplexity})
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(training_args.output_dir, save_function=accelerator.save)
    # save_dir = os.path.join(training_args.output_dir, f"step_{step}")
    # accelerator.save_state(save_dir)
    # # if accelerator.is_main_process:
    # #     hf_repo.push_to_hub(commit_message="final model")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
