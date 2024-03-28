import argparse
import json
import logging
import math
import os
import random
import datetime
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
import torch

import transformers
from transformers import (
    SchedulerType,
    get_scheduler,
)

from configuration_gpt_neo import GPTNeoConfig
from modeling_gpt_neo import GPTNeoForCausalLM

device = torch.device("hpu")
torch.cuda.current_device = lambda: None
torch.cuda.set_device = lambda x: None

# os.environ['TPC_RUNNER'] = str(1)
# os.environ['HABANA_PROFILE'] = str(1)
from torch.profiler import profile, ProfilerActivity

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=6,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        default=6,
        help="Total number of test steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument("--n_positions", type=int, default=1024, help="The maximum sequence length.")
    parser.add_argument("--n_embd_per_head", type=int, default=64, help="Dimensionality of the embeddings and hidden states per head.")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of hidden layers in the Transformer encoder.")
    parser.add_argument("--n_head", type=int, default=12, help="Number of attention heads for each attention layer in the Transformer encoder.")
    parser.add_argument("--use_lazy_mode", default='True', type=lambda x: x.lower() == 'true',
                        help='Whether to run model in lazy or eager execution mode, default=True for lazy mode')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    n_embd = args.n_embd_per_head * args.n_head
    configuration = GPTNeoConfig(max_position_embeddings=args.n_positions, hidden_size=n_embd, num_layers=args.n_layer, num_heads=args.n_head, attention_types=[[["global", "local"], 2]])
    model = GPTNeoForCausalLM(configuration)
    config = model.config
    print(config)

    model = model.to(device)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    current_time = datetime.datetime.now()
    timestamp = f'{current_time.day}_{current_time.hour}_{current_time.minute}'
    def trace_handler(prof):
        # output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        # print(output)
        prof.export_chrome_trace(f"./traces/gpt_neo_{args.n_layer}l_{args.n_positions}pos_{timestamp}.json")

    for epoch in range(args.num_train_epochs):
        # model.train()
        train_loss = 0
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.HPU],
        #     record_shapes=True,
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        #     on_trace_ready=trace_handler
        # ) as prof:
        # # if True:
        #     for step in range(args.max_train_steps):
        #         input_ids = torch.randint(low=0, high=20000, size=(args.per_device_train_batch_size, args.block_size), dtype=torch.long).to(device)
        #         attention_mask = torch.ones(args.per_device_train_batch_size, args.block_size, dtype=torch.float).to(device)
        #         labels = torch.randint(low=0, high=20000, size=(args.per_device_train_batch_size, args.block_size), dtype=torch.long).to(device)

        #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        #         loss = outputs.loss
        #         train_loss += loss.detach().float()
        #         optimizer.zero_grad()

        #         loss.backward()
        #         if args.use_lazy_mode:
        #             htcore.mark_step()

        #         optimizer.step()
        #         if args.use_lazy_mode:
        #             htcore.mark_step()
        #         lr_scheduler.step()
        #         ht.hpu.synchronize()
        #         prof.step()
        # print(f'train_loss {train_loss.item() / args.max_train_steps} ')

    test_loss = 0
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.HPU],
    #     record_shapes=True,
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    #     on_trace_ready=trace_handler
    # ) as prof:
    if True:
        model.eval()
        with torch.no_grad():
            for step in range(args.max_test_steps):
                input_ids = torch.randint(low=0, high=20000, size=(args.per_device_eval_batch_size, args.block_size), dtype=torch.long).to(device)
                attention_mask = torch.ones(args.per_device_eval_batch_size, args.block_size, dtype=torch.float).to(device)
                labels = torch.randint(low=0, high=20000, size=(args.per_device_eval_batch_size, args.block_size), dtype=torch.long).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                htcore.mark_step()
                test_loss += loss
                # ht.hpu.synchronize()
                # prof.step()
    print(f'test_loss {test_loss / args.max_test_steps} ')

    ht.hpu.synchronize()
    print('finish !!!')


if __name__ == "__main__":
    main()
