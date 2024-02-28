import argparse
import os

import sys
import torch

from transformers import (SchedulerType,

                          TrainingArguments,

                          LlamaForCausalLM,

                          LlamaTokenizer,

                          DataCollatorForSeq2Seq,

                          AutoConfig,

                          HfArgumentParser

                          )

from utils.arguments import DatasetArguments, CLArguments

from peft import prepare_model_for_int8_training
import dataset

from method.BaseTrainerCL import BaseTrainerCL
from dataset import create_datasets


def parse_args():

    def list_of_strings(arg):

        return arg.split(',')


    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument('--data_path',

                        type=str,

                        required=True,

                        help='Path to the training dataset, a single data path.')


    parser.add_argument('--dataset_name',

                        type=list_of_strings,

                        help='Dataset to be used.'

                        )


    parser.add_argument('--data_output_path',

                        type=str,

                        default='/tmp/data_files/',
                        help=

                        'Where to store the data-related files such as shuffle index. This needs to be on a local '

                        'storage of a node ('

                        'not on a shared storage)'

                        )


    parser.add_argument(

        "--model_name_or_path",

        type=str,
        help=

        "Path to pretrained model or model identifier from huggingface.co/models.",

        required=True,

    )


    parser.add_argument("--num_train_epochs",

                        type=int,

                        default=5,

                        help="""in the paper

                                'TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models http://arxiv.org/abs/2310.06762',

                                better set the epochs to 5 """,

                        )


    parser.add_argument("--per_device_train_batch_size",

                        type=int,

                        default=1,

                        help="Batch size (per device) for the training dataloader.",

                        )


    parser.add_argument("--per_device_eval_batch_size",

                        type=int,

                        default=8,

                        help="Batch size (per device) for the evaluation dataloader.",

                        )


    parser.add_argument("--max_prompt_len",

                        type=int,

                        default=512,

                        help="The maximum sequence length.",

                        )


    parser.add_argument("--max_ans_len",

                        type=int,

                        default=512,

                        help="The maximum sequence length.",

                        )


    parser.add_argument("--learning_rate",

                        type=float,

                        default=1e-4,
                        help=

                        "Initial learning rate (after the potential warmup period) to use.",

                        )


    parser.add_argument("--weight_decay",

                        type=float,

                        default=0.,

                        help="Weight decay to use.")


    parser.add_argument(

        "--gradient_accumulation_steps",

        type=int,

        default=4,
        help=

        "Number of updates steps to accumulate before performing a backward/update pass.",

    )


    parser.add_argument("--lr_scheduler_type",

                        type=str,

                        default="constant_with_warmup",

                        help="The scheduler type to use.",

                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",

                                 "constant_with_warmup"],

                        )

    parser.add_argument("--num_warmup_steps",

                        type=int,

                        default=0,

                        help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--warmup_ratio",

                        type=float,

                        default=0.2,

                        help="Ratio of total training steps used for warmup.")

    parser.add_argument("--output_dir",

                        type=str,

                        default=None,

                        help="Where to store the model.")

    parser.add_argument("--local_rank",

                        type=int,

                        default=-1,

                        )

    parser.add_argument("--seed",

                        type=int,

                        default=42,

                        help="A seed for reproducible training.")


    parser.add_argument('--gradient_checkpointing',
                        action='store_true',

                        help='Enable HF gradient checkpointing for model.')


    parser.add_argument('--cl_method',

                        default=None,

                        help='continual learning method used')


    parser.add_argument('--use_wandb',

                        default=True,

                        help='whether use wandb to log'

                        )

    parser.add_argument('--cut_off_len',

                        default=512,
                        help='cut off length for input'

                        )

    parser.add_argument('--load_8bit',

                        default=False,

                        help='whether load 8bit model'

                        )

    parser.add_argument('--resume_from_checkpoint',

                        default=None,

                        help='whether resume from checkpoint'

                        )

    parser.add_argument('--eval_steps',

                        default=500,

                        help='eval steps'

                        )

    parser.add_argument('--save_steps',

                        default=2000,

                        help='save steps'

                        )

    parser.add_argument('--adapter',

                        default=None,

                        choices=['lora', 'prompt', 'prefix'],

                        required=True,

                        help='adapter type')

    parser.add_argument('--lora_r',

                        default=8,

                        help='low rank of LoRA adapter')

    parser.add_argument('--lora_alpha',

                        default=16,

                        help='alpha of LoRA adapter')

    parser.add_argument('--lora_target_modules',

                        default=None,

                        help='target modules of LoRA adapter')

    parser.add_argument('--lora_dropout',

                        default=0.05,

                        help='dropout of LoRA adapter')

    parser.add_argument('--lora_bias',
                        default='none',

                        help='bias of LoRA adapter')

    parser.add_argument('--bottleneck_size',

                        default=256,

                        help='bottleneck size of bottleneck adapter')

    parser.add_argument('--non_linearity',
                        default='tanh',

                        help='non linearity of bottleneck adapter')

    parser.add_argument('--adapter_dropout',

                        default=0.0,

                        help='dropout of bottleneck adapter')

    parser.add_argument('--use_parallel_adapter',

                        default=False,

                        help='whether use parallel adapter')

    parser.add_argument('--use_adapterp',

                        default=False,

                        help='whether use adapterp')

    parser.add_argument('--target_modules',

                        default=['q_proj', 'k_proj', 'v_proj'],

                        help='target modules of bottleneck adapter')

    parser.add_argument('--scaling',

                        default=1.0,

                        help='scaling of bottleneck adapter')

    parser.add_argument('--num_virtual_tokens',

                        default=30,

                        help='num virtual tokens of prefix tuning')


    args = parser.parse_args()
    return args



def main():

    args = parse_args()

    # args.num_examples = re.findall(r'\d+', args.dataset_name)

    device_map = 'auto'

    world_size = int(os.environ.get('WORLD_SIZE', 1))

    ddp = world_size != 1

    if ddp:

        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,

                                             load_in_8bit=args.load_8bit,

                                             torch_dtype=torch.bfloat16,

                                             device_map=device_map,

                                            #  cache_dir="/root/autodl-tmp/cache_dir",

                                             local_files_only=True,

                                             )

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,

                                            #    cache_dir="/root/autodl-tmp/cache_dir",

                                               fast_tokenizer=True)

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token_id = 0

    tokenizer.padding_side = "left"

    if args.cl_method is None:
        pass


    if not ddp and torch.cuda.device_count() > 1:

        model.is_parallelizable = True

        model.model_parallel = True


    # prepare dataset

    data = create_datasets(args.dataset_name, args.data_path, tokenizer=tokenizer, cutoff_len=args.cut_off_len)

    train_dataset = {name: data[name] for name in args.dataset_name}


    train_args = dict(

        model=model if not args.load_8bit else prepare_model_for_int8_training(model),

        args=TrainingArguments(

            per_device_train_batch_size=args.per_device_train_batch_size,

            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,

            lr_scheduler_type=args.lr_scheduler_type,

            warmup_ratio=args.warmup_ratio,

            weight_decay=args.weight_decay,

            warmup_steps=args.num_warmup_steps,
            num_train_epochs=args.num_train_epochs,

            bf16=True,

            logging_steps=10,

            optim='adamw_torch',

            evaluation_strategy='no',

            save_strategy='no',

            eval_steps=args.eval_steps,

            save_steps=args.save_steps,
            output_dir=args.output_dir,

            save_total_limit=3,

            load_best_model_at_end=True,

            ddp_find_unused_parameters=False if ddp else None,

            group_by_length=False,

            report_to='wandb' if args.use_wandb else None,

            do_eval=False,

        ),
        train_dataset=train_dataset,

        # eval_dataset=eval_dataset,

        tokenizer=tokenizer,

        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,

                                             pad_to_multiple_of=8,

                                             return_tensors="pt",

                                             padding=True),
        adapter=args.adapter,
        cl_method=args.cl_method,

        # LoRA adapter
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,

        lora_bias=args.lora_bias,

        # prompt tuning

        num_virtual_tokens=args.num_virtual_tokens,

        # num_examples=args.num_examples

    )

    if args.cl_method is None:

        cl_trainer = BaseTrainerCL(**train_args)

    elif args.cl_method.lower() == 'ewc':

        from method.EWC import EWCTrainer

        cl_trainer = EWCTrainer(**train_args)

    elif args.cl_method.lower() == 'er':

        from method.ER import ERTrainer

        cl_trainer = ERTrainer(**train_args)

    elif args.cl_method.lower() == 'gem':

        from method.GEM import GEMTrainer

        cl_trainer = GEMTrainer(**train_args)

    elif args.cl_method.lower() == 'agem':

        from method.AGEM import AveGEMTrainer

        cl_trainer = AveGEMTrainer(**train_args)

    elif args.cl_method.lower() == 'l2p':

        from method.L2P import L2PTrainer

        cl_trainer = L2PTrainer(**train_args)

    elif args.cl_method.lower() == 'pp':

        from method.PP import PPTrainer

        cl_trainer = PPTrainer(**train_args)

    elif args.cl_method.lower() == 'ilora':

        from method.ILORA import ILoRATrainer
        cl_trainer = ILoRATrainer(**train_args)

    elif args.cl_method.lower() == 'mtl':

        from method.MTL import MTLTrainer

        cl_trainer = MTLTrainer(**train_args)

    elif args.cl_method.lower() == 'one':

        from method.ONE import ONETrainer

        cl_trainer = ONETrainer(**train_args)
    else:
        ValueError(f"continual learning method: {args.cl_method} not implement yet")


    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":

        model = torch.compile(model)

    cl_trainer.continual_learning()



if __name__ == '__main__':

    main()

