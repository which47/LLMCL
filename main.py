import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from utils.arg_configs import get_args, CLArguments, TuningArguments, DataArguments
from dataclasses import asdict
from get_dataset import get_datasets, DataCollector
from utils.functions import set_all_seed
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    train_args, cl_args, tuning_args, data_args = get_args()
    set_all_seed(tuning_args.manual_seed)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=tuning_args.load_8bit,
    )
    model = AutoModelForCausalLM.from_pretrained(
        tuning_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(tuning_args.model_name_or_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # prepare datasets
    train_datasets = get_datasets(**asdict(data_args), tokenizer=tokenizer, split='train')   
    valid_datasets = get_datasets(**asdict(data_args), tokenizer=tokenizer, split='eval')


    train_args = dict(
        model=model if not tuning_args.load_8bit else prepare_model_for_kbit_training(model),
        args=train_args,
        train_dataset=train_datasets,
        eval_dataset=valid_datasets,
        tokenizer=tokenizer,
        cl_args=cl_args,
        tuning_args=tuning_args,
        data_args=data_args,
        data_collator=DataCollector(tokenizer, padding=True, max_length=data_args.max_length)
    )

    if cl_args.cl_method.lower() == 'seq':
        from method.BaseTrainerCL import BaseTrainerCL
        cl_trainer = BaseTrainerCL(**train_args)
    elif cl_args.cl_method.lower() == 'ewc':
        from method.EWC import EWCTrainer
        cl_trainer = EWCTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'er':
        from method.ER import ERTrainer
        cl_trainer = ERTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'gem':
        from method.GEM import GEMTrainer
        cl_trainer = GEMTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'agem':
        from method.AGEM import AveGEMTrainer
        cl_trainer = AveGEMTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'l2p':
        from method.L2P import L2PTrainer
        cl_trainer = L2PTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'pp':
        from method.PP import PPTrainer
        cl_trainer = PPTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'ilora':
        from method.ILORA import ILoRATrainer
        cl_trainer = ILoRATrainer(**train_args)
    elif cl_args.cl_method.lower() == 'mtl':
        from method.MTL import MTLTrainer
        cl_trainer = MTLTrainer(**train_args)
    elif cl_args.cl_method.lower() == 'one':
        from method.ONE import ONETrainer
        cl_trainer = ONETrainer(**train_args)
    else:
        ValueError(f"continual learning method: {cl_args.cl_method} not implement yet")
    
    cl_trainer.continual_learning()

if __name__ == '__main__':

    main()

