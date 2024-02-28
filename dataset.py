import os
from typing import List, Dict
from transformers import PreTrainedTokenizerBase
from datasets import concatenate_datasets as con
from datasets import load_dataset, Dataset
from utils.prompter import Prompter
"""
    To create a dataset, you need to specify the path of the dataset and the name of the dataset, and make sure the coulmn name contains "input_ids, attention_mask, labels" for training,
    and "prompt, answer" for testing.
"""
prompter = Prompter(template_name="alpaca")

def create_datasets(dataset_names: List[str],
                    trace_path: str = "./data_files",
                    jecqa_path: str = "./data_files/JEC-QA",
                    medmcqa_path: str = "./data_files/medmcqa", tokenizer: object = None, cutoff_len: int = 1024,
                    num_samples: int = 5000,
                    cache_dir: str = './dataset_cache',
                    add_eos_token=False) -> Dict[str, Dataset]:
    """
        return train and eval dataset which consists of tasks and their corresponding datasets
        data_path: the path of the dataset
        dataset_names: the names of the datasets
        tokenizer: the tokenizer used to tokenize the dataset
        cutoff_len: the max length of the input sequence
    """
    if not os.path.exists(trace_path):
        raise ValueError(f"path {trace_path} does not exist")
    if not os.path.exists(jecqa_path) or not os.path.exists(medmcqa_path):
        raise ValueError(f"path {jecqa_path} or {medmcqa_path} does not exist")

    data = {}
    for i, name in enumerate(dataset_names):

        if name == 'medmcqa':
            data[name] = load_dataset("json", data_files=os.path.join(medmcqa_path, "train.json"),
                                      cache_dir=cache_dir, split="train")  # feature: id, question, opa,
            # opb, opc, opd, cop, choice_type, exp, subject_name, topic_name
            data[name] = data[name].filter(lambda x: x['choice_type'] == 'single')
            data[name] = data[name].map(medmcqa_preprocess).select(range(num_samples)).remove_columns(
                ['question', 'exp', 'cop', 'opa', 'opb', 'opc', 'opd', 'subject_name', 'topic_name', 'id',
                 'choice_type']
            ).shuffle()

        elif name == 'jecqa':
            data_files = {"train": [os.path.join(jecqa_path, '0_train.json'), os.path.join(jecqa_path, '1_train.json')]}
            data[name] = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split="train")
            # feature: answer, id, option_list, statement, subject, type
            data[name] = data[name].filter(lambda x: len(x['answer']) == 1).select(range(num_samples))
            data[name] = data[name].map(jecQA_preprocess).remove_columns(
                ['id', 'option_list', 'statement', 'subject', 'type']
            ).shuffle()

        elif name in ["C-STANCE", "FOMC", "MeetingBank", "NumGLUE-cm", "ScienceQA", "20Minuten"]:
            data[name] = load_dataset("json", data_files=os.path.join(trace_path, name, "train.json"),
                                      cache_dir=cache_dir, split="train").shuffle()
        else:
            raise ValueError(f"dataset {name} is not supported")

    if isinstance(tokenizer, PreTrainedTokenizerBase):
        for name, d in data.items():
            data[name] = d.map(tokenize_fn(tokenizer, cutoff_len, add_eos_token=add_eos_token))

    return data


def create_test_datasets(dataset_names: List[str],
                         trace_path: str = "./data_files",
                         jecqa_path: str = "./data_files/JEC-QA",
                         medmcqa_path: str = "./data_files/medmcqa",
                         num_samples: int = 500,
                         cache_dir: str = './dataset_cache') -> Dict[str, Dataset]:
    """
        Dict[str, Dataset]: the key is the name of the dataset, the value is the dataset
        feature: prompt, answer
    """
    data = {}
    for i, name in enumerate(dataset_names):

        if name == 'medmcqa':
            data[name] = load_dataset("json", data_files=os.path.join(medmcqa_path, 'dev.json'),
                                      cache_dir=cache_dir, split="train").filter(lambda x: x['choice_type'] == 'single')
            len_of_data = len(data[name])
            select_range = range(len_of_data - min(num_samples, len_of_data), len_of_data)
            data[name] = data[name].select(select_range)
            data[name] = data[name].map(medmcqa_preprocess).remove_columns(
                ['question', 'exp', 'cop', 'opa', 'opb', 'opc', 'opd', 'subject_name', 'topic_name', 'id',
                 'choice_type']
            ).shuffle()

        elif name == 'jecqa':
            data_files = {"train": [os.path.join(jecqa_path, '0_train.json'), os.path.join(jecqa_path, '0_train.json')]}
            data[name] = (load_dataset("json", data_files=data_files, cache_dir=cache_dir, split="train").
                          filter(lambda x: len(x['answer']) == 1))
            len_of_data = len(data[name])
            select_range = range(len_of_data - min(num_samples, len_of_data), len_of_data)
            data[name] = data[name].select(select_range).map(jecQA_preprocess).remove_columns(
                ['id', 'option_list', 'statement', 'subject', 'type']
            ).shuffle()

        elif name in ["C-STANCE", "FOMC", "MeetingBank", "NumGLUE-cm", "ScienceQA", "20Minuten"]:
            data[name] = load_dataset("json", data_files=os.path.join(trace_path, name, "test.json"),
                                      cache_dir=cache_dir, split="train")
            len_of_data = len(data[name])
            select_range = range(len_of_data - min(num_samples, len_of_data), len_of_data)
            data[name] = data[name].select(select_range).shuffle()

    return data


def create_joint_datasets(trace_path: str, dataset_names: List[str], jecqa_path: str = "./data_files/JEC-QA",
                          medmcqa_path: str = "./data_files/medmcqa", tokenizer=None, cutoff_len=512):
    """
        mix up training datasets
    """
    data = create_datasets(trace_path=trace_path, dataset_names=dataset_names, jecqa_path=jecqa_path,
                           medmcqa_path=medmcqa_path, tokenizer=tokenizer, cutoff_len=cutoff_len)
    data = con([data[name] for name in dataset_names]).shuffle()
    return data


def tokenize_fn(tokenizer, cutoff_len, add_eos_token=True, train_on_inputs=True) -> callable:
    def tokenize_data_point(data_point):
        full_prompt = prompter.generate_prompt(data_point['prompt'], None, data_point['answer'])
        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len, add_eos_token=add_eos_token)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point['prompt'])
            tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt['input_ids'])
            if add_eos_token:
                user_prompt_len -= 1
                tokenized_full_prompt['labels'] = [
                    -100 
                ] * user_prompt_len + tokenized_full_prompt['labels'][user_prompt_len:]
        return tokenized_full_prompt

    return tokenize_data_point


def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def medmcqa_preprocess(datapoint):
    int2choice_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    sample = {"prompt": TASK_PROMT["medmcqa"] + datapoint["question"] + "\n" + "Choices: \n" + \
                        f"A.{datapoint['opa']}, B.{datapoint['opb']}, C.{datapoint['opc']}, D.{datapoint['opd']}\n Answer:",
              "answer": f"{int2choice_map[datapoint['cop']]}"}
    return sample


def jecQA_preprocess(datapoint):
    op_list = datapoint['option_list']
    sample = {"prompt": TASK_PROMT["jecqa"] + datapoint['statement'] + "\n" + "选项: \n" + \
                        f"A. {op_list['A']}，B. {op_list['B']}，C. {op_list['C']}，D. {op_list['D']}\n 答案:",
              "answer": f"{datapoint['answer'][0]}"}
    return sample


TASK_PROMT = {
    "FOMC": "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "C-STANCE": "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA": "Choose an answer for the following question and give your reasons.\n\n",
    "NumGLUE-cm": "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    "20Minuten": "Provide a simplified version of the following paragraph in German.\n\n",
    "medmcqa": "Solve the following medical problem by choosing the correct answer from following four choices.\nQuestion:\n",
    "jecqa": "根据以下法律问题，从选项A，B，C，D中选择一项正确的答案\n问题："
}

GENERAL_PROMPT = {
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
    "response_split": "### Response:"
}