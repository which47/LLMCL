import os
import json
from prettytable import PrettyTable
from utils.metrics import (
    eval_FOMC,
    eval_SciQA,
    eval_CStance,
    eval_NumGLUE,
    eval_PapyrusF,
    eval_20Minuten,
    eval_MeetingBank,
    eval_medmcqa,
    eval_jecqa,
)
from transformers import HfArgumentParser
import logging
from utils.arg_configs import EvalArguments
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

EVAL_FUNCs = {
        "FOMC": eval_FOMC,
        "ScienceQA": eval_SciQA,
        "C-STANCE": eval_CStance,
        "NumGLUE-cm": eval_NumGLUE,
        "NumGLUE-ds": eval_PapyrusF,
        "20Minuten": eval_20Minuten,
        "MeetingBank": eval_MeetingBank,
        "medmcqa": eval_medmcqa,
        "jecqa": eval_jecqa,
    }

def main():
    parser = HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # check if the json_dir exists
    for j_dir in args.json_dirs:
        if not os.path.exists(j_dir):
            raise ValueError(f"Path {j_dir} does not exist")
    if args.cl_method == "mtl" and len(args.json_dirs) != 1:
        raise ValueError(f"Multi-task learning should only have one json_dir")
    
    logger.info(f"Evaluting args: {args}")
    logger.info(f"Make sure your `increment_order` is in the same order as you train the datasets!!")
    results = {}
    for json_dir in args.json_dirs:
        data = json.load(open(json_dir, "r"))
        trained_task = os.path.split(json_dir)[1].split(".json")[0]
        
        if trained_task not in results:
            results[trained_task] = {}
        
        for infer_task, infer_result in data.items():
            # try:
                eval_func = EVAL_FUNCs[infer_task]
                prompts = []
                answers = []
                generated_texts = []
                
                for item in infer_result:
                    prompts.append(item["prompt"])
                    answers.append(item["answer"])
                    generated_texts.append(item["generated_text"])

                try:
                # special case for `20Minuten`
                    if infer_task == "20Minuten":
                        eval_result = eval_func(prompts, generated_texts, answers)
                    else:
                        eval_result = eval_func(generated_texts, answers)
                    logger.info(f"Inference result {json_dir} on task` {infer_task}`: {eval_result}")
                except Exception as e:
                    eval_result = None
                    logger.error(f"Error processing file {json_dir} with dataset `{infer_task}`: {e}")
                    continue
                results[trained_task][infer_task] = get_res(eval_result, infer_task)   
            
    # Print table
    table = PrettyTable()
    table.field_names = [args.cl_method] + args.increment_order
    print(results)
    for row_name in args.increment_order:
        if row_name not in results:
            logger.warning(f"Missing result for `{row_name}`, skipping")
            continue
        row = [row_name]
        for col_name in args.increment_order:
            if col_name not in results[row_name]:
                row.append("-")
                logger.warning(f"Missing result for `{row_name}` on `{col_name}`")
            else:
                row.append(results[row_name][col_name])
        table.add_row(row)
    
    print(table)
    save_tabel(table, args.save_path)

    
def save_tabel(table, path):
    "save table to .csv file"
    base_path = os.path.split(path)[0]
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    from pandas import DataFrame
    df = DataFrame([table.field_names] + table._rows)
    df.to_csv(path, index=False, header=False)
    
    logger.info(f"Results saved to {path}")
    
def get_res(result: dict, name: str):
    if result is None:
        return -1
    if name == "20Minuten":
        return round(result['sari'][0]['sari'] / 100, 3)
    elif name in ["C-STANCE", "FOMC", "NumGLUE-cm", "NUmGLUE-ds", "ScienceQA", "medmcqa", "jecqa"]:
        return round(result['accuracy'], 3)
    elif name == "Py150":
        return round(result['similarity'], 3)
    elif name == "MeetingBank":
        return round(result['rouge-L'], 3)


if __name__ == "__main__":
    main()