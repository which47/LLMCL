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
from fire import Fire
import re
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def list_of_strings(arg):
    return arg.split(',')


def main(json_dir: str,  # directory of output json files, format: f"{FOMC}_{C-STANCE}.json"
         increment_order: list_of_strings  # incremental learning order
         ):
    dataset_map = {
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
    increment_order = list_of_strings(increment_order)
    assert os.path.isdir(json_dir), f"{json_dir} does not exist"
    json_files = os.listdir(json_dir)

    dataset_names = list(dataset_map.keys())

    result_dict = {}
    for current_name in json_files:
        if not current_name.endswith(".json"):
            continue

        match = re.match(r"(.+)_(.+)\.json", current_name)

        if match:
            first_name = match.group(1)
            second_name = match.group(2)
        else:
            raise ValueError(f"Invalid json file name: {current_name}")

        logger.info(f"Trained on {first_name}, Testing on {second_name}")

        if first_name not in result_dict:
            result_dict[first_name] = {}

        json_path = os.path.join(json_dir, current_name)
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        prompts = json_data["prompts"]
        answers = json_data["answers"]
        outputs = json_data["outputs"]

        eval_result = (
            dataset_map[second_name](outputs, answers)
            if second_name != "20Minuten"
            else dataset_map[second_name](prompts, outputs, answers)
        )
        print(f"Results: {eval_result}")
        eval_result = get_res(eval_result, second_name)
        result_dict[first_name][second_name] = eval_result

    # Print table
    table = PrettyTable()
    table.field_names = [""] + increment_order
    if first_name.lower() == "base": # zero-shot
        table.add_row(["base"] + [result_dict["Base"][name] for name in increment_order])
    elif first_name.lower() == "multi": # multi task
        table.add_row(["mtl"] + [result_dict["multi"][name] for name in increment_order])
    else :
        for first_name in increment_order:
            row = [first_name]
            for second_name in increment_order:
                try:
                    row.append(result_dict[first_name][second_name])
                except:
                    row.append("-")
            table.add_row(row)
    print(table)
    save_tabel(table, json_dir)
    
def save_tabel(table, path):
    "save table to .csv file"
    save_path = os.path.join(path, "results.csv")
    from pandas import DataFrame
    df = DataFrame([table.field_names] + table._rows)
    df.to_csv(save_path, index=False, header=False)
    
    print(f"\n------Save table to {save_path}!------\n")
    
def get_res(result: dict, name: str):
    if name == "20Minuten":
        return round(result['sari'][0]['sari'] / 100, 3)
    elif name in ["C-STANCE", "FOMC", "NumGLUE-cm", "NUmGLUE-ds", "ScienceQA", "medmcqa", "jecqa"]:
        return round(result['accuracy'], 3)
    elif name == "Py150":
        return round(result['similarity'], 3)
    elif name == "MeetingBank":
        return round(result['rouge-L'], 3)


if __name__ == "__main__":
    Fire(main)
