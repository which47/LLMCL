import re
from rouge import Rouge
from fuzzywuzzy import fuzz
from datasets import load_metric
from nltk.translate.bleu_score import sentence_bleu
import evaluate


########################
# BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1.,))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def calculate_bleu(results, data, gram):
    bleus = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(results)
    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    print(type(str1), type(str2))
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def calculate_rouge(results, data):
    rouges = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)
    avg_rouge = sum(rouges) / len(results) if rouges else 0.0
    return avg_rouge

def calculate_rouge_(results, data):
    rouge_evaluator = evaluate.load("rouge")
    eval_results = []
    eval_refs = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        eval_results.append(prediction)
        eval_refs.append(target)
    rouge_scores = rouge_evaluator.compute(predictions=eval_results, references=eval_refs, rouge_types=["rougeL"])
    return rouge_scores['rougeL']

def caculate_accuracy(results, data):
    scores = 0
    for output_id in range(len(results)):
        target = data[output_id]
        prediction = results[output_id]
        if prediction == "" or target == "":
            continue
        if prediction == target:
            scores += 1
    avg_score = scores / len(results)
    return avg_score



def f1_score(list1, list2):
    # TP: item in list1 and list2
    # FP: item in list1 but not in list2
    # TN: item not in list1 and list2
    # FN: item in list2 but not in list1
    num_TP = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                num_TP += 1
                break
    precision = num_TP / len(list1)
    recall = num_TP / len(list2)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall / (precision + recall))


def calculate_f1(results, data):
    scores = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if len(prediction) == 0 or len(target) == 0:
            continue
        score = f1_score(target, prediction)
        scores.append(score)
    avg_score = sum(scores) / len(results)
    return avg_score



def calculate_sari(inputs, results, data):
    sari = load_metric("sari")
    result = sari.compute(sources=inputs, predictions=results, references=[[label] for label in data]), # one reference for each prediction
    return result


def eval_20Minuten(input_sequences, predicted_sequences, ground_truths):
    sari = calculate_sari(input_sequences, predicted_sequences, ground_truths)
    evaluation_result = {"sari": sari}
    return evaluation_result

def eval_medmcqa(predicted_sequences, ground_truths):
    predicted_sequences = postprocess_choice_acc(predicted_sequences)
    ground_truths = postprocess_choice_acc(ground_truths)
    
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result

def eval_jecqa(predicted_sequences, ground_truths):
    predicted_sequences = postprocess_choice_acc(predicted_sequences)
    ground_truths = postprocess_choice_acc(ground_truths)
    
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result

def eval_CStance(predicted_sequences, ground_truths):
    predicted_sequences = postprocess_choice_acc(predicted_sequences)
    ground_truths = postprocess_choice_acc(ground_truths)
    
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result


def eval_FOMC(predicted_sequences, ground_truths):
    predicted_sequences = postprocess_choice_acc(predicted_sequences)
    ground_truths = postprocess_choice_acc(ground_truths)
    
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result


def eval_MeetingBank(predicted_sequences, ground_truths):
    # bleu_1 = calculate_bleu(predicted_sequences, ground_truths, 1)
    # bleu_4 = calculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = calculate_rouge_(predicted_sequences, ground_truths)
    # evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge}
    evaluation_result = {"rouge-L": rouge}
    return evaluation_result


def eval_NumGLUE(predicted_sequences, ground_truths):
    predicted_sequences = postprocess_choice_num(predicted_sequences)
    ground_truths = postprocess_choice_num(ground_truths)
    
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result


def resolve(dataset: list):
    keyword_list = []
    for datium in dataset:
        keyword_list.append(datium.split(" , "))
    return keyword_list


def eval_PapyrusF(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    f1 = calculate_f1(outputs, gts)
    evaluation_result = {"F1": f1}
    return evaluation_result

def postprocess_choice_acc(predicted_sequences):
    outputs = []
    for output in predicted_sequences:
        if not output:
            outputs.append("")
            continue
        match = re.search(r"[A-D]", output)
        if match:
            outputs.append(match.group(0))
        else:
            outputs.append("")
    return outputs

def postprocess_choice_num(predicted_sequences):
    outputs = []
    for output in predicted_sequences:
        if not output:
            outputs.append("")
            continue    
        match = re.search(r"\d+\.?\d*", output)
        if match:
            outputs.append(match.group(0))
        else:
            outputs.append("")
    return outputs


def resolve_sciQA(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        if len(datium) >= 3:
            answers.append(datium[0])  # the first char is the answer. e.g. A, B,...
            reasonings.append(datium[2:])  # A/nBecause...
        elif 1 <= len(datium) < 3:
            answers.append(datium[0])
            reasonings.append("")
        else:
            answers.append("")
            reasonings.append("")
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval_SciQA(predicted_sequences, ground_truths):
    outputs = resolve_sciQA(predicted_sequences)
    gts = resolve_sciQA(ground_truths)
    outputs["answers"] = postprocess_choice_acc(outputs["answers"])
    gts["answers"] = postprocess_choice_acc(gts["answers"])
    
    # bleu_1 = calculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    # bleu_4 = calculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    # rouge = calculate_rouge_(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    # evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, 
    #                     #  "rouge-L": rouge,
    #                      "accuracy": accuracy}
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
