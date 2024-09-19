import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import roc_auc_score
def load_gsm8k_dataset(split="test"):
    print(f"Loading GSM8K {split} dataset...")
    dataset = load_dataset("gsm8k", "main", split=split)
    print(f"GSM8K {split} dataset loaded successfully")
    return dataset

def load_mmlu_dataset(subject, split="test"):
    print(f"Loading MMLU {subject} {split} dataset from 'cais/mmlu'...")
    dataset = load_dataset("cais/mmlu", subject, split=split)
    print(f"MMLU {subject} {split} dataset loaded successfully")
    return dataset

def load_strategyqa_dataset(split="test"):
    print(f"Loading StrategyQA {split} dataset from 'ChilleD/StrategyQA'...")
    dataset = load_dataset("ChilleD/StrategyQA", split=split)
    print(f"StrategyQA {split} dataset loaded successfully")
    return dataset


def load_popqa_dataset(split="test"):
    print(f"Loading PopQA {split} dataset from 'akariasai/PopQA'...")
    dataset = load_dataset("akariasai/PopQA", split=split).select(range(5))
    print(f"PopQA {split} dataset loaded successfully")
    return dataset

def setup_llama3_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    cache_dir = "/mnt/hdd/brian/hub"
    os.environ['TFHUB_CACHE_DIR'] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer's pad token is set
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model and tokenizer loaded successfully from {model_name}")
    return tokenizer, model, device

def evaluate_exact_match(predictions, references):
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_set = set(pred.strip().lower().split(', '))
        ref_set = set(ref.strip().lower().split(', '))
        if pred_set == ref_set:
            correct += 1
    accuracy = correct / len(predictions)
    print(f"Exact Match Accuracy: {accuracy:.2%}")

def calculate_ece(predictions, references, confidence_scores, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for pred, ref, conf in zip(predictions, references, confidence_scores):
        bin_idx = np.digitize(conf, bin_boundaries) - 1
        # Ensure bin_idx is within the valid range
        bin_idx = min(bin_idx, num_bins - 1)

        pred_set = set(pred.strip().lower().split(', '))
        ref_set = set(ref.strip().lower().split(', '))
        accuracies[bin_idx] += int(pred_set == ref_set)
        confidences[bin_idx] += conf
        bin_counts[bin_idx] += 1

    accuracies = np.divide(accuracies, bin_counts, out=np.zeros_like(accuracies), where=bin_counts != 0)
    confidences = np.divide(confidences, bin_counts, out=np.zeros_like(confidences), where=bin_counts != 0)

    ece = np.sum((bin_counts / len(predictions)) * np.abs(accuracies - confidences))
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

def calculate_auroc(predictions, references, confidence_scores):
    labels = [1 if set(pred.strip().lower().split(', ')) == set(ref.strip().lower().split(', ')) else 0 for pred, ref in zip(predictions, references)]
    auroc = roc_auc_score(labels, confidence_scores)
    print(f"Area Under ROC (AUROC): {auroc:.4f}")
 

def run_inference(tokenizer, model, device, dataset, question_key, answer_key):
    args_start_prompt_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    args_end_prompt_footer = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    predictions = []
    references = []
    confidences = []

    for i, data in enumerate(dataset):
        input_text = data[question_key]
        reference_answer = data[answer_key]
        prompt = f"""{args_start_prompt_header}Provide your 20 best guesses for the following question. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses; not a complete sentence, just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text} Guesses:{args_end_prompt_footer}"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.pad_token_id)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(output_text)
        references.append(reference_answer)

        # Confidence is not directly available without some processing or adaptation from logits
        with torch.no_grad():
            last_hidden_states = model(**inputs, output_hidden_states=True).hidden_states[-1]
            logits = model.lm_head(last_hidden_states)
            confidence_score = torch.softmax(logits, dim=-1).max().item()
            confidences.append(confidence_score)

        print(f"Question {i+1}: {input_text}")
        print(f"Expected Answer: {reference_answer}")
        print(f"Predicted Answer: {output_text}")

    return predictions, references, confidences

def main():
    split = "test"

    # Load GSM8K dataset
    # gsm8k_dataset = load_gsm8k_dataset(split=split)

    # Load MMLU Professional Law dataset
    # mmlu_prf_law_dataset = load_mmlu_dataset(subject="professional_law", split=split)

    # Load MMLU Business Ethics dataset
    # mmlu_biz_ethics_dataset = load_mmlu_dataset(subject="business_ethics", split=split)

    # # Load StrategyQA dataset
    # strategyqa_dataset = load_strategyqa_dataset(split=split)


    # Load PopQA dataset
    popqa_dataset = load_popqa_dataset(split=split)


    # for i,e in enumerate(popqa_dataset):
    #     print ('ie',i,e)
    #     print ('question',e['question'])
    #     print ('labels',e['possible_answers'])
    #     import json
    #     print(json.loads(e['possible_answers'])[0])
    #     if i == 5:
    #         sys.exit()

    tokenizer, model, device = setup_llama3_model()

    # # Run inference on GSM8K dataset
    # print("\n=== GSM8K Dataset Inference ===")
    # gsm8k_predictions, gsm8k_references = run_inference(tokenizer, model, device, gsm8k_dataset)
    # evaluate_exact_match(gsm8k_predictions, gsm8k_references)

    # # Run inference on MMLU Professional Law dataset
    # print("\n=== MMLU Professional Law Dataset Inference ===")
    # mmlu_prf_law_predictions, mmlu_prf_law_references = run_inference(tokenizer, model, device, mmlu_prf_law_dataset)
    # evaluate_exact_match(mmlu_prf_law_predictions, mmlu_prf_law_references)

    # # Run inference on MMLU Business Ethics dataset
    # print("\n=== MMLU Business Ethics Dataset Inference ===")
    # mmlu_biz_ethics_predictions, mmlu_biz_ethics_references = run_inference(tokenizer, model, device, mmlu_biz_ethics_dataset)
    # evaluate_exact_match(mmlu_biz_ethics_predictions, mmlu_biz_ethics_references)

    # # Run inference on StrategyQA dataset
    # print("\n=== StrategyQA Dataset Inference ===")
    # strategyqa_predictions, strategyqa_references = run_inference(tokenizer, model, device, strategyqa_dataset, question_key="question", answer_key="answer")
    # evaluate_exact_match(strategyqa_predictions, strategyqa_references)

    # Run inference on PopQA dataset
    print("\n=== PopQA Dataset Inference ===")
    popqa_predictions, popqa_references, confidences = run_inference(tokenizer, model, device, popqa_dataset, question_key="question", answer_key="possible_answers")
    evaluate_exact_match(popqa_predictions, popqa_references)
    calculate_ece(popqa_predictions, popqa_references, confidences)
    calculate_auroc(popqa_predictions, popqa_references, confidences)


if __name__ == "__main__":
    main()