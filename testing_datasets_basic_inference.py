

import os
os.environ['HF_DATASETS_CACHE'] = '/mnt/hdd/brian/datasets'
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
def load_gsm8k_dataset(split="test"):
    print(f"Loading GSM8K {split} dataset...")
    dataset = load_dataset("gsm8k", "main", split=split)
    print(f"GSM8K {split} dataset loaded successfully")
    return dataset

def load_triviaqa_dataset(split="test"):
    split = 'validation'
    print(f"Loading TriviaQA {split} dataset from 'mandarjoshi/trivia_qa'...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
    print(f"TriviaQA {split} dataset loaded successfully", len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(2000))
    filtered_dataset = dataset.filter(lambda example: example['entity_pages']['wiki_context'])
    dataset = filtered_dataset
    # dataset = dataset.shuffle(seed=42).select(range(4))
    dataset = dataset.shuffle(seed=42).select(range(100))
    # sys.exit()
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
    # os.environ['TFHUB_CACHE_DIR'] = cache_dir
    # os.environ['HF_DATASETS_CACHE'] = cache_dir
    # os.environ['HF_HOME'] = '/mnt/hdd/brian/hub'
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

def evaluate_extended_exact_match(predictions, list_of_reference_lists):
    correct = 0
    for pred, refs in zip(predictions, list_of_reference_lists):
        pred_set = set(pred.strip().lower().split(', '))
        matched = False
        print ('refs',refs)
        refs_aliases = refs['normalized_aliases']
        for ref in refs_aliases:
            ref_set = set(ref.strip().lower().split(', '))
            print ('cmp:',pred_set,ref_set)
            if pred_set == ref_set:
                matched = True
                break
        if matched:
            correct += 1
    accuracy = correct / len(predictions)
    print(f"Extended Exact Match Accuracy: {accuracy:.2%}")

def evaluate_set_exact_match(guess_sets, label_sets):
    """
    Evaluates exact match between each guess in guess_sets and the corresponding label in label_sets.

    Parameters:
      - guess_sets: A list of sets, each containing guesses for each example.
      - label_sets: A list of sets, each containing the correct labels for each example.

    Returns:
      - Averages of exact match scores across all examples.
    """
    scores = []
    # print ('guesses',guess_sets, label_sets) 
    
    
    # Initialize an empty list to hold the elements
    elements_list = []

    # Process each string and split by ", "
    for s in guess_sets:
        s = s.lower()
        elements = s.split(", ")
        elements_list.append(elements)

    label_sets = [set(label_set['normalized_aliases']) for label_set in label_sets ]
    guess_sets = elements_list
    # print ('guesses',label_sets,guess_sets) 
    for guesses, labels in zip(guess_sets, label_sets):
        print ('guesses',guesses, labels)
        match_scores = [1 if guess in labels else 0 for guess in guesses]
        average_score = sum(match_scores) / len(guesses)
        scores.append(average_score)
        print ('scores',scores)
    

    overall_average_score = sum(scores) / len(scores) if scores else 0
    print(f"Average Exact Match Score: {overall_average_score:.2f}")
    sys.exit()
    return overall_average_score


def evaluate_f1_score(predictions, list_of_reference_lists):
    def f1_score(pred_tokens, ref_tokens):
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    scores = []
    for pred, refs in zip(predictions, list_of_reference_lists):
        pred_tokens = pred.lower().split()
        best_f1 = 0
        for ref in refs['normalized_aliases']:
            ref_tokens = ref.lower().split()
            score = f1_score(pred_tokens, ref_tokens)
            if score > best_f1:
                best_f1 = score
        scores.append(best_f1)
    average_f1 = sum(scores) / len(scores)
    print(f"Average F1 Score: {average_f1:.2f}")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_bleu_score(predictions, list_of_reference_lists):
    scores = []
    smoothing_function = SmoothingFunction().method1

    for pred, refs in zip(predictions, list_of_reference_lists):
        reference_tokens = [ref.lower().split() for ref in refs['normalized_aliases']]
        prediction_tokens = pred.lower().split()
        # print ('reference_tokens',reference_tokens)
        # print ('prediction_tokens',prediction_tokens)
         
        max_score = 0
        for ref in reference_tokens:
            score = sentence_bleu([ref], prediction_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function)
            max_score = max(max_score, score)

        scores.append(max_score)

    overall_average_bleu = sum(scores) / len(scores) if scores else 0
    print(f"Average Maximum BLEU Score: {overall_average_bleu:.2f}") 

from rouge_score import rouge_scorer

def evaluate_rouge_score(predictions, list_of_reference_lists):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, refs in zip(predictions, list_of_reference_lists):
        best_rouge = 0
        for ref in refs['normalized_aliases']:
            score = scorer.score(pred.lower(), ref.lower())
            rouge_l = score['rougeL'].fmeasure
            if rouge_l > best_rouge:
                best_rouge = rouge_l
        scores.append(best_rouge)
    average_rouge = sum(scores) / len(scores)
    print(f"Average ROUGE-L Score: {average_rouge:.2f}")

def calculate_rouge_score_max(prediction, references):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    max_score = 0
    for ref in references:
        scores = scorer.score(prediction, ref)
        max_score = max(max_score, scores['rouge1'].fmeasure)
    return max_score

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

    # for i, data in enumerate(dataset):
    #     print (i,data['answer'])
    #     if len(data['answer']['aliases']) == 0:
    #         sys.exit()
    # sys.exit()
    for i, data in enumerate(dataset):
        # print (data['search_results'])
        # print (data['question'])
        # print (data['entity_pages']['wiki_context'])
        # print (data['answer'])
        input_text = data[question_key]
        reference_answer = data[answer_key]
        context = data['entity_pages']['wiki_context'] 
        context = ' '.join( context[0].split(' ')[:50])
        print ('c2',context )
        # prompt = f"""{args_start_prompt_header}Provide your 20 best guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; not a complete sentence, just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text}, \n\n Guesses:{args_end_prompt_footer}"""

        prompt = f"""{args_start_prompt_header}Provide your 20 best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text}, \n\n The context is: {context} \n\n Guesses:{args_end_prompt_footer}"""

        # prompt = f"""{args_start_prompt_header}Provide your best guess for the following question. Give ONLY the guess. \n\nFor example:\n\nGuesses: <most likely guesses>\n\nThe question is: {input_text}, \n\n The context is: {context} \n\n Guesses:{args_end_prompt_footer}"""

        # prompt = f"""{args_start_prompt_header}Provide your best guess for the following question. Give ONLY the guess. \n\nFor example:\n\nGuesses: <most likely guesses>\n\nThe question is: {input_text}, \n\n Guesses:{args_end_prompt_footer}"""


        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.pad_token_id)
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predictions.append(output_text)
        references.append(reference_answer)

        # Confidence is not directly available without some processing or adaptation from logits
        with torch.no_grad():
            last_hidden_states = model(**inputs, output_hidden_states=True).hidden_states[-1]
            logits = model.lm_head(last_hidden_states)
            confidence_score = torch.softmax(logits, dim=-1).max().item()
            confidences.append(confidence_score)

        print(f"Question {i+1}: {input_text} \n")
        print(f"Expected Answer: {reference_answer} \n")
        print(f"Predicted Answer: {output_text} \n")

    return predictions, references, confidences

import re
def run_inference_guesses(tokenizer, model, device, dataset, question_key, answer_key):
    args_start_prompt_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    args_end_prompt_footer = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    predictions = []
    references = []
    confidences = []

    # for i, data in enumerate(dataset):
    #     print (i,data['answer'])
    #     if len(data['answer']['aliases']) == 0:
    #         sys.exit()
    # sys.exit()
    for i, data in enumerate(dataset):
        # print (data['search_results'])
        # print (data['question'])
        # print (data['entity_pages']['wiki_context'])
        # print (data['answer'])
        input_text = data[question_key]
        reference_answer = data[answer_key]
        context = data['entity_pages']['wiki_context'] 
        context = ' '.join( context[0].split(' ')[:50])
        print ('c2',context )
        # prompt = f"""{args_start_prompt_header}Provide your 20 best guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; not a complete sentence, just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text}, \n\n Guesses:{args_end_prompt_footer}"""

        prompt = f"""{args_start_prompt_header}Provide your 20 best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text}, \n\n The context is: {context} \n\n Guesses:{args_end_prompt_footer}"""

        # prompt = f"""{args_start_prompt_header}Provide your best guess for the following question. Give ONLY the guess. \n\nFor example:\n\nGuesses: <most likely guesses>\n\nThe question is: {input_text}, \n\n The context is: {context} \n\n Guesses:{args_end_prompt_footer}"""

        # prompt = f"""{args_start_prompt_header}Provide your best guess for the following question. Give ONLY the guess. \n\nFor example:\n\nGuesses: <most likely guesses>\n\nThe question is: {input_text}, \n\n Guesses:{args_end_prompt_footer}"""


        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.pad_token_id)
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        target_count = 20
        words = [word.strip() for word in output_text.split(',') if word.strip()]

        # If there are fewer words than the target, append None to make up count
        if len(words) < target_count:
            words.extend(['null'] * (target_count - len(words)))
        # Trim to the target count if there are more
        elif len(words) > target_count:
            words = words[:target_count]
        output_text = ', '.join(words)    
        predictions.append(output_text)
        references.append(reference_answer)

        # Confidence is not directly available without some processing or adaptation from logits
        # with torch.no_grad():
        #     last_hidden_states = model(**inputs, output_hidden_states=True).hidden_states[-1]
        #     logits = model.lm_head(last_hidden_states)
        #     confidence_score = torch.softmax(logits, dim=-1).max().item()
        #     confidences.append(confidence_score)
       
       
        # prompt_confidence = f"""{args_start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your 20 best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text}, \n\n The context is: {context} \n\n the guesses were: {output_text} , given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either (Highest, High, Medium, Low, Lowest) that your guesses are correct, without any extra commentary whatsoever, for example [Highest, High, Medium, Low, Lowest ...]; just the confidence! Separated by a coma> Confidences: {args_end_prompt_footer}"""
        prompt_confidence = f"""{args_start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your 20 best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {input_text}, \n\n The context is: {context} \n\n the guesses were: {output_text} , given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the numerical confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9) that your guesses are correct, without any extra commentary whatsoever, for example [0.1, 0.5, 0.3, 0.7 ...]; just the confidence! Separated by a coma> Confidences: {args_end_prompt_footer}"""
        inputs_conf = tokenizer(prompt_confidence, return_tensors="pt").to(device)
        input_length_conf = len(tokenizer.encode(prompt_confidence, return_tensors="pt")[0])
        with torch.no_grad():
            outputs_conf = model.generate(**inputs_conf, max_length=712, pad_token_id=tokenizer.pad_token_id)
        generated_tokens_conf = outputs_conf[0][input_length_conf:]
        output_text_conf = tokenizer.decode(generated_tokens_conf, skip_special_tokens=True).strip()
        regex = r'\d+\.\d+' 
        matches = re.findall(regex, output_text_conf) 
        float_list = [float(match) for match in matches] 
        target_length = 20
        fill_value = 0.1
        if len(float_list) < target_length:
            float_list.extend([fill_value] * (target_length - len(float_list)))
        # If the list is longer, trim it to the target length
        elif len(float_list) > target_length:
            float_list = float_list[:target_length]
        confidences.append(float_list)

        # references.append(reference_answer)
        #  confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        

        print(f"Question {i+1}: {input_text} \n")
        print(f"Expected Answer: {reference_answer} \n")
        print(f"Predicted Answer: {output_text} \n")
        print(f'Predicted Conf: {output_text_conf} \n')

    return predictions, references, confidences
 
from sklearn.metrics import roc_auc_score 
from sklearn.calibration import calibration_curve
def evaluate_triviaqa_system(guess_lists, label_sets, guess_confidences): 

    # Process each string and split by ", "
    elements_list = []
    for s in guess_lists:
        s = s.lower()
        elements = s.split(", ")
        elements_list.append(elements)
    guess_lists = elements_list

    conf_list = []
    for c in guess_confidences:
        # c = c.lower()
        # confidence = c.split(", ")
        conf_list.append(c)
    guess_confidences = conf_list

    label_sets = [set(label_set['normalized_aliases']) for label_set in label_sets ] 


    true_labels = []

    for guesses, labels in zip(guess_lists, label_sets):
        for guess in guesses:
            print ('tgt:',guess, labels)
            rouge1_score = calculate_rouge_score_max(guess, labels)
            print ('rouge1_score',rouge1_score)
            is_correct = 1 if rouge1_score >= 0.3 else 0
            true_labels.append(is_correct)

    # Flatten the list of confidences and guesses
    flat_confidences = [conf for conf_list_i in guess_confidences for conf in conf_list_i]
    print ('flat_confidences',flat_confidences,len(flat_confidences))
    print('true_labels',true_labels)
    # Compute AUROC
    auroc = roc_auc_score(true_labels, flat_confidences)

    print(f"AUROC: {auroc:.4f}")

    # Calculate ECE
    prob_true, prob_pred = calibration_curve(true_labels, flat_confidences, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"Expected Calibration Error: {ece:.4f}")


def main():
    split = "test"
    # cache_dir =  "/mnt/hdd/brian/hub"
    # os.environ['TFHUB_CACHE_DIR'] = cache_dir
    # os.environ['HF_HOME'] = '/mnt/hdd/brian/hub'
    # os.environ['HF_DATASETS_CACHE'] = '/mnt/hdd/brian/datasets'

    # Load GSM8K dataset
    # gsm8k_dataset = load_gsm8k_dataset(split=split)

    # Load MMLU Professional Law dataset
    # mmlu_prf_law_dataset = load_mmlu_dataset(subject="professional_law", split=split)

    # Load MMLU Business Ethics dataset
    # mmlu_biz_ethics_dataset = load_mmlu_dataset(subject="business_ethics", split=split)

    # # Load StrategyQA dataset
    # strategyqa_dataset = load_strategyqa_dataset(split=split)


    # Load PopQA dataset
    # popqa_dataset = load_popqa_dataset(split=split)

    # Load TriviaQA dataset
    triviaqa_dataset = load_triviaqa_dataset(split=split)


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
    # print("\n=== PopQA Dataset Inference ===")
    # popqa_predictions, popqa_references, confidences = run_inference(tokenizer, model, device, popqa_dataset, question_key="question", answer_key="possible_answers")
    # evaluate_exact_match(popqa_predictions, popqa_references)
    # calculate_ece(popqa_predictions, popqa_references, confidences)
    # calculate_auroc(popqa_predictions, popqa_references, confidences)

     # Run inference on TriviaQA dataset
    print("\n=== TriviaQA Dataset Inference ===")
    triviaqa_predictions, triviaqa_references, confidences = run_inference_guesses(
        tokenizer, model, device, triviaqa_dataset, question_key="question", answer_key="answer"
    )
    evaluate_triviaqa_system(triviaqa_predictions, triviaqa_references, confidences)
    # evaluate_extended_exact_match(triviaqa_predictions, triviaqa_references)
    # evaluate_set_exact_match(triviaqa_predictions, triviaqa_references)
    # evaluate_f1_score(triviaqa_predictions, triviaqa_references)
    # evaluate_bleu_score(triviaqa_predictions, triviaqa_references)
    # evaluate_rouge_score(triviaqa_predictions, triviaqa_references)
    # calculate_ece(triviaqa_predictions, triviaqa_references, confidences)
    # calculate_auroc(triviaqa_predictions, triviaqa_references, confidences)


if __name__ == "__main__":
    main()