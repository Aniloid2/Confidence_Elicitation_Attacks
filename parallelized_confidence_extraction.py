import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import csv
import random
import os
from transformers import BitsAndBytesConfig
from accelerate import init_empty_weights, init_on_device, dispatch_model

# Constants
WORLD_SIZE = 4
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Patterns and options
PATTERNS = {
    "sst2": [
        ("Review:\n{sentence}\nIs this movie review sentence negative or positive?\n{options_}", "{answer}"),
        ("Short movie review: {sentence}\nDid the critic thinking positively or negatively of the movie?\n{options_}", "{answer}"),
        ("Sentence from a movie review: {sentence}\nWas the movie seen positively or negatively based on the preceding review?\n\n{options_}", "{answer}"),
        ("\"{sentence}\"\nHow would the sentiment of this sentence be perceived? \n\n{options_}", "{answer}"),
        ("Is the sentiment of the following sentence positive or negative? \n{sentence}\n{options_}", "{answer}"),
        ("What is the sentiment of the following movie review sentence? \n{sentence}\n{options_}", "{answer}"),
        ("Would the following phrase be considered positive or negative? \n\n{sentence}\n{options_}", "{answer}"),
        ("Does the following review have a positive or negative opinion of the movie?\n\n{sentence}\n{options_}", "{answer}")
    ],
}

PATTERNS_OPTIONS = {
    "sst2": [
        ("positive, negative , netural"),
        ("A) positive B) negative C) netural"),
        ("I) positive II) negative III) netural"),
        ("i) positive ii) negative iii) netural"),
        ("a) positive b) negative c) netural"),
        ("positive, negative , none of the above"),
        ("A) positive B) negative C) none of the above"),
        ("I) positive II) negative III) none of the above"),
        ("i) positive ii) negative iii) none of the above"),
        ("a) positive b) negative c) none of the above"),
        ("positive, negative , all of the above"),
        ("A) positive B) negative C) all of the above"),
        ("I) positive II) negative III) all of the above"),
        ("i) positive ii) negative iii) all of the above"),
        ("a) positive b) negative c) all of the above"),
    ],
}

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def random_label(index, label_type):
    """Generates a label based on the type and index"""
    if label_type == 'uppercase':
        return f"{chr(65 + index)})"  # ASCII for A, B, C,...
    elif label_type == 'lowercase':
        return f"{chr(97 + index)})"  # ASCII for a, b, c,...
    elif label_type == 'roman':
        roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        return f"{roman_numerals[index]})"

def generate_options(options):
    # Define the options and distractors
    random.shuffle(options)
    distractors = ['None of the above', 'All of the above', 'Neutral']

    # Choose label type randomly
    label_types = ['uppercase', 'lowercase', 'roman']
    label_type = random.choice(label_types)

    # Randomly decide if distractors should be included
    include_distractors = random.choice([True, False])

    # Generate the options string
    option_str = ''
    for i, option in enumerate(options):
        label = random_label(i, label_type)
        option_str += f"{label} {option}  "

    # Add distractors if needed
    if include_distractors:
        # Randomly add one or more distractors
        num_distractors = random.randint(1, len(distractors))
        chosen_distractors = random.sample(distractors, num_distractors)
        for i, distractor in enumerate(chosen_distractors, start=len(options)):
            label = random_label(i, label_type)
            option_str += f"{label} {distractor}  "

    return option_str.strip()

def predict_sentiment_and_confidence(rank, model, tokenizer, text):
    def inner_predict_sentiment_and_confidence(text):
        template, answer_template = random.choice(PATTERNS['sst2'])
        options = generate_options(['negative', 'positive'])
        prompt = template.format(sentence=text, options_=options)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(rank)

        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.92,
            "temperature": 0.7
        }

        with torch.no_grad():
            outputs = model.module.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        match_sentiment = re.search(r'positive|POSITIVE|Positive|negative|NEGATIVE|Negative', generated_text)

        if match_sentiment:
            sentiment_result = match_sentiment.group(0).lower()
            return sentiment_result
        else:
            return 'null'
    return inner_predict_sentiment_and_confidence(text)

def analyze_results(rank, model, tokenizer, text, expected_sentiment):
    def inner_analyze_results(text, expected_sentiment):
        results = [predict_sentiment_and_confidence(rank, model, tokenizer, text) for _ in range(20)]
        correct_predictions = sum(1 for sentiment in results if sentiment == expected_sentiment)
        confidence_empirical = (correct_predictions / len(results)) * 100

        sentiment_counts = {'positive': 0, 'negative': 0, 'null': 0}
        for sentiment in results:
            sentiment_counts[sentiment] += 1

        print(f"Results for '{text}':")
        print(f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Null: {sentiment_counts['null']}")
        print(f"Empirical confidence: {confidence_empirical}%")

        return expected_sentiment, confidence_empirical
    return inner_analyze_results(text, expected_sentiment)

def process_and_save_data(rank, world_size, dataset_part):
    setup(rank, world_size)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Initialize an empty model and load it using model parallelism
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    # Dispatch model to the correct devices
    main_device = torch.device(f"cuda:{rank}")
    dispatch_model(model, device_map={name: main_device for name, _ in model.named_parameters()}, main_device=main_device)

    model = DDP(model, device_ids=[rank], output_device=rank)
    model.module.config.use_cache = False
    model.module.config.pretraining_tp = 1

    # Save data to a separate file for each rank
    with open(f'train_data_rank_{rank}.csv', mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Instruction', 'Sentence', 'Prediction', 'Confidence', 'Ground Truth'])

        for example in dataset_part:
            sentence = example['sentence']
            label = example['label']
            expected_prediction = 'negative' if label == 0 else 'positive'

            answer, confidence = analyze_results(rank, model, tokenizer, sentence, expected_sentiment=expected_prediction)
            template, answer_template = random.choice(PATTERNS['sst2'])
            options = generate_options(['negative', 'positive'])
            prompt = template.format(sentence=sentence, options_=options)

            datapoint = (prompt, sentence, answer, confidence, label)
            writer.writerow(datapoint)

    cleanup()

def main():
    world_size = min(WORLD_SIZE, torch.cuda.device_count())
    dataset = load_dataset("glue", "sst2", split='train')

    len_dataset = len(dataset)
    len_per_split = len_dataset // world_size
    lengths = [len_per_split] * world_size
    lengths[-1] += len_dataset % world_size

    dataset_parts = torch.utils.data.random_split(dataset, lengths)

    mp.spawn(process_and_save_data, args=(world_size, dataset_parts), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()