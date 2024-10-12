import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack import Attacker, AttackArgs

import torch.nn.functional as F
import csv 
import torch


from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.search_methods import AlzantotGeneticAlgorithm
from textattack.transformations import WordSwapEmbedding, WordSwapWordNet,WordSwap
from textattack.goal_functions import UntargetedClassification
from textattack.shared import AttackedText
from textattack.attack import Attack

from textattack.search_methods import GreedyWordSwapWIR,  BeamSearch

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics.bert_score import BERTScore
from textattack.constraints.grammaticality import PartOfSpeech
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
 

import os 
# Load model, tokenizer, and model wrapper


from numpy.random import dirichlet


# from src.utils.shared.misc import self.predictor.prompt_class._identify_correct_incorrect_labels


# import random
# import numpy as np
# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
 

# from src.utils.shared.arg_config import get_args

# args = get_args()

# # Function to print Namespace in a human-readable format
# def print_args_human_readable(namespace):
#     print("Arguments Namespace:")
#     for key, value in vars(namespace).items():
#         print(f"  - {key.replace('_', ' ').capitalize()}: {value}")

# # Print the Namespace
# print_args_human_readable(args)
 
# args_dict = vars(args)
  
# cache_dir = args.cache_transformers # "/mnt/hdd/brian/hub"
# os.environ['TFHUB_CACHE_DIR'] = cache_dir

from src.utils.shared.misc import environment_setup

args  = environment_setup() 

# high_level_folder = args.experiment_name_folder
#     # Replace 'desired_log_folder' with your folder name
# test_folder = os.path.join(high_level_folder, f'{args.model_type}_{args.task}_log_folder')

# # Ensure the directory exists; if not, create it

# # Ensure the high-level directory exists; if not, create it
# print ('high_level_folder',high_level_folder,test_folder)
 
# if not os.path.exists(high_level_folder):
#     os.makedirs(high_level_folder)

# # Ensure the specific directory exists; if not, create it
# if not os.path.exists(test_folder):
#     os.makedirs(test_folder)


# high_level_folder = args.experiment_name_folder
# test_folder = os.path.join(high_level_folder, f'{args.model_type}_{args.task}_log_folder') 
# args_dict['test_folder'] = test_folder 
# # Print the folder paths
# print('high_level_folder:', high_level_folder)
# print('test_folder:', test_folder)

# Ensure both the high-level and specific directories are created
os.makedirs(args.test_folder, exist_ok=True)

# Define attack arguments
# if args.prompting_type == 'empirical':
#     num_examples = 100
# else:
#     num_examples = 500
name_of_test = f'EN{str(args.num_examples)}_MT{args.model_type}_TA{args.task}_PT{args.prompting_type}_PST{args.prompt_shot_type}_ST{args.similarity_technique}_NT{args.num_transformations}'

# log_file = open(f'{name_of_test}.txt', 'w')
# sys.stdout = log_file
# print ('args_dict',args_dict)
# model_type = 'llama2'
# task = 'sst2'
# prompting_type = 's1'
# prompt_shot_type = 'fs'
# num_transformations = 20
# index_order_technique = 'prompt_top_k'


# if args.model_type =='llama2':
#     model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
#     args.start_prompt_header = "<s>[INST]"
#     args.end_prompt_footer = "[/INST]"
# elif args.model_type == 'mistral':
#     model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#     args.start_prompt_header = "<s>[INST]"
#     args.end_prompt_footer = "[/INST]"
# elif args.model_type == 'llama3':
#     model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#     args.start_prompt_header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
#     args.end_prompt_footer = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP,TASK_N_CLASSES,MODEL_INFO # TASK_LABEL_TO_NAME, TASK_NAME_TO_LABEL


args.n_classes =  TASK_N_CLASSES[args.task] 
args.confidence_type_dict = CONFIDENCE_LEVELS[args.confidence_type] 
args.confidence_map_dict = CONFIDENCE_MAP[args.confidence_type] 
# args.task_label_to_name_dict = TASK_LABEL_TO_NAME[args.task]
# args.task_name_to_lebel_dict = TASK_NAME_TO_LABEL[args.task]
model_info = MODEL_INFO[args.model_type]
 
args.model_name =  model_info['model_name']
args.start_prompt_header = model_info['start_prompt_header']
args.end_prompt_footer = model_info['end_prompt_footer']


# elif args.model_type == 'llama2_13b':
#     model_name = "meta-llama/Llama-2-13b-chat-hf"  
# model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
# model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
# model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "TheBloke/Llama-2-70B-AWQ"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained("./finetuned_sentiment_model")
# model = AutoModelForCausalLM.from_pretrained("./finetuned_sentiment_model")



#Constraints
# Define constraints (optional but recommended to refine the search space)
# stopwords = set(
#             ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
#         )
# extended_stopwords = set()
# for word in stopwords:
#     extended_stopwords.add(word)             # Original lower-cased version
#     extended_stopwords.add(word.upper())     # Fully upper-cased version
#     extended_stopwords.add(word.capitalize())
from src.utils.shared.misc import set_stopwords
stopwords = set_stopwords() 
constraints = [ RepeatModification(),StopwordModification(stopwords=stopwords)]
import math
if args.similarity_technique == 'USE':
    angular_use_threshold = args.similarity_threshold
    # use_threshold = 0.5
    use_threshold = 1 - (angular_use_threshold) / math.pi
    if args.transformation_method == 'word_swap_embedding':
        compare_against_original = True
        window_size = None 
        skip_text_shorter_than_window=False
    elif args.transformation_method == 'sspattack' :
        compare_against_original = True
        window_size = None 
        skip_text_shorter_than_window=False
    else:
        compare_against_original = True
        window_size = None 
        skip_text_shorter_than_window=False


    use_constraint = UniversalSentenceEncoder(
                threshold=use_threshold,
                metric="angular",
                compare_against_original=compare_against_original,
                window_size=window_size,
                skip_text_shorter_than_window=skip_text_shorter_than_window,
            )
elif args.similarity_technique == 'BERTScore':
    # bert_score = 0.85
    bert_score = args.similarity_threshold
    use_constraint = BERTScore(min_bert_score =bert_score)


if args.transformation_method == 'sspattack': 
    pass
elif args.transformation_method == 'texthoaxer': 
    pass
else:
    constraints.append(use_constraint) 

args.use_constraint = use_constraint


constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))

input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
)
constraints.append(input_column_modification)

if args.task == 'strategyQA':
    constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
    if args.transformation_method == 'self_word_sub':
        pass
    else:
        from src.custom_constraints.swap_constraints import NoNounConstraint
        constraints.append(NoNounConstraint())
else:
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))



from src.utils.shared.misc import initialize_model_and_tokenizer

args = initialize_model_and_tokenizer(args)
# tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_transformers,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_transformers,trust_remote_code=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer's pad token is set
# model.config.pad_token_id = tokenizer.pad_token_id
# print ('tokenizer.pad_token',tokenizer.pad_token )
# print ( 'tokenizer.pad_token_id', tokenizer.pad_token_id)
# # args.update({'model': model,'device':device, 'tokenizer':tokenizer})
# args.model = model
# args.device = device
# args.tokenizer = tokenizer 


from src.utils.shared import load_data
dataset_class, label_names = load_data(args)
from src.utils.shared import SimpleDataset
dataset_class =  SimpleDataset(dataset_class,label_names = label_names ) 
args.dataset = dataset_class


from src.inference import Step2KPredAvg

# if args.prompting_type == 'step2_k_pred_avg':
from src.inference.inference_config import DYNAMIC_INFERENCE
args.predictor = DYNAMIC_INFERENCE[args.prompting_type](**vars(args))


# else:
#     print ('invalid')
#     sys.exit()



from textattack.models.wrappers import ModelWrapper
import re

 
# from datasets import load_dataset
# # Load and process dataset
# if args.task == 'sst2':
#     if args.prompting_type == 'empirical':
#         num_samples = 50
#     else:
#         num_samples = 250
#     dataset = HuggingFaceDataset("glue", "sst2", split="validation", shuffle=True) 

#     # For dataset_class_1, only include sentences with 3 or more characters and label == 1
#     dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1 and len(text['sentence']) >= 3]
#     # dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1]
#     # used to take 3000:3250
#     dataset_class_1_t = dataset_class_1[:num_samples]#[3000:3250]
#     incontext_dataset_class_1 = dataset_class_1[-5:]
#     # For dataset_class_0, only include sentences with 3 or more characters and label == 0
#     dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0 and len(text['sentence']) >= 3]
#     # dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0]
#     dataset_class_0_t = dataset_class_0[:num_samples]#[3000:3250]
#     incontext_dataset_class_0 = dataset_class_0[-5:]
#     label_names = ['negative','positive']
#     dataset_class =   dataset_class_1_t + dataset_class_0_t
# elif args.task == 'ag_news':
#     # dataset = load_dataset('ag_news', split='train').shuffle(seed=42)
#     if args.prompting_type == 'empirical':
#         num_samples = 25
#     else:
#         num_samples = 125
#     dataset = HuggingFaceDataset('ag_news', split="test", shuffle=True) 
#     # For dataset_class_1, only include documents with 3 or more characters and label == 1
#     dataset_class_1 = [(text['text'], label) for (text, label) in dataset if label == 1 and len(text['text']) >= 3]
#     dataset_class_1_t = dataset_class_1[:num_samples]
#     incontext_dataset_class_1 = dataset_class_1[-5:]

#     # For dataset_class_0, only include documents with 3 or more characters and label == 0
#     dataset_class_0 = [(text['text'], label) for (text, label) in dataset if label == 0 and len(text['text']) >= 3]
#     dataset_class_0_t = dataset_class_0[:num_samples]
#     incontext_dataset_class_0 = dataset_class_0[-5:]

#     # You can extend for classes 2 and 3 similarly if needed
#     dataset_class_2 = [(text['text'], label) for (text, label) in dataset if label == 2 and len(text['text']) >= 3]
#     dataset_class_2_t = dataset_class_2[:num_samples]
#     incontext_dataset_class_2 = dataset_class_2[-5:]

#     dataset_class_3 = [(text['text'], label) for (text, label) in dataset if label == 3 and len(text['text']) >= 3]
#     dataset_class_3_t = dataset_class_3[:num_samples]
#     incontext_dataset_class_3 = dataset_class_3[-5:]

#     label_names = ['world','sport','business','tech/sci']

#     # Combine datasets from different classes
#     dataset_class =   dataset_class_0_t + dataset_class_1_t + dataset_class_2_t + dataset_class_3_t

#     print(f'Total filtered dataset size: {len(dataset_class)}')
#     print(f'In-context samples for class 1: {incontext_dataset_class_1}')
#     print(f'In-context samples for class 0: {incontext_dataset_class_0}')
#     print(f'In-context samples for class 2: {incontext_dataset_class_2}')
#     print(f'In-context samples for class 3: {incontext_dataset_class_3}')
# else:
#     print("Task not supported.")


# from textattack.datasets import Dataset

# class SimpleDataset(Dataset):
#     def __init__(self, examples, label_names=None):
#         """
#         args:
#             examples: list of tuples where each tuple is (text, label)
#             label_names: list of strings representing label names (optional)
#         """
#         self.examples = examples
#         self.label_names = label_names
#         self.shuffled = False  # Set to True if you shuffle the examples

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         return self.examples[idx]

# Convert filtered datasets into TextAttack Dataset format 









import random
class HuggingFaceLLMWrapper(ModelWrapper):
    """A wrapper around HuggingFace for LLMs.

    Args:
        model: A HuggingFace pretrained LLM
        tokenizer: A HuggingFace pretrained tokenizer
    """

    def __init__(self, **kwargs):# model, tokenizer):

        # self.model = args.model
        # self.tokenizer = args.tokenizer
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.generated_max_length = 2
        self.inference_step = 0
        self.current_sample = 1



    def reset_inference_steps(self):
        """Resets the inference step counter to 0 and current sample to +1"""
        self.inference_step = 0
        self.current_sample +=1


    def scale_vector_by_confidence(self,vector, confidence, margin=0.01):
        # Normalize the confidence value to a range of 0 to 1
        # vector = np.array([0,0,0,1])#np.array([0,1])
        # confidence = 0
        print ('vector',vector)
        normalized_confidence = confidence #/ 100.0

        # Create an almost equalized vector with a slight margin
        num_classes = len(self.dataset.label_names)
        print('num_classes',num_classes)
        half_margin = margin 
        equalized_vector = [0.5 + half_margin if i == np.argmax(vector) else 0.5 - half_margin for i in range(len(vector))]
        print ('equalized_vector',equalized_vector)
        # Interpolate between the equalized vector and the original vector
        scaled_vector = [
            (1 - normalized_confidence) * equalized_value + normalized_confidence * original_value
            for equalized_value, original_value in zip(equalized_vector, vector)
        ]
        print ('scaled_vector',scaled_vector)
        # Ensure the scaled vector sums to 1 by re-normalizing
        sum_scaled_vector = sum(scaled_vector)
        normalized_scaled_vector = [value / sum_scaled_vector for value in scaled_vector]
        print ('normalized_scaled_vector',normalized_scaled_vector) 
        return normalized_scaled_vector

    def predict_sentiment_and_confidence(self,text,expected_sentiment):
        # for llama2 7 b
        num_classes = 2
        # prompt = f"[INST]What is the sentiment of the following movie review sentence, and provide your prediction confidence as a percentage? \n{t}\npositive, negative[/INST]"
        # for quantiside llama 70b
        prompt = f"""{self.start_prompt_header}Provide your best guess and the probability that it is correct (0% to 100%) for
                the following sentiment. Give ONLY the guess and probability, no other words or
                explanation. For example:\n\nGuess: <most likely guess, either positive or negative; not
                a complete sentence, just the guess!>\n Probability: <the probability between 0%
                and 100% that your guess is correct, without any extra commentary whatsoever; just
                the probability!>\n\nThe question is: ${text} Sentiment & confidence:{self.end_prompt_footer}"""
         

        #. “lowest”, “low”, “medium”,“high”, “highest”
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        
        # Get model predictions as text
        generate_args = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "do_sample": True,  # enable sampling
                "top_k": 40,  # top-k sampling
                "top_p": 0.92,  # nucleus sampling probability
                "temperature": 0.7,  # sampling temperature
                "max_new_tokens":200
            }

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200,temperature = 0.7 ,top_k= 1)
        
            # full_text  = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the generated text from the model
        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
        print("Generated Text:", generated_text)

        # Use regex to extract sentiment and confidence
        match_sentiment = re.search(r'positive|POSITIVE|Positive|negative|NEGATIVE|Negative', generated_text)
        match_confidence = re.search(r"[-+]?\d*\.?\d+", generated_text)
        print ('match_sentiment',match_sentiment,'match_confidence',match_confidence)
        
        


        if match_sentiment and match_confidence:
            sentiment_result = match_sentiment.group(0).lower()
            confidence_result = float(match_confidence.group(0)) /100   # Append '%'
            
            confidence_result = 100 if confidence_result > 100 else confidence_result
            if sentiment_result == 'positive':
                # probs = np.array([1-confidence_result,confidence_result])
                sentiment_confidences = { 'negative': 1-confidence_result, 'positive': confidence_result, 'null': 0.0 }
            else:
                sentiment_confidences = { 'negative': confidence_result, 'positive': 1- confidence_result, 'null': 0.0 }
                # probs = np.array([confidence_result,1-confidence_result])

            # concatenated_logits = torch.tensor([probs[1], probs[0],0.0], device=device)
            # print ('softmax_probabilities',softmax_probabilities) 
            # logit_list.append(concatenated_logits)
             
            expected_sentiment = expected_sentiment
            confidence_empirical = confidence_result
            # return sentiment_result, probs ,confidence_result
            return expected_sentiment,confidence_empirical,sentiment_confidences
        else: 
            # temp_logit = torch.zeros(num_classes+1, dtype=torch.float)  
            # temp_logit[-1] = 1.0  
            # logit_list.append(torch.tensor(temp_logit, device=device))
            print ('expected_sentiment',expected_sentiment)
            if expected_sentiment == 1:

                sentiment_confidences = { 'negative': 0, 'positive': 1.0, 'null': 0.0 }
            else:
                sentiment_confidences = { 'negative': 1.0, 'positive': 0.0, 'null': 0.0 }
            expected_sentiment = expected_sentiment
            confidence_empirical = 100.0
            return  expected_sentiment,confidence_empirical,sentiment_confidences
            # probs = np.array([0.5,0.5])
            
            # return 'null', np.array([0,0]) , 0.0
        
    
    def predict_sentiment_self_fool_wordattack_baseline_w1(self,text,expected_sentiment):
        
        
        if self.inference_step == 0:
            print ('its the first step')
        else:
            print ('other steps we dont care')
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
            class_number  = 2
            label_list = ['negative', 'positive']
        elif self.task == 'ag_news':
            guess_pattern = r'(world|business|tech|science|sports)'
            class_number  = 4
            label_list = ['world','business', 'tech','science','tech/sci', 'sports']

    
    

        if self.task == 'sst2':
            # prompt = f"""[INST]Review:\n{text}\nIs this movie review sentence negative or positive? Answer:[/INST]"""
            # prompt = f"""Analyze the tone of this statement and respond with either 'positive' or 'negative': Sentence: {text}Answer:"""
            options = ', '.join(self.dataset.label_names)
            options2 = ' or '.join(self.dataset.label_names)
            prompt = f"""{self.start_prompt_header}Provide your best guess for the following text ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options2}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
            # label_types = ['negative','positive']
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0
            # text_type = 'text'
            # print ('expected_sentiment',expected_sentiment, expected_sentiment_label)
            
            # filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            # print ('filtered_label_types',filtered_label_types)
            # # Join the filtered labels with ' or '
            # other_classes = ' or '.join(filtered_label_types)
            # attack_type = 'Replace at most two words in the sentence with synonyms.'
            # print ('other_classes',other_classes)
            # original_input = f"""The original {text_type} {text} is classified as {expected_sentiment} """
            # attack_objective = f""" Your task is to generate a new {text_type} a which must satisfy the following conditions:
            #     1) Keeping the semantic meaning of the new {text_type} unchanged
            #     2) The new {text_type} and the original {text_type} should be classified as {other_classes}. """
            # attack_guidance = f"""You can finish the task by modifying {text_type} using the following guidance:
            #     {attack_type}
            #     Only output the new {text_type} without anything else. """
            # prompt = '[INST]' + original_input + attack_objective + attack_guidance + '[/INST]'
            
        elif self.task == 'ag_news':
            # prompt = f"""Analyze the tone of this statement and respond with either 'positive' or 'negative': Sentence: {text}Answer:"""
            # from flan
            options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            # prompt = f"""text:{text}\n\nWhat is this text about out of the possible options ({options})? Answer: """
            # prompt = f"""[INST]Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:[/INST]"""
            prompt = f"""{self.start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
            
            # prompt = f"""[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') for the following news article. Give ONLY the guess and verbal confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, either world, sports, business or tech/science; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!>\n\nThe text is: ${text} News type & confidence:[/INST]"""
        print ('prompt - area 1' , prompt) 
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

        # Get model predictions as text
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("Generated Text:", generated_text)
        
        # Use regex to extract the guess (sentiment or category) and verbal confidence
        match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
        # match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
        print('match_guess', match_guess )

        # confidence_map = {
        #     'lowest': 0,
        #     'low': 25,
        #     'medium': 50,
        #     'high': 75,
        #     'highest': 100
        # }

        if match_guess:
            guess_result = match_guess.group(0).lower()
            # match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            # confidence_result = float(match_confidence) / 100  # Normalize confidence
            confidence_result = 1.0


            if guess_result not in label_list:
                guess_result = 'null'

            # Handle probabilities for different categories
            probs = np.zeros(class_number)
            if guess_result != 'null':
                if self.task == 'sst2':
                    if guess_result == 'positive':

                        probs = np.array([0.0,1.0,0.0]) 
                    else:
                        probs = np.array([1.0,0.0,0.0])
                    return guess_result, probs ,confidence_result
                elif self.task == 'ag_news':
                    granularity = 4 # how many confidence labels do we have in confidence elicitation
                    ignore_classes  = 1 # ignore null and the current class
                    other_level_candidates = class_number
                    
                    confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                    confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                    if guess_result == 'world':
                        probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                    elif guess_result == 'sports' :
                        probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                    elif guess_result == 'business':
                        probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                    elif guess_result == 'tech' or 'science' or 'tech/science':
                        guess_result = 'tech/sci'
                        probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                    return guess_result, probs ,confidence_result
                

            return guess_result, probs, confidence_result
        else:
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null', probs, 1.0
    
    def predict_sentiment_and_verbal_confidence_2steps(self,text, expected_sentiment):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
            class_number  = 2
            label_list = ['positive', 'negative']
        elif self.task == 'ag_news':
            guess_pattern = r'(world|business|tech|science|sports)'
            class_number  = 4
            label_list = ['world','business', 'tech','science','tech/sci', 'sports']

    
    

        if self.task == 'sst2':
            prompt = f"""{self.start_prompt_header}Provide your best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
        
        elif self.task == 'ag_news':
            
            options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            
            prompt = f"""{self.start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

        # Get model predictions as text
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("Generated Prediction Text:", generated_text)

        # Use regex to extract the guess (sentiment or category) and verbal confidence
        match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
        if match_guess:
            guess_result = match_guess.group(0).lower()
            if guess_result not in label_list:
                guess_result = 'null'
        else:
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null', probs, 1.0

        # confidence_prompt = f"""[INST]Provide the verbal confidence that your guess is correct ('lowest', 'low', 'medium','high', 'highest') Give ONLY the verbal confidence, no other words or explanation. For example: Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!> The text is: "{text}" with guess: "{guess_result}" Confidence:[/INST]"""
        if self.task == 'sst2':
            confidence_prompt = f"""{self.start_prompt_header}Provide your best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess: {guess_result} Provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!> Confidence:{self.end_prompt_footer}"""
        
        elif self.task == 'ag_news':
            confidence_prompt = f"""{self.start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess: {guess_result} Provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!> Confidence:{self.end_prompt_footer}"""

        inputs = tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.92,
            "temperature": 0.7,
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }
        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("Generated Confidence Text:", generated_text) 

        match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
        print('match_guess', match_guess, 'match_confidence', match_verbal_confidence)

        confidence_map = {
            'lowest': 0,
            'low': 25,
            'medium': 50,
            'high': 75,
            'highest': 100
        }

        if not match_verbal_confidence:
            guess_result = match_guess.group(0).lower()
            # match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            # confidence_result = float(match_confidence) / 100  # Normalize confidence
            confidence_result = 1.0


            if guess_result not in label_list:
                guess_result = 'null'

            # Handle probabilities for different categories
            probs = np.zeros(class_number)
            if guess_result != 'null':
                if self.task == 'sst2':
                    if guess_result == 'positive':

                        probs = np.array([0.0,1.0,0.0]) 
                    else:
                        probs = np.array([1.0,0.0,0.0])
                    return guess_result, probs ,confidence_result
                elif self.task == 'ag_news':
                    granularity = 4 # how many confidence labels do we have in confidence elicitation
                    ignore_classes  = 1 # ignore null and the current class
                    other_level_candidates = class_number
                    
                    confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                    confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                    if guess_result == 'world':
                        probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                    elif guess_result == 'sports' :
                        probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                    elif guess_result == 'business':
                        probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                    elif guess_result == 'tech' or 'science' or 'tech/science':
                        guess_result = 'tech/sci'
                        probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                    return guess_result, probs ,confidence_result

        guess_result = match_guess.group(0).lower()
        match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
        confidence_result = float(match_confidence) / 100  # Normalize confidence


        if guess_result not in label_list:
            guess_result = 'null'

        # Handle probabilities for different categories
        probs = np.zeros(class_number)
        if guess_result != 'null':
            if self.task == 'sst2':
                if guess_result == 'positive':
                    probs = np.array(self.scale_vector_by_confidence([0,1],confidence_result) + [0.0])
                    # probs = np.array([1-confidence_result,confidence_result,0.0])
                else:
                    probs = np.array(self.scale_vector_by_confidence([1,0],confidence_result) + [0.0])
                    # probs = np.array([confidence_result,1-confidence_result,0.0])
                print ('probs 2step',probs)
                return guess_result, probs ,confidence_result
            elif self.task == 'ag_news':
                granularity = 4 # how many confidence labels do we have in confidence elicitation
                ignore_classes  = 1 # ignore null and the current class
                other_level_candidates = class_number
                
                confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                if guess_result == 'world':
                    conf_vec = self.scale_vector_by_confidence([1.0,0.0,0.0,0.0],confidence_result)
                    probs = np.array(conf_vec+[0.0])
                    # probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                elif guess_result == 'sports' :
                    conf_vec = self.scale_vector_by_confidence([0.0,1.0,0.0,0.0],confidence_result)
                    probs = np.array(conf_vec+[0.0])
                    # probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                elif guess_result == 'business':
                    conf_vec = self.scale_vector_by_confidence([0.0,0.0,1.0,0.0],confidence_result)
                    probs = np.array(conf_vec+[0.0])
                    # probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                elif guess_result == 'tech' or 'science' or 'tech/science':
                    guess_result = 'tech/sci'
                    conf_vec = self.scale_vector_by_confidence([0.0,0.0,0.0,1.0],confidence_result)
                    probs = np.array(conf_vec+[0.0])
                    # probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                return guess_result, probs ,confidence_result
            

        return guess_result, probs, confidence_result




        match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
        print('match_guess', match_guess, 'match_confidence', match_verbal_confidence)

        confidence_map = {
            'lowest': 0,
            'low': 25,
            'medium': 50,
            'high': 75,
            'highest': 100
        }

        if match_guess and match_verbal_confidence:
            guess_result = match_guess.group(0).lower()
            match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            confidence_result = float(match_confidence) / 100  # Normalize confidence


            if guess_result not in label_list:
                guess_result = 'null'

            # Handle probabilities for different categories
            probs = np.zeros(class_number)
            if guess_result != 'null':
                if self.task == 'sst2':
                    if guess_result == 'positive':

                        probs = np.array([1-confidence_result,confidence_result,0.0])
                    else:
                        probs = np.array([confidence_result,1-confidence_result,0.0])
                    return guess_result, probs ,confidence_result
                elif self.task == 'ag_news':
                    granularity = 4 # how many confidence labels do we have in confidence elicitation
                    ignore_classes  = 1 # ignore null and the current class
                    other_level_candidates = class_number
                    
                    confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                    confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                    if guess_result == 'world':
                        probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                    elif guess_result == 'sports' :
                        probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                    elif guess_result == 'business':
                        probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                    elif guess_result == 'tech' or 'science' or 'tech/science':
                        guess_result = 'tech/sci'
                        probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                    return guess_result, probs ,confidence_result
                

            return guess_result, probs, confidence_result
        elif match_guess:
            guess_result = match_guess.group(0).lower()
            # match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            # confidence_result = float(match_confidence) / 100  # Normalize confidence
            confidence_result = 1.0


            if guess_result not in label_list:
                guess_result = 'null'

            # Handle probabilities for different categories
            probs = np.zeros(class_number)
            if guess_result != 'null':
                if self.task == 'sst2':
                    if guess_result == 'positive':

                        probs = np.array([0.0,1.0,0.0]) 
                    else:
                        probs = np.array([1.0,0.0,0.0])
                    return guess_result, probs ,confidence_result
                elif self.task == 'ag_news':
                    granularity = 4 # how many confidence labels do we have in confidence elicitation
                    ignore_classes  = 1 # ignore null and the current class
                    other_level_candidates = class_number
                    
                    confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                    confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                    if guess_result == 'world':
                        probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                    elif guess_result == 'sports' :
                        probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                    elif guess_result == 'business':
                        probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                    elif guess_result == 'tech' or 'science' or 'tech/science':
                        guess_result = 'tech/sci'
                        probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                    return guess_result, probs ,confidence_result

        else:
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null', probs, 1.0




    def predict_sentiment_and_verbal_confidence_2step_k_pred_avg(self,text, expected_prediction):
        k_pred = 20
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
            class_number  = 2
            label_list = self.dataset.label_names# ['positive', 'negative']
        elif self.task == 'ag_news':
            guess_pattern = r'(world|business|tech|science|sports)'
            class_number  = 4
            label_list = self.dataset.label_names# ['positive', 'negative'] #label_list = ['world','business', 'tech','science','tech/sci', 'sports']

        task_dictionary_counts = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
        task_dictionary_confidences = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
            



        if self.task == 'sst2':
            prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a coma, for example [Negative, Positive, Positive, Negative ...]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
        
        elif self.task == 'ag_news':
            
            options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            
            prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses! Separated by a coma, for example [Sport, Business, Sport, Politics, Sci/Tech ...]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

        # Get model predictions as text
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print ('prompt', prompt)
        print("Generated Prediction Text:", generated_text) 
        

        # Extract guesses, assuming they're separated by commas and ignoring case
        results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
        results = [result for result in results if result in label_list]
        # If fewer results than k_pred, fill with 'null'
        results.extend(['null'] * (k_pred - len(results)))
        print ('results',results,expected_prediction)
        correct_predictions = sum(1 for sentiment in results if sentiment == expected_prediction)
        confidence_empirical = (correct_predictions / len(results)) * 100
        

        for sentiment in results:
            task_dictionary_counts[self.task][sentiment] += 1
        
        # sentiment_confidences = { 'positive': 0, 'negative': 0, 'null': 0 }
        for sentiment, number_of_results in task_dictionary_counts[self.task].items():
            task_dictionary_confidences[self.task][sentiment] = (number_of_results / len(results))

        # average_confidence = sum(confidence for _, confidence in results) / len(results)

        print(f"Results for '{text}':")
        print(f"Counter: {task_dictionary_counts[self.task]}")
        # print(f"Average model confidence: {average_confidence}%")
        print(f"Empirical confidence: {confidence_empirical}%")
        max_class = max(task_dictionary_counts[self.task], key=task_dictionary_counts[self.task].get)
        print ('max_class',max_class,expected_prediction)
        print ('empricial:',expected_prediction,confidence_empirical,task_dictionary_confidences[self.task])


        # if null majority, we return null as main
        
        # if max_class != 'null':
        #     guess_result = max_class
        #     if guess_result not in label_list:
        #         guess_result = 'null'
        # else:
        #     probs = np.zeros(class_number+1)
        #     probs[-1] = 1.0  # Only null has confidence 1
        #     return 'null', probs, 1.0

        guess_result = max_class
        guesses_output = results
        
        if self.task == 'sst2':
            # confidence_prompt = f"""{start_prompt_header}Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[self.confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [Low, Highest, Medium, Lowest, High, High ...]> Confidences:{end_prompt_footer}"""
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[self.confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [Low, Highest, Medium, Lowest, High, High ...]> Confidences:{self.end_prompt_footer}"""
        
        elif self.task == 'ag_news':
            confidence_prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guesses for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either {options}; not a complete sentence, just the guesses!>\n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[self.confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidencees! Separated by a coma, for example [Low, Highest, Medium, Lowest, High, High ...]> Confidences:{self.end_prompt_footer}"""


        print ('confidence_prompt',confidence_prompt)
        inputs = tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.92,
            "temperature": 0.7,
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }
        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("Generated Confidence Text:", generated_text) 
        # sorted_confidence_options = sorted(CONFIDENCE_LEVELS[self.confidence_type], key=len, reverse=True) 
        # print ('sorted_confidence_options',sorted_confidence_options)


        confidence_options = '|'.join(CONFIDENCE_LEVELS[self.confidence_type]) 
        confidence_guesses = re.findall(confidence_options, generated_text, flags=re.IGNORECASE)
        confidence_guesses =  [match.lower() for match in confidence_guesses]
        # confidence_guesses = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
        print ('confidence_guesses',confidence_guesses)
        confidence_list = CONFIDENCE_LEVELS[self.confidence_type]
        print ('confidence_list',confidence_list)
        confidence_results = [result for result in confidence_guesses if result in confidence_list]
        # confidence_results=[]
        print ('confidence_results',confidence_results)
        # If fewer results than k_pred, fill with 'null'
        confidence_results.extend(['null'] * (k_pred - len(confidence_results)))
        
        
        confidence_map = CONFIDENCE_MAP[self.confidence_type]

        # can count number of occurances or other stuff when computing the confidence 
        # confidence_numerical_results = [confidence_map[result] for result in confidence_results  min(list(confidence_map.values())) if result == 'null' ]
        
        confidence_numerical_results = [
            confidence_map[result] if result != 'null' else min(confidence_map.values()) 
            for result in confidence_results
        ] 
        # confidence_numerical_results.extend(([min(list(confidence_map.values()))]) * (k_pred - len(confidence_results)))
         
        # confidence_numerical_results.extend([min(list(confidence_map.values()))] * (k_pred - len(confidence_results)))
        
        print ('confidence_numerical_results',confidence_numerical_results) 
        print ('guesses_output',guesses_output)

        counts = task_dictionary_counts[self.task]   
        weighted_counts = { 'negative': 0.0,'positive': 0.0,'null':0.0} 
        for sentiment, confidence in zip(guesses_output, confidence_numerical_results):
            print ('confidence',confidence)
            if confidence:
                weighted_counts[sentiment] += confidence
            else:
                weighted_counts[sentiment] += 1

        print ('weighted_counts',weighted_counts) 
        alpha_prior = 1.0

        alpha = { 'negative': weighted_counts['negative'] + alpha_prior,'positive': weighted_counts['positive'] + alpha_prior,'null': weighted_counts['null'] + alpha_prior}

        # alpha = { 'negative': counts['negative'] + alpha_prior,'positive': counts['positive'] + alpha_prior,'null': counts['null'] + alpha_prior}

        alpha_total = sum(alpha.values())
        from utils.shared.plotting import ternary_plot, ternary_mean_plot
        sample_size=1000
        alpha_values = list(alpha.values())
        dirichlet_distribution = dirichlet(alpha_values, size=sample_size)
        # print ('dirichlet_distribution',dirichlet_distribution)
        # Normalized probabilities from the Dirichlet distribution
        samples_ternary = [(p[0], p[1], p[2]) for p in dirichlet_distribution]  
        # print ('dirichlet_distribution',samples_ternary)
        empirical_means = np.mean(dirichlet_distribution, axis=0)
        empirical_means_ternary = (empirical_means[0], empirical_means[1], empirical_means[2])
        print ('empirical_means',empirical_means) 
        if self.current_sample == 1:
            self.inference_step +=1
            ternary_plot_file = os.path.join(self.test_folder, f'dirichlet_cs{self.current_sample}_is{self.inference_step}_a({alpha_values})_n{str(sample_size)}')
            ternary_mean_plot(samples_ternary,alpha_values,empirical_means_ternary,ternary_plot_file)

        probabilities = dirichlet_distribution[0]

        def dirichlet_variance(alpha):
            alpha_0 = sum(alpha)
            variances = [(alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1)) for alpha_i in alpha]
            return variances

        alpha_vector = [ alpha['negative'],alpha['positive'],alpha['null']]
        second_order_uncertainty = dirichlet_variance(alpha_vector)

        print("Counts:", counts)
        print("Numerical Confidences:", confidence_numerical_results)
        print("Weighted Counts:", weighted_counts)
        print("Alpha Vector:", alpha_vector)
        print("Probabilities:", probabilities)
        print("Second Order Uncertainty:", second_order_uncertainty)
        probs = probabilities
        confidence_result = max(probabilities)
        print ('guess_result',guess_result, 'probs',probs,'confidence_result',confidence_result   )
        
        return guess_result, probs ,confidence_result

        
    def predict_sentiment_and_verbal_confidence_s1(self,text, expected_sentiment):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
            class_number  = 2
            label_list = ['positive', 'negative']
        elif self.task == 'ag_news':
            guess_pattern = r'(world|business|tech|science|sports)'
            class_number  = 4
            label_list = ['world','business', 'tech','science','tech/sci', 'sports']

    
    

        if self.task == 'sst2':
            prompt = f"""{self.start_prompt_header}Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') for the following sentiment. Give ONLY the guess and verbal confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!>\n\nThe text is: ${text} Sentiment & confidence:{self.end_prompt_footer}"""
        elif self.task == 'ag_news':
            prompt = f"""{self.start_prompt_header}Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') for the following news article. Give ONLY the guess and verbal confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, either world, sports, business or tech/science; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!>\n\nThe text is: ${text} News type & confidence:{self.end_prompt_footer}"""
            # options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            
            # prompt = f"""[INST]Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:[/INST]"""
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

        # Get model predictions as text
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("Generated Text:", generated_text)

        # Use regex to extract the guess (sentiment or category) and verbal confidence
        match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
        match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
        print('match_guess', match_guess, 'match_confidence', match_verbal_confidence)

        confidence_map = {
            'lowest': 0,
            'low': 25,
            'medium': 50,
            'high': 75,
            'highest': 100
        }

        if match_guess and match_verbal_confidence:
            guess_result = match_guess.group(0).lower()
            match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            confidence_result = float(match_confidence) / 100  # Normalize confidence


            if guess_result not in label_list:
                guess_result = 'null'

            # Handle probabilities for different categories
            probs = np.zeros(class_number)
            if guess_result != 'null':
                if self.task == 'sst2':
                    if guess_result == 'positive':

                        probs = np.array([1-confidence_result,confidence_result,0.0])
                    else:
                        probs = np.array([confidence_result,1-confidence_result,0.0])
                    return guess_result, probs ,confidence_result
                elif self.task == 'ag_news':
                    granularity = 4 # how many confidence labels do we have in confidence elicitation
                    ignore_classes  = 1 # ignore null and the current class
                    other_level_candidates = class_number
                    
                    confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                    confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                    if guess_result == 'world':
                        probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                    elif guess_result == 'sports' :
                        probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                    elif guess_result == 'business':
                        probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                    elif guess_result == 'tech' or 'science' or 'tech/science':
                        guess_result = 'tech/sci'
                        probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                    return guess_result, probs ,confidence_result
                

            return guess_result, probs, confidence_result
        elif match_guess:
            guess_result = match_guess.group(0).lower()
            # match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            # confidence_result = float(match_confidence) / 100  # Normalize confidence
            confidence_result = 1.0


            if guess_result not in label_list:
                guess_result = 'null'

            # Handle probabilities for different categories
            probs = np.zeros(class_number)
            if guess_result != 'null':
                if self.task == 'sst2':
                    if guess_result == 'positive':

                        probs = np.array([0.0,1.0,0.0]) 
                    else:
                        probs = np.array([1.0,0.0,0.0])
                    return guess_result, probs ,confidence_result
                elif self.task == 'ag_news':
                    granularity = 4 # how many confidence labels do we have in confidence elicitation
                    ignore_classes  = 1 # ignore null and the current class
                    other_level_candidates = class_number
                    
                    confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                    confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                    if guess_result == 'world':
                        probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                    elif guess_result == 'sports' :
                        probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                    elif guess_result == 'business':
                        probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                    elif guess_result == 'tech' or 'science' or 'tech/science':
                        guess_result = 'tech/sci'
                        probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                    return guess_result, probs ,confidence_result

        else:
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null', probs, 1.0



    def random_label(self,index, label_type):
        """ Generates a label based on the type and index """
        if label_type == 'uppercase':
            return f"{chr(65 + index)})"  # ASCII for A, B, C,...
        elif label_type == 'lowercase':
            return f"{chr(97 + index)})"  # ASCII for a, b, c,...
        elif label_type == 'roman':
            roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
            return f"{roman_numerals[index]})"

    def generate_options(self,options):
        # Define the options and distractors
        options = options
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
            label = self.random_label(i, label_type)
            option_str += f"{label} {option}  "

        # Add distractors if needed
        if include_distractors:
            # Randomly add one or more distractors
            num_distractors = random.randint(1, len(distractors))
            chosen_distractors = random.sample(distractors, num_distractors)
            for i, distractor in enumerate(chosen_distractors, start=len(options)):
                label = self.random_label(i, label_type)
                option_str += f"{label} {distractor}  "

        # Return formatted options string
        return option_str.strip()

    def predict_sentiment(self,text):
        
        template, answer_template = random.choice(PATTERNS['sst2'])

        # Prepare options
        # options = "positive, negative"  # This should match your scenario, could also be ['Positive', 'Negative'] if handling a list 
        # options = random.choice(PATTERNS_OPTIONS['sst2'])
        options = self.generate_options( ['negative', 'positive'])


        # prompt = f"[INST]What is the sentiment of the following movie review sentence, and provide your prediction confidence as a percentage? \n{text}\npositive, negative[/INST]"
        
        prompt = template.format(sentence=text, options_=options) 
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Get model predictions as text
        generate_args = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "do_sample": True,  # enable sampling
                "top_k": 40,  # top-k sampling
                "top_p": 0.92,  # nucleus sampling probability
                "temperature": 0.7,  # sampling temperature
                "max_new_tokens":200
            }

        with torch.no_grad():
            outputs = model.generate(**generate_args)
            # outputs = model.generate(**inputs, max_new_tokens=200,temperature = 0.01 ,top_k= 1)
        
            # full_text  = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the generated text from the model
        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
        print ('Text inf:', text)
        print("Generated Text:", generated_text)

        # Use regex to extract sentiment and confidence
        match_sentiment = re.search(r'positive|POSITIVE|Positive|negative|NEGATIVE|Negative', generated_text)
        match_confidence = re.search(r"[-+]?\d*\.?\d+", generated_text)
        print ('match_sentiment',match_sentiment,'match_confidence',match_confidence)
        
        
        if match_sentiment :
            sentiment_result = match_sentiment.group(0).lower()
            return sentiment_result 
        else: 
            return 'null'

    def predict_generate(self,text):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
            class_number  = 2
            label_list = ['positive', 'negative']
        elif self.task == 'ag_news':
            guess_pattern = r'(world|business|tech|science|sports)'
            class_number  = 4
            label_list = ['world','business', 'tech','science','tech/sci','sci/tech', 'sports']

    
    

        if self.task == 'sst2':
            prompt = f"""{self.start_prompt_header}Provide your best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
        
        elif self.task == 'ag_news':
            
            options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            
            prompt = f"""{self.start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

        # Get model predictions as text
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("Generated Prediction Text (Single):", generated_text)

        # Use regex to extract the guess (sentiment or category) and verbal confidence
        match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
        if match_guess:
            guess_result = match_guess.group(0).lower()
            if guess_result not in label_list:
                guess_result = 'null'
            else:
                print ('guess_result',guess_result) 
                if guess_result == 'tech' or guess_result == 'science' or guess_result == 'tech/sci':
                    guess_result = 'sci/tech'
                return guess_result
        else:
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null' 

    def perform_multiple_predictions(self,text, n=20):
        results = []
        for _ in range(n):
            result = self.predict_generate(text)
            results.append(result)
        return results

    def predict_sentiment_and_empirical_confidence(self,text, expected_sentiment):
        # we need to change the emprirical function so that we have another function that returns something compatible with
        # the __call__ function
        if self.task == 'sst2':
            class_number = 2
        elif self.task == 'ag_news':
            class_number = 4
        task_dictionary_counts = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
        task_dictionary_confidences = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
        
        print ('perform multiple predictions')
        results = self.perform_multiple_predictions(text)
        correct_predictions = sum(1 for sentiment in results if sentiment == expected_sentiment)
        confidence_empirical = (correct_predictions / len(results)) * 100

        # sentiment_counts = { 'positive': 0, 'negative': 0, 'null': 0 }
        for sentiment in results:
            task_dictionary_counts[self.task][sentiment] += 1
        
        # sentiment_confidences = { 'positive': 0, 'negative': 0, 'null': 0 }
        for sentiment, number_of_results in task_dictionary_counts[self.task].items():
            task_dictionary_confidences[self.task][sentiment] = (number_of_results / len(results))

        # average_confidence = sum(confidence for _, confidence in results) / len(results)

        print(f"Results for '{text}':")
        print(f"Counter: {task_dictionary_counts[self.task]}")
        # print(f"Average model confidence: {average_confidence}%")
        print(f"Empirical confidence: {confidence_empirical}%")
        max_class = max(task_dictionary_counts[self.task], key=task_dictionary_counts[self.task].get)
        print ('max_class',max_class,expected_sentiment)
        print ('empricial:',expected_sentiment,confidence_empirical,task_dictionary_confidences[self.task])
        # return guess_result, probs, confidence_result
        # normalized_task_dictionary_confidences = task_dictionary_confidences[self.task]
        # normalized_task_dictionary_confidences['null'] = 0
        # del normalized_task_dictionary_confidences
        # remaining_mass = sum(normalized_task_dictionary_confidences.velues())
        # normalized_task_dictionary_confidences = {i:j/remaining_mass for i,j in normalized_task_dictionary_confidences.items() if j!=0 else pass }
        # print ('normalized_task_dictionary_confidences',normalized_task_dictionary_confidences)
        guess_result = max_class
        confidence_result = confidence_empirical/100
        if guess_result != 'null':
            if self.task == 'sst2':
                probs = np.array([task_dictionary_confidences[self.task]['negative'],
                    task_dictionary_confidences[self.task]['positive'],
                    task_dictionary_confidences[self.task]['null']])
                # if guess_result == 'positive':
                    # probs = np.array([task_dictionary_confidences[self.task]['negative'],task_dictionary_confidences[self.task]['positive'],task_dictionary_confidences[self.task]['null']])
                    # probs = np.array([1-confidence_result,confidence_result,0.0])
                # else:
                #     probs = np.array([task_dictionary_confidences[self.task]['negative'],task_dictionary_confidences[self.task]['positive'],task_dictionary_confidences[self.task]['null']])
                    # probs = np.array([confidence_result,1-confidence_result,0.0])
                return guess_result, probs ,confidence_result
            elif self.task == 'ag_news':
                # granularity = 4 # how many confidence labels do we have in confidence elicitation
                # ignore_classes  = 1 # ignore null and the current class
                # other_level_candidates = class_number
                
                # confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                # confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                probs = np.array([task_dictionary_confidences[self.task]['world'],
                task_dictionary_confidences[self.task]['sports'],
                task_dictionary_confidences[self.task]['business'],
                task_dictionary_confidences[self.task]['sci/tech'],
                task_dictionary_confidences[self.task]['null'],
                ])
                print ('probs',probs)
                # if guess_result == 'world':
                #     probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                # elif guess_result == 'sports' :
                #     probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                # elif guess_result == 'business':
                #     probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                # if guess_result == 'tech' or 'science' or 'tech/science':
                #     guess_result = 'tech/sci'
                    # probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                return guess_result, probs ,confidence_result
        else:        
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null', probs, 1.0
            

        # return expected_sentiment,confidence_empirical,sentiment_confidences
    def predict_sentiment_and_verbal_confidence_k_pred_avg(self,text, expected_prediction):
        k_pred = 10
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
            class_number  = 2
            label_list = self.dataset.label_names# ['positive', 'negative']
        elif self.task == 'ag_news':
            guess_pattern = r'(world|business|tech|science|sports)'
            class_number  = 4
            label_list = self.dataset.label_names# ['world','business', 'tech','science','tech/sci', 'sports']

        task_dictionary_counts = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
        task_dictionary_confidences = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
            



        if self.task == 'sst2':
            prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a coma, for example [Negative, Positive, Positive, Negative ...]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
        
        elif self.task == 'ag_news':
            
            options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            
            prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses!>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
            
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

        # Get model predictions as text
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print ('prompt', prompt)
        print("Generated Prediction Text:", generated_text) 
        # Use regex to extract the guess (sentiment or category) and verbal confidence
        # match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
        # if match_guess:
        #     guess_result = match_guess.group(0).lower()
        #     if guess_result not in label_list:
        #         guess_result = 'null'
        # else:
        #     probs = np.zeros(class_number+1)
        #     probs[-1] = 1.0  # Only null has confidence 1
        #     return 'null', probs, 1.0

        # Extract guesses, assuming they're separated by commas and ignoring case
        results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
        results = [result for result in results if result in label_list]
        # If fewer results than k_pred, fill with 'null'
        results.extend(['null'] * (k_pred - len(results)))
        print ('results',results,expected_prediction)
        correct_predictions = sum(1 for sentiment in results if sentiment == expected_prediction)
        confidence_empirical = (correct_predictions / len(results)) * 100
        # Manually count occurrences

        # sentiment_counts = {k:0 for k in label_list}
        # sentiment_counts['null'] = 0
        # print ('count',sentiment_counts)
        # for guess in results:
        #     if guess in sentiment_counts:
        #         sentiment_counts[guess] += 1
        #     else:
        #         sentiment_counts[guess] = 1

        # # percentages = {key: (value / k_pred) * 100 for key, value in sentiment_counts.items()} 
        # # print ('percentages',percentages)
    
        # sentiment_confidences = {k:0 for k in label_list}
        # sentiment_confidences['null'] = 0
        # for sentiment, number_of_results in sentiment_counts.items():
        #     sentiment_confidences[sentiment] = (number_of_results / len(results))

    
        # print(f"Results for '{text}':")
        # print(f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Null: {sentiment_counts['null']}")
        # # print(f"Average model confidence: {average_confidence}%")
        # print(f"Empirical confidence: {confidence_empirical}%")
        # max_class = max(sentiment_counts, key=sentiment_counts.get)
        # print ('max_class',max_class,expected_prediction)

        for sentiment in results:
            task_dictionary_counts[self.task][sentiment] += 1
        
        # sentiment_confidences = { 'positive': 0, 'negative': 0, 'null': 0 }
        for sentiment, number_of_results in task_dictionary_counts[self.task].items():
            task_dictionary_confidences[self.task][sentiment] = (number_of_results / len(results))

        # average_confidence = sum(confidence for _, confidence in results) / len(results)

        print(f"Results for '{text}':")
        print(f"Counter: {task_dictionary_counts[self.task]}")
        # print(f"Average model confidence: {average_confidence}%")
        print(f"Empirical confidence: {confidence_empirical}%")
        max_class = max(task_dictionary_counts[self.task], key=task_dictionary_counts[self.task].get)
        print ('max_class',max_class,expected_prediction)
        print ('empricial:',expected_prediction,confidence_empirical,task_dictionary_confidences[self.task])


        guess_result = max_class
        confidence_result = confidence_empirical/100
        if guess_result != 'null':
            if self.task == 'sst2':
                probs = np.array([task_dictionary_confidences[self.task]['negative'],
                    task_dictionary_confidences[self.task]['positive'],
                    task_dictionary_confidences[self.task]['null']])
                # if guess_result == 'positive':
                    # probs = np.array([task_dictionary_confidences[self.task]['negative'],task_dictionary_confidences[self.task]['positive'],task_dictionary_confidences[self.task]['null']])
                    # probs = np.array([1-confidence_result,confidence_result,0.0])
                # else:
                #     probs = np.array([task_dictionary_confidences[self.task]['negative'],task_dictionary_confidences[self.task]['positive'],task_dictionary_confidences[self.task]['null']])
                    # probs = np.array([confidence_result,1-confidence_result,0.0])
                return guess_result, probs ,confidence_result
            elif self.task == 'ag_news':
                # granularity = 4 # how many confidence labels do we have in confidence elicitation
                # ignore_classes  = 1 # ignore null and the current class
                # other_level_candidates = class_number
                
                # confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
                # confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
                probs = np.array([task_dictionary_confidences[self.task]['world'],
                task_dictionary_confidences[self.task]['sports'],
                task_dictionary_confidences[self.task]['business'],
                task_dictionary_confidences[self.task]['sci/tech'],
                task_dictionary_confidences[self.task]['null'],
                ])
                print ('probs',probs)
                # if guess_result == 'world':
                #     probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
                # elif guess_result == 'sports' :
                #     probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
                # elif guess_result == 'business':
                #     probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
                # if guess_result == 'tech' or 'science' or 'tech/science':
                #     guess_result = 'tech/sci'
                    # probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
                return guess_result, probs ,confidence_result
        else:        
            probs = np.zeros(class_number+1)
            probs[-1] = 1.0  # Only null has confidence 1
            return 'null', probs, 1.0

    def __call__(self, text_input_list,ground_truth_output):
        
        self.device = next(self.model.parameters()).device
        # print ('device',self.device)
        logit_list = []
        for t in text_input_list:
            datapoint = (t,ground_truth_output)
            
            # if self.task == 'sst2':
                # class_number = 2
                # print ('ground_truth_output',ground_truth_output)
                # expected_prediction = self.dataset.label_names[ground_truth_output]
                # expected_prediction = 'negative' if ground_truth_output == 0 else 'positive'
            # elif self.task == 'ag_news':
                # class_number = 4
                
                # print ('label list',self.dataset.label_names,ground_truth_output )
                # expected_prediction = self.dataset.label_names[ground_truth_output]
                # sys.exit()
                #
            
            # elif task =='agnews': 
            #     # what to do? 
            self.inference_step +=1
            self.predictor.inference_step = self.inference_step 
            self.predictor.current_sample = self.current_sample
            guess, probs, confidence = self.predictor.predict_and_confidence(datapoint)


            # if self.prompting_type == 's1' or self.prompting_type == 's1_black_box':
            #     guess, probs, confidence  = self.predict_sentiment_and_verbal_confidence_s1(t, expected_sentiment=expected_prediction )
            
            # elif self.prompting_type == 'w1':
            #     guess, probs, confidence  = self.predict_sentiment_self_fool_wordattack_baseline_w1(t, expected_sentiment=expected_prediction )
            # elif self.prompting_type == '2step':
            #     guess, probs, confidence = self.predict_sentiment_and_verbal_confidence_2steps(t, expected_sentiment=expected_prediction )
            # elif self.prompting_type == 'empirical':
            #     guess, probs, confidence  = self.predict_sentiment_and_empirical_confidence(t, expected_sentiment=expected_prediction)
            # elif self.prompting_type == 'k_pred_avg':
            #     guess, probs, confidence = self.predict_sentiment_and_verbal_confidence_k_pred_avg(t, expected_prediction=expected_prediction)
            # elif self.prompting_type == '2step_k_pred_avg':
            #     guess, probs, confidence = self.predict_sentiment_and_verbal_confidence_2step_k_pred_avg(t, expected_prediction=expected_prediction)
            # elif self.prompting_type == 'sspattack':
            #     guess, probs, confidence = self.predict_sentiment_self_fool_wordattack_baseline_w1(t, expected_sentiment=expected_prediction)
            # elif self.prompting_type == 'e_guided_paraphrasing':
            #     guess, probs, confidence = self.predict_sentiment_and_verbal_confidence_2step_k_pred_avg(t, expected_prediction=expected_prediction)
            # else:
            #     guess, probs, confidence  = self.analyze_results(t, expected_sentiment=expected_prediction)
            # can set the correct confidence that includes - null, but set null to 0, so that we never attack towards this
            print ('guess',guess,probs,confidence)
            
            # if first step is null, we treat it as a miss classification 'usually', but because of thiss 
            # logic, we set the prediction to be correct...
            # if null
            # if inference on original sample: # this will skip the sample and move to next
                # probs = torch.zeros(class_number+1, dtype=torch.float16)
                # probs[-1] = 1.0
            # else: # if we get it correct though when we do inference on transformed text, we want to ignore
                # any sample that achives probs[-1] = 1
                # probs = torch.zeros(class_number+1, dtype=torch.float16)
                # probs[ground_truth_output] = 1.0

            # potentially, always set it to  probs[-1] = 1.0 then in transformation code, check
            # the logits or the prediction class, if the prediction class is -1 then ignore...

            # if guess == 'null': # return the logits with original label at 1
            #     probs = torch.zeros(class_number+1, dtype=torch.float16)
            #     probs[ground_truth_output] = 1.0
            
            if guess == 'null': # return the logits with original label at 1
                probs = torch.zeros(self.n_classes+1, dtype=torch.float16)
                probs[-1] = 1.0

            logit_list.append(torch.tensor(probs, device=self.device)) 
            
            
            

        print ('logit list:',logit_list)
        logit_tensor = torch.stack(logit_list)
        print('logit_tensor:', logit_tensor)
        return logit_tensor



args.dataset = dataset_class
model_wrapper = HuggingFaceLLMWrapper(**vars(args))#model, tokenizer)





# print ('dataset_class_1',len(dataset_class_1_t))
# print ('dataset_class_0',len(dataset_class_0_t))
# sys.exit()


# from textattack.transformations import Transformation

# class PromptFocusedTransformation(Transformation):
#     def __init__(self, ...):
#         # Initialization code



class LLMSelfWordSubstitutionW1(WordSwap):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # def __init__(self, tokenizer, model ,task,prompt_shot_type,num_transformations,goal_function):
    #     self.tokenizer = tokenizer
    #     self.model = model
    #     self.task = task
    #     self.goal_function = goal_function
    #     self.num_transformations = num_transformations
    #     self.prompt_shot_type = prompt_shot_type
    #     self.device = next(self.model.parameters()).device
        
    def _query_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        # Generate the output with the model
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)


        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print ('generated_text:',generated_text) 
        return generated_text.strip()

    # def _query_model(self, prompt):
    #     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
    #     generate_args = {
    #         "input_ids": inputs['input_ids'],
    #         "attention_mask": inputs['attention_mask'],
    #         "do_sample": True,  # enable sampling
    #         "top_k": 40,  # top-k sampling
    #         "top_p": 0.92,  # nucleus sampling probability
    #         "temperature": 0.7,  # sampling temperature
    #         "max_new_tokens": 200,
    #         'pad_token_id': self.tokenizer.eos_token_id
    #     }

    #     # Generate the output with the model
    #     with torch.no_grad():
    #         outputs = self.model.generate(**generate_args)


    #     prompt_length = len(inputs['input_ids'][0])
    #     generated_tokens = outputs[0][prompt_length:]
    #     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    #     # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print ('generated_text:',generated_text)
    #     # generated_text = ' this is a random sentence BUSINESS .Business (business) and ;sport; and business and businesss and lbusinessk and kbusiness'
    #     # Use regex to extract the modified text
    #     # match_pattern = "(?:Here's a possible new sentence:|Based on the given conditions, here's the new sentence:|Here's my attempt:|Here's a suggestion:|Here's a new sentence that fits your requirements:|Here is the new sentence:|One possible solution:|A possible aswer:|One possible new sentence:|New sentence:|Here is the solution:|A possible solution could be:|A possible solution for the given task could be:|One possible solution:|Here's a possible solution:)(?:[.,;?!])?"
    #     # match_generated_text = re.search(r"New sentence: (.+)", generated_text)
    #     # match_generated_text = re.search(rf"{match_pattern} (.+)", generated_text)
    #     match_generated_text = None
    #     if self.task == 'sst2' or self.task == 'strategyQA' :
    #         pattern = r"(?<=:)(.+)"  
    #         match_generated_text = re.search(pattern, generated_text)
    #     elif self.task == 'ag_news': # label leaking filtering
    #         substrings_to_remove = ['business', 'world', 'tech/sci','sci/tech', 'tech', 'science', 'sport', 'sports']
            

    #         pattern = r'\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b'

    #         # Remove the matched keywords and their surrounding punctuation, ensuring spacing is maintained.
    #         def replace(match):
    #             preceding = generated_text[max(0, match.start()-1)]
    #             following = generated_text[min(len(generated_text), match.end()):min(len(generated_text), match.end() + 1)]

    #             # Check for spaces to avoid having multiple spaces
    #             need_space = (preceding not in ' \t\n\r') and (following not in ' \t\n\r')

    #             if need_space:
    #                 return ' '
    #             else:
    #                 return ''

    #         clean_text = re.sub(r'[\[\]{}(),]*\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b[\[\]{}(),]*', replace, generated_text,flags=re.IGNORECASE)

    #         # Clean up remaining extra spaces left by the removals
    #         clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    #         generated_text = clean_text 

    #         pattern = r"(?<=:)(.+)"  
    #         match_generated_text = re.search(pattern, generated_text)
    #     print ('generated_text_after_cleanup:',generated_text)
    #     print ('match_generated_text',match_generated_text)
    #     if match_generated_text:
    #         print ('match_generated_text.group(1).strip()',match_generated_text.group(1).strip())
    #         return match_generated_text.group(1).strip()
    #     return generated_text.strip()

    # def _generate_prompt(self, context_sentence, expected_sentiment):
    #     if self.task not in ['sst2', 'ag_news']:
    #         raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

    #     if self.task == 'sst2':
    #         text_type = 'sentence' 
    #         expected_sentiment = 'positive' if expected_sentiment == 1 else 0
    #         expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
    #         label_types = ['negative', 'positive']
    #         filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
    #         other_classes = ' or '.join(filtered_label_types)
    #         attack_type = 'Add at most two semantically neutral words to the sentence..'
    #         original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
    #         attack_objective = (
    #             f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
    #             f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
    #             f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}. "
    #         )
    #         if self.prompt_shot_type == 'zs':
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
    #                 f"Only output the new {text_type} without anything else."
    #                 f"The new sentece is:"
    #             )
    #         elif  self.prompt_shot_type == 'fs':
    #             original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
    #             perturbed_example = ['The feline cat is on the table desk', 'The the boy is is playing soccer', 'She drove her car to work i think', 'The round sun is shining brightly mmmh', 'He cooked a large dinner for his family']
                
    #             list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
    #                 f"Here are five examples that fit the guidance: {list_examples}"
    #                 f"Only output the new {text_type} without anything else."
    #                 f"The new sentece is:"
    #             )
    #         # prompt = original_input + attack_objective + attack_guidance
    #         prompt =f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
    #         # print ('prompt',prompt)
            
    #     elif self.task == 'ag_news':
    #         text_type = 'sentence' 
    #         # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
    #         # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
    #         expected_sentiment_label = self.goal_function.ground_truth_output
            
    #         print ('expected sentiment',expected_sentiment) 
    #         label_types= self.dataset.label_names
    #         expected_sentiment = label_types[expected_sentiment_label]
    #         print ('label_types',label_types)
    #         filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
    #         other_classes = ' or '.join(filtered_label_types)
    #         attack_type = 'Add at most two semantically neutral words to the sentence.'
    #         original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
    #         attack_objective = (
    #             f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
    #             f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
    #             f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
    #             # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
    #         )
    #         if self.prompt_shot_type == 'zs':
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
    #                 f"Only output the new {text_type} without anything else."
    #                 f"The new sentece is:"
    #             )
    #         elif  self.prompt_shot_type == 'fs':
    #             original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
    #             perturbed_example = ['The cat cat is on the table table', 'The the boy is is playing soccer', 'She drove her car to work i think', 'The round sun is shining brightly mmmh', 'He cooked a large dinner for his family']
                
    #             list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} \n"
    #                 f"Here are five examples that fit the guidance: {list_examples} \n"
    #                 f"Only output the new {text_type} without anything else. Don't provide reasoning "
    #                 f"The new sentece is: "
    #             )
    #         # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
    #         #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
    #         # system_prompt = ''
    #         # <<SYS>>{system_prompt}<</SYS>>
    #         prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
    #         # prompt =  original_input + attack_objective + attack_guidance 

    #         # print ('attack_prompt:', prompt)

    #         # prompt = (
    #         #     f"[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium',"
    #         #     f"'high', 'highest') for the following news article. Give ONLY the guess and verbal confidence, no other "
    #         #     f"words or explanation. For example:\nGuess: <most likely guess, either world, sports, business or tech/"
    #         #     f"science; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', "
    #         #     f"'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the "
    #         #     f"confidence!>\n\nThe text is: {context_sentence} News type & confidence:[/INST]"
    #         # )
    #     print ('prompt',prompt)
    #     sys.exit()
    #     return prompt

    def _generate_prompt(self, context_sentence, label_index):
        # if self.task not in ['sst2', 'ag_news', 'strategyQA']:
        #     raise ValueError("Unsupported task. Please choose either 'sst2', 'ag_news', or 'strategyQA'.")

        label_list =  self.dataset.label_names 
        print ('label_index',label_index)
        expected_prediction, filtered_label_list = self.predictor.prompt_class._identify_correct_incorrect_labels(label_index)
            
        other_classes = ' or '.join(filtered_label_list)
        attack_type = 'Replace at most two words in the sentence with synonyms.'
        original_input = f"The original sentence '{context_sentence}' is classified as {expected_prediction}. "
        attack_objective = (
            f"Your task is to generate a new sentence which must satisfy the following conditions:\n"
            f"1) Keeping the semantic meaning of the new sentence unchanged.\n"
            f"2) The new sentence and the original sentence should be classified as {other_classes}.\n "
        )
        if self.prompt_shot_type == 'zs':
            attack_guidance = (
                f"You can finish the task by modifying sentence using the following guidance: {attack_type} "
                f"Only output the new sentence without anything else."
                f"The new sentence is: [INSERT YOUR SENTENCE TRANSFORMATION HERE]"
            )
        elif self.prompt_shot_type == 'fs':
            original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work', 'The sun is shining brightly', 'He cooked dinner for his family']
            perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']

            list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])

            attack_guidance = (
                f"You can finish the task by modifying sentence using the following guidance: {attack_type} "
                f"Here are five examples that fit the guidance: {list_examples}"
                f"Only output the new sentence without anything else."
                f"The new sentence is: [INSERT YOUR SENTENCE TRANSFORMATION HERE]"
            )
        prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
        
        return prompt

    def _generate_extractor_prompt(self, context_sentence, label_index):
        # if self.task not in ['sst2', 'ag_news', 'strategyQA']:
        #     raise ValueError("Unsupported task. Please choose either 'sst2', 'ag_news', or 'strategyQA'.")

        
        prompt = f'{self.start_prompt_header}' + f'The text in the brackets is what a langugage model has returned from a query [{context_sentence}] can you extract only the generated text and dont return anything else, if multiple answers are given return only 1! The text is:' + f'{self.end_prompt_footer}'
        
        return prompt
    # def _get_transformations(self, current_text, indices_to_modify):
    #     print ('Current_Text',current_text.attack_attrs )
    #     # original_index_map = current_text.attack_attrs['original_index_map']
    #     # print ('current_text.attack_attrs',self.ground_truth_output)
    #     print ('self.goal_function',self.goal_function )
    #     print ('self gto', self.goal_function.ground_truth_output)
    #     expected_sentiment =  self.goal_function.ground_truth_output
    #     context_sentence = current_text.text
    #     transformations = []
    #     for i in range(self.num_transformations):
    #         prompt = self._generate_prompt(context_sentence,expected_sentiment)
    #         new_sentence = self._query_model(prompt)

    #         if (new_sentence) and (new_sentence != context_sentence):
    #             Att_sen_new_sentence = AttackedText(new_sentence)
    #             # print ('words',Att_sen_new_sentence)
    #             # current_text.generate_new_attacked_text(Att_sen_new_sentence.words)
    #             print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
    #             print ('att sen new words',Att_sen_new_sentence.words, len(Att_sen_new_sentence.words) ) 
    #             Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
    #             Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
    #             # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(range(len(Att_sen_new_sentence.words)))
    #             # Att_sen_new_sentence.attack_attrs['original_index_map'] = original_index_map
    #             Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
    #             print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)

    #             if len(Att_sen_new_sentence.attack_attrs['original_index_map']) == 0:
    #                 continue
                
                
                
    #             transformations.append(Att_sen_new_sentence)
                
    #     return transformations 

    def _remove_labels(self, text):
        # Create a regex pattern with ignoring case and match as whole word \b
        pattern = r'\b(' + '|'.join(map(re.escape, self.dataset.label_names)) + r')\b'

        # Substitute the matched word with an empty string
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Removing extra spaces if any
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text

    def _get_transformations(self, current_text, indices_to_modify):
        print ('Current_Text',current_text.attack_attrs )
        # original_index_map = current_text.attack_attrs['original_index_map']
        # print ('current_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = current_text.text
        transformations = []
        
        
        prompt = self._generate_prompt(context_sentence,expected_sentiment)
        generated_sentence = self._query_model(prompt)
        print ('Generated_sentence:',generated_sentence)
        extract_ans_prompt = self._generate_extractor_prompt(generated_sentence,expected_sentiment)
        extract_ans_prompt = self._remove_labels(extract_ans_prompt) 

        new_sentence = self._query_model(extract_ans_prompt)
        print ('New_sentence:', new_sentence)
        if (new_sentence) and (new_sentence != context_sentence):
            Att_sen_new_sentence = AttackedText(new_sentence)
            # print ('words',Att_sen_new_sentence)
            # current_text.generate_new_attacked_text(Att_sen_new_sentence.words)
            print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
            print ('att sen new words',Att_sen_new_sentence.words, len(Att_sen_new_sentence.words) ) 
            Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
            # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(range(len(Att_sen_new_sentence.words)))
            # Att_sen_new_sentence.attack_attrs['original_index_map'] = original_index_map
            Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)

            # if the model returns a string with no words and only punctuation and nothering else e.g -> we return an empty list
            if len(Att_sen_new_sentence.attack_attrs['original_index_map']) == 0:
                return []
            
            
            
            transformations.append(Att_sen_new_sentence)
                
        return transformations 


# print ('current_text',current_text )
#         # print ('current_text.attack_attrs',self.ground_truth_output)
#         print ('self.goal_function',self.goal_function )
#         print ('self gto', self.goal_function.ground_truth_output)
#         expected_sentiment =  self.goal_function.ground_truth_output
#         context_sentence = current_text.text
#         transformations = []
#         for i in range(self.num_transformations):
            
#             prompt = self._generate_prompt(context_sentence,expected_sentiment)
            
#             new_sentence = self._query_model(prompt) 
#             print ('new_sentences_explore',new_sentence) 

#             if new_sentence and new_sentence != context_sentence:
#                 Att_sen_new_sentence = AttackedText(new_sentence)
#                 print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
#                 Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
#                 Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
#                 print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
#                 transformations.append(Att_sen_new_sentence)

#         print ('transformations',transformations)



class LLMSelfFoolS2(WordSwap):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

 

    def _query_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        # Generate the output with the model
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)


        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print ('generated_text:',generated_text)
 
        match_generated_text = None
        if self.task == 'sst2':
            pattern = r"(?<=:)(.+)"  
            match_generated_text = re.search(pattern, generated_text)
        elif self.task == 'ag_news': # label leaking filtering
            substrings_to_remove = ['business', 'world', 'tech/sci','sci/tech', 'tech', 'science', 'sport']
            

            pattern = r'\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b'

            # Remove the matched keywords and their surrounding punctuation, ensuring spacing is maintained.
            def replace(match):
                preceding = generated_text[max(0, match.start()-1)]
                following = generated_text[min(len(generated_text), match.end()):min(len(generated_text), match.end() + 1)]

                # Check for spaces to avoid having multiple spaces
                need_space = (preceding not in ' \t\n\r') and (following not in ' \t\n\r')

                if need_space:
                    return ' '
                else:
                    return ''

            clean_text = re.sub(r'[\[\]{}(),]*\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b[\[\]{}(),]*', replace, generated_text,flags=re.IGNORECASE)

            # Clean up remaining extra spaces left by the removals
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            generated_text = clean_text 

            pattern = r"(?<=:)(.+)"  
            match_generated_text = re.search(pattern, generated_text)
        print ('generated_text_after_cleanup:',generated_text)
        print ('match_generated_text',match_generated_text)
        if match_generated_text:
            print ('match_generated_text.group(1).strip()',match_generated_text.group(1).strip())
            return match_generated_text.group(1).strip()
        return generated_text.strip()

 

    def _generate_prompt(self, context_sentence, expected_sentiment):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            text_type = 'sentence' 
            expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            label_types = ['negative', 'positive']
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Paraphrase the sentence.'
            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
            attack_objective = (
                f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
                f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
                f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}. "
            )
            if self.prompt_shot_type == 'zs':
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            elif  self.prompt_shot_type == 'fs':
                original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
                perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
                
                list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Here are five examples that fit the guidance: {list_examples}"
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            # prompt = original_input + attack_objective + attack_guidance
            prompt =f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
            # print ('prompt',prompt)
            
        elif self.task == 'ag_news':
            text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
            expected_sentiment_label = self.goal_function.ground_truth_output
            
            print ('expected sentiment',expected_sentiment) 
            label_types= self.dataset.label_names
            expected_sentiment = label_types[expected_sentiment_label]
            print ('label_types',label_types)
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Replace at most two words in the sentence with synonyms.'
            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
            attack_objective = (
                f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
                f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
                f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
                # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
            )
            if self.prompt_shot_type == 'zs':
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            elif  self.prompt_shot_type == 'fs':
                original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
                perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
                
                list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Here are five examples that fit the guidance: {list_examples} "
                    f"Only output the new {text_type} without anything else. Don't provide reasoning "
                    f"The new sentece is: "
                )
            # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
            #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
            # system_prompt = ''
            # <<SYS>>{system_prompt}<</SYS>>
            prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
  
        return prompt

    def _get_transformations(self, current_text, indices_to_modify):
        print ('current_text',current_text )
        # print ('current_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = current_text.text
        transformations = []
        for i in range(self.num_transformations):
            prompt = self._generate_prompt(context_sentence,expected_sentiment)
            new_sentence = self._query_model(prompt)

            if new_sentence and new_sentence != context_sentence:
                transformations.append(AttackedText(new_sentence))
        return transformations 




class LLMEGuidedParaphrasing(WordSwap):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

 

    def _query_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 1,  # nucleus sampling probability
            "temperature": 1,  # sampling temperature
            "max_new_tokens": 2000,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        # Generate the output with the model
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)


        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print ('generated_text:',generated_text)
 
        match_generated_text = None
        if self.task == 'sst2':
            pattern = r"(?<=:)(.+)"  
            match_generated_text = re.search(pattern, generated_text)
        elif self.task == 'ag_news': # label leaking filtering
            substrings_to_remove = ['business', 'world', 'tech/sci','sci/tech', 'tech', 'science', 'sport']
            

            pattern = r'\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b'

            # Remove the matched keywords and their surrounding punctuation, ensuring spacing is maintained.
            def replace(match):
                preceding = generated_text[max(0, match.start()-1)]
                following = generated_text[min(len(generated_text), match.end()):min(len(generated_text), match.end() + 1)]

                # Check for spaces to avoid having multiple spaces
                need_space = (preceding not in ' \t\n\r') and (following not in ' \t\n\r')

                if need_space:
                    return ' '
                else:
                    return ''

            clean_text = re.sub(r'[\[\]{}(),]*\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b[\[\]{}(),]*', replace, generated_text,flags=re.IGNORECASE)

            # Clean up remaining extra spaces left by the removals
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            generated_text = clean_text 

            pattern = r"(?<=:)(.+)"  
            match_generated_text = re.search(pattern, generated_text)
        print ('generated_text_after_cleanup:',generated_text)
        print ('match_generated_text',match_generated_text)
        if match_generated_text:
            print ('match_generated_text.group(1).strip()',match_generated_text.group(1).strip())
            return match_generated_text.group(1).strip()
        return generated_text.strip()

 

    def _generate_prompt(self, context_sentence, expected_sentiment):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            # text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            # label_types = ['negative', 'positive']
            # filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            # other_classes = ' or '.join(filtered_label_types)
            # attack_type = 'Rewrite the sentance.'
            # original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment} \n. "
            # attack_objective = (
            #     f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
            #     f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
            #     f"2) The new {text_type} should be classified as {other_classes}.\n"
            #     )
            # if self.prompt_shot_type == 'zs':
            #     attack_guidance = (
            #         f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
            #         f"Only output the new {text_type} without anything else."
            #         f"The new sentece is:"
            #     )
            # elif  self.prompt_shot_type == 'fs':
            #     # original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            #     # perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
            #     original_example = ['This is a great day', 'never again will I play this game', 'She drove a wonderfull car to work']
            #     perturbed_example = ['The day was wonderful', 'last time that this game will be played is today', 'She guided her good vehicle to work']
                
            #     list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
            #     attack_guidance = (
            #         f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
            #         f"Here are five examples that fit the guidance: {list_examples}.\n"
            #         f"Only output the new {text_type} without anything else.\n"
            #         f"The new sentece is:"
            #     )
                

            text_type = 'sentence'
            expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            label_types = ['negative', 'positive']
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Rewrite the sentence.'

            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}.\n"
            examples = [
                ("It's a wonderful and uplifting journey.", "It's a decent and somewhat inspiring journey."),
                ("The movie is a delightful experience.", "The movie is a fairly enjoyable experience.")
            ]

            example_prompts = ""
            for pos, neg in examples:
                example_prompts += f"- Original: '{pos}'\n  New: '{neg}'\n  Explanation: The semantic meaning remains the same, but the emotional tone has been subtly shifted to be more neutral.\n\n"

            attack_objective = (f"Your task is to generate a new {text_type}")
            
            #  (
            #     f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
            #     # f"1) create a large semantic meaning change of the new {text_type}, however, the meaning the information and message should be the same.\n"
            #     # f"2) The new {text_type} should be classified as {other_classes}.\n"
            #     # f"3) Make subtle changes to adjust the sentiment while keeping the overall intent and meaning unaltered.\n" 
            #     # f"Here are some examples:\n{example_prompts}"
            #     # f"Think through the following steps to complete your task:\n"
            #     # f"Step 1: Rephrase or paraphrase the entire original sentance.\n"
            #     # f"Step 2: Ensure this change has a large semantic change.\n"
            #     # f"Step 3: Determine words or phrases that convey the positive sentiment in the new paraphrased sentance and list them.\n"
            #     # f"Step 4: Reword this new pharaprased sentance with counterfitted word substitutions\n"
            #     # f"Step 5: Ensure the new {text_type} maintains the original meaning.\n"
            # )

            # attack_guidance = (
            #     f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
            #     f"Only output the new {text_type} without anything else.\n"
            #     f"The new sentence is:"
            # )

            attack_guidance = (
                f"You can finish the task by modifying {text_type} by rewriting the sentance, be creative with your response. "
                f"Only output the new {text_type} without anything else.\n"
                f"The new sentence is:"
            )
            prompt =f'{self.start_prompt_header}' + original_input + attack_objective  + attack_guidance + f'{self.end_prompt_footer}'
             
            
            
            print ('prompt',prompt)
            
        elif self.task == 'ag_news':
            text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
            expected_sentiment_label = self.goal_function.ground_truth_output
            
            print ('expected sentiment',expected_sentiment) 
            label_types= self.dataset.label_names
            expected_sentiment = label_types[expected_sentiment_label]
            print ('label_types',label_types)
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Replace at most two words in the sentence with synonyms.'
            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
            attack_objective = (
                f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
                f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
                f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
                # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
            )
            if self.prompt_shot_type == 'zs':
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            elif  self.prompt_shot_type == 'fs':
                original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
                perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
                
                list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Here are five examples that fit the guidance: {list_examples} "
                    f"Only output the new {text_type} without anything else. Don't provide reasoning "
                    f"The new sentece is: "
                )
            # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
            #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
            # system_prompt = ''
            # <<SYS>>{system_prompt}<</SYS>>
            prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
  
        return prompt

    def _get_transformations(self, current_text, indices_to_modify):
        print ('current_text',current_text )
        # print ('current_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = current_text.text
        transformations = []
        for i in range(self.num_transformations):
            # prompt = self._generate_prompt(context_sentence,expected_sentiment)
            
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences',new_sentences) 

            # if new_sentence and new_sentence != context_sentence:
            #     Att_sen_new_sentence = AttackedText(new_sentence)
            #     print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
            #     Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            #     Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            #     print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
            #     transformations.append(Att_sen_new_sentence)

            prompt = self._generate_prompt(context_sentence,expected_sentiment)
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences_explore',new_sentences)
            # sys.exit()
            new_sentence = self._query_model(prompt) 
            print ('new_sentences_explore',new_sentence) 

            if new_sentence and new_sentence != context_sentence:
                Att_sen_new_sentence = AttackedText(new_sentence)
                print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
                Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
                Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
                print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
                transformations.append(Att_sen_new_sentence)

        print ('transformations',transformations)

        return transformations  
    
    def _generate_prompt_maximise_semantic_sim(self, context_sentence, best_sentence):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2': 

            text_type = 'sentence'
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            # label_types = ['negative', 'positive']
            # filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            # other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Rewrite the sentence.'

            # original_input = f"The original {text_type} '{best_sentence}' is very different to the original {context_sentence}.\n"
            original_input = f"The original {text_type} '{best_sentence}'"
            # attack_objective = (f"Your task is to increase the semantic similarity between '{best_sentence}' and '{context_sentence} \n'")
            attack_objective = (f"Your task is to rephrase the following sentence: '{context_sentence}' to make it slightly similar to '{best_sentence}', don't make many changes!")
             

            attack_guidance = (
                # f"You can finish the task by modifying {text_type} by rewriting the sentance, be creative with your response. "
                f"Only output the new {text_type} without anything else.\n"
                f"The new sentence is:"
            )
            prompt =f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
             
            
            
            print ('prompt',prompt)
            
        elif self.task == 'ag_news':
            text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
            expected_sentiment_label = self.goal_function.ground_truth_output
            
            print ('expected sentiment',expected_sentiment) 
            label_types= self.dataset.label_names
            expected_sentiment = label_types[expected_sentiment_label]
            print ('label_types',label_types)
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Replace at most two words in the sentence with synonyms.'
            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
            attack_objective = (
                f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
                f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
                f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
                # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
            )
            if self.prompt_shot_type == 'zs':
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            elif  self.prompt_shot_type == 'fs':
                original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
                perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
                
                list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Here are five examples that fit the guidance: {list_examples} "
                    f"Only output the new {text_type} without anything else. Don't provide reasoning "
                    f"The new sentece is: "
                )
            # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
            #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
            # system_prompt = ''
            # <<SYS>>{system_prompt}<</SYS>>
            prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
  
        return prompt

    def _maximise_semantic_sim(self, original_attacked_text, best_attacked_text ):
        print ('current_text',original_attacked_text )
        # print ('original_attacked_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = original_attacked_text.text
        best_sentence = best_attacked_text.text
        transformations = []
        for i in range(self.num_transformations):
            # prompt = self._generate_prompt(context_sentence,expected_sentiment)
            
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences',new_sentences) 

            # if new_sentence and new_sentence != context_sentence:
            #     Att_sen_new_sentence = AttackedText(new_sentence)
            #     print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
            #     Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            #     Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            #     print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
            #     transformations.append(Att_sen_new_sentence)

            prompt = self._generate_prompt_maximise_semantic_sim(context_sentence,best_sentence)

            print ('prompt increase semantic sim',prompt)
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences_explore',new_sentences)
            # sys.exit()
            new_sentence = self._query_model(prompt) 
            print ('new_sentences_explore',new_sentence) 

            if new_sentence and new_sentence != context_sentence:
                Att_sen_new_sentence = AttackedText(new_sentence)
                print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
                Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
                Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
                print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
                transformations.append(Att_sen_new_sentence)

        print ('transformations',transformations)
        return transformations







# import nltk
# from nltk.corpus import wordnet as wn
# from textattack.transformations import Transformation
# from textattack.shared.utils import is_one_word

# nltk.download('wordnet')
# nltk.download('omw-1.4')

# class PromptBasedWordSwapWordNet(Transformation):
#     def __init__(self, language="eng"):
#         if language not in wn.langs():
#             raise ValueError(f"Language {language} not one of {wn.langs()}")
#         self.language = language

#     def _get_replacement_words(self, word):
#         """Retrieve synonyms for a word from WordNet in the specified language."""
#         synonyms = set()
#         for syn in wn.synsets(word, lang=self.language):
#             for lemma in syn.lemmas(lang=self.language):
#                 syn_word = lemma.name().replace('_', ' ')
#                 if is_one_word(syn_word) and syn_word.lower() != word.lower():
#                     synonyms.add(syn_word)
#         return list(synonyms)

#     def _get_transformations(self, current_text, original_text=None):
#         """Generates transformations for the text within a prompt."""
#         transformed_texts = []
#         text_under_analysis_start = "Text under analysis: '"
#         text_under_analysis_end = "'\n\nSentiment:"

#         print ('current_text',current_text)
#         start_index = current_text.find(text_under_analysis_start) + len(text_under_analysis_start)
#         end_index = current_text.find(text_under_analysis_end)
#         if start_index < len(text_under_analysis_start) or end_index == -1:
#             # The specific markers are not found; likely an issue with the input format
#             return []

#         # Extract the portion of the prompt that is to be analyzed
#         text_under_analysis = current_text[start_index:end_index]

#         words = text_under_analysis.split()

#         for i, word in enumerate(words):
#             replacement_words = self._get_replacement_words(word)
#             for replacement in replacement_words:
#                 if replacement == word:
#                     continue
#                 # Create the new text with the word replaced
#                 new_words = words.copy()
#                 new_words[i] = replacement  # Replace the word
#                 new_text_under_analysis = " ".join(new_words)
#                 # Reconstruct the full text including the prompt with the new text_under_analysis
#                 new_text = current_text[:start_index] + new_text_under_analysis + current_text[end_index:]

#                 transformed_texts.append(new_text)

#         return transformed_texts

# # Usage example with a prompt structure
# # text = """Please perform a sentiment analysis on the provided text. Your task is to classify the sentiment of the text as either positive or negative. After analyzing the text, output only one of the following tokens: [Positive] or [Negative], based on the sentiment expressed in the text. Do not provide additional explanations or text.

# #                 Text under analysis: 'This movie is amazing and exciting.'

# #                 Sentiment:"""
# transformation = PromptBasedWordSwapWordNet()







# Define the search method

# Define the goal function
from textattack.goal_functions import ClassificationGoalFunction


# UntargetedClassification(ClassificationGoalFunction) (the same as before)

#ClassificationGoalFunction(GoalFunction) # the same as before

# GoalFunction # need to change the get_results so it takes an extra variable to singnal we are doing transformations
# def get_results # polymorphysim

# since it's all inheritence, can we not do polymorphisim in CustomConfidenceGoalFunction by defining get_results?


# we have the goal function class accessible in the beamsearch class, set a hyperparameter to true when we start the transformation inf? this can then be accessed 
# by the huggingface class?

class CustomConfidenceGoalFunction(UntargetedClassification):
    def __init__(self, *args, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        # """
        # Check if the confidence of the true label is within the target range.
        # """
        # print ('model_output',model_output,self.ground_truth_output, model_output.softmax(dim=-1))
        # # true_label_confidence = model_output.softmax(dim=-1)[self.ground_truth_output].item()
        # true_label_confidence = model_output[self.ground_truth_output].item()
        # print ('true_label_confidence is goal complete',true_label_confidence, self.lower_bound <= true_label_confidence <= self.upper_bound)
        
        # return self.lower_bound <= true_label_confidence <= self.upper_bound
    
        # print ('true_label_confidence is goal complete',model_output[self.ground_truth_output])
        if self.target_max_score:
            print ('1')
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):  
            print ('2')
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            print ('3', model_output.argmax(),self.ground_truth_output, model_output.argmax() != self.ground_truth_output)
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        # print ('true_label_confidence is get score',model_output[self.ground_truth_output],1 - model_output[self.ground_truth_output])
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            # print ('model output get score',model_output,self.ground_truth_output, 1 - model_output[self.ground_truth_output])

            # issue here, need to change so that it's not 1-, but instead is current prediction score
            # we are doing now [negative 0.7000, positive 0.2500, null 0.0500], 1-0.25 because ground truth point 1, but we want
            # confidence 0.7
            return 1 - model_output[self.ground_truth_output]
        
        
    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []
        
 
        inputs = [at.tokenizer_input for at in attacked_text_list]
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + self.batch_size]
            batch_preds = self.model(batch,self.ground_truth_output)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                # outputs.append(batch_preds)
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            i += self.batch_size

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], np.ndarray):
            outputs = np.concatenate(outputs).ravel()

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)
        
    def get_results(self, attacked_text_list, check_skip=False): 
        # this is what get_goal_results in search function calls indirectly so get_goal_results=get_results
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        print ('attacked_text_list',attacked_text_list,check_skip)
        print ('self.num_queries',self.num_queries)
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries 
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget


goal_function = CustomConfidenceGoalFunction(model_wrapper,query_budget=args.query_budget)
# goal_function = UntargetedClassification(model_wrapper)


args.goal_function = goal_function
 
# args.dataset = dataset_class
if args.transformation_method == 's1' or args.transformation_method == 's1_black_box' or  args.transformation_method == '2step' or args.transformation_method =='empirical' or args.transformation_method =='k_pred_avg'  :
    transformation = WordSwapWordNet() #WordSwapEmbedding(max_candidates=50)
elif args.transformation_method =='word_swap_embedding':
    transformation = WordSwapEmbedding(max_candidates=args.n_embeddings)
elif args.transformation_method =='sspattack':
    transformation = WordSwapEmbedding(max_candidates=args.n_embeddings)
elif args.transformation_method =='texthoaxer':
    transformation = WordSwapEmbedding(max_candidates=args.n_embeddings)
elif args.transformation_method == 'self_word_sub':
    transformation = LLMSelfWordSubstitutionW1(**vars(args))
elif args.transformation_method == 'e_guided_paraphrasing':
    transformation = LLMEGuidedParaphrasing(**vars(args))

# print ('transformation',args.transformation)
args.transformation = transformation




"""
Beam Search
===============

"""

import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod

class BlackBoxSearch(SearchMethod):
    """A black-box search that queries the model only to get transformations
    and evaluates each transformation to determine if it meets the goal.

    Args:
        num_transformations (int): The number of transformations to generate for each query.
    """
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # def __init__(self, num_transformations=20):
        #     self.num_transformations = num_transformations
        

    def perform_search(self, initial_result):
        print ('starting queries',self.goal_function.num_queries)
        self.number_of_queries = 0
        for i in range(self.num_transformations): # we ask the model N types of perturbations then


            # Get transformations using the custom transformation method 
            transformed_text_candidates = self.get_transformations(
            initial_result.attacked_text,
            original_text=initial_result.attacked_text,
            indices_to_modify=None,  # Modify the entire text
            )  
            self.number_of_queries +=1 # to get trasnformations we have to query 1 time the model
            # random.shuffle(transformed_text_candidates)
            # transformed_text_candidates = transformed_text_candidates[:min(self.num_transformations, len(transformed_text_candidates))]
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            if not transformed_text_candidates:
                continue # try to get another transformation 

            # Evaluate each transformation
            results, search_over = self.get_goal_results(transformed_text_candidates)
            #get the number of classes, then do #classes+1 in results.predicted pop any that meets this value
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter',results)
            print ('null_label',null_label)
            

            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            print ('results_after_null_filter',results)
            # Return the first successful perturbation
            for result in results:
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    print ('successful adv sample ', self.goal_function.num_queries, self.number_of_queries )
                    self.goal_function.num_queries += self.number_of_queries # if adv sample found we query model N times to find a suitable transformation then N times to check it's actually adv
                    return result

        print ('ending queries',self.goal_function.num_queries,self.number_of_queries )
        self.goal_function.num_queries += self.number_of_queries # no succesful adv samples found, we still query model N times to generate transformations
        
        return initial_result

    # def get_transformations(self, current_text, original_text=None, indices_to_modify=None):
    #     """Generate N transformations using the transformation method."""
    #     print ('current_text2',current_text)
    #     if hasattr(self.transformation, "transform"):
    #         print ('list transforms',list(self.transformation.transform(current_text, self.num_transformations)))
    #         sys.exit()
    #         return list(self.transformation.transform(current_text, self.num_transformations))
    #     return []

    # def get_goal_results(self, transformed_text_candidates):
    #     print ('transformed_text_candidates',transformed_text_candidates)
    #     sys.exit()
    #     """Evaluate the goal on the transformed text candidates."""
    #     results = [self.goal_function.get_result(text) for text in transformed_text_candidates]
    #     search_over = any(result.goal_status == GoalFunctionResultStatus.SUCCEEDED for result in results)
    #     return results, search_over

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["num_transformations"]

    def __repr__(self):
        return default_class_repr(self)

#sspattack
from textattack.shared import  WordEmbedding
import nltk
from textattack.constraints.semantics.sentence_encoders.sentence_encoder import SentenceEncoder
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
# class SSPAttackSearch(SearchMethod):
#     def __init__(self, max_iterations=100,**kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)
#         self.max_iterations = max_iterations
#         self.embedding = WordEmbedding.counterfitted_GLOVE_embedding()
#         self.sentence_encoder_use = UniversalSentenceEncoder()

#     def perform_search(self, initial_result):
#         attacked_text = initial_result.attacked_text

#         print ('goal_function',self.goal_function)
        
#         # Step 1: Initialization
#         # perturbed_text = [self.random_initialization(attacked_text)]
#         perturbed_text = [self.random_initialization(attacked_text) for i in range(20)]

#         results, search_over = self.get_goal_results(perturbed_text)
        
#         results_success = [result for result in results if result.ground_truth_output!=result.output] 

#         results_success=[]
#         perturbed_text_success = []
#         for i,result in enumerate(results):
#             if result.ground_truth_output!=result.output:
#                 results_success.append(result)
#                 perturbed_text_success.append(perturbed_text[i])

#         if len(results_success) == 0:
#             return results[0] # return a random result that wasnt perturbed to show it failed.

#         perturbed_text = perturbed_text_success[0]
#         results = results_success[0]
 

#         # check that sample is adversarial

#         print ('attacked_text',attacked_text)
#         print ('perturbed_text',perturbed_text)
        
#         # Main iteration loop
#         for _ in range(self.max_iterations):
#             # Step 2: Remove Unnecessary Replacement Words
            
#             perturbed_text = self.remove_unnecessary_words(perturbed_text, attacked_text)
#             print ('perturned+text',perturbed_text)
#             # Step 3: Push Substitution Words towards Original Words
#             perturbed_text = self.push_words_towards_original(perturbed_text, attacked_text)
#             print ('perturned+text2',perturbed_text) 
#             # Check if attack is successful
#             results, search_over = self.get_goal_results([perturbed_text])
            
            
#             if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
#                 return results[0]
        
#         return initial_result

#     def random_initialization(self, text):
#         words = text.words
#         tmp_text = text
#         size_text = len(text.words)
#         start_i = 0
#         while start_i < size_text:
#             # print ('start tmp text',tmp_text)
#             words = tmp_text.words
#             pos_tags = nltk.pos_tag(words)   
#             # print ('pos_tags',pos_tags)
#             if pos_tags[start_i][1].startswith(('VB', 'NN', 'JJ', 'RB')): 
#                 # print ('pos_tags[start_i][1]',pos_tags[start_i][1])
#                 replaced_with_synonyms = self.get_transformations(tmp_text, original_text=tmp_text,indices_to_modify=[start_i])
#                 # print ('replaced_with_synonyms',replaced_with_synonyms)
#                 if replaced_with_synonyms:
#                     tmp_text = random.choice(replaced_with_synonyms)
#                 else:
#                     pass
                
#             start_i+=1
            
#         adv_text = tmp_text
#         return adv_text

 
#     def remove_unnecessary_words(self, perturbed_text, original_text, check_skip=False): # can update to include semantic sim optim
        
#         self.sentence_encoder_use()

#         for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
#             if perturbed_word != original_word:
#                 temp_text = perturbed_text.replace_word_at_index(i, original_word) 
#                 model_outputs = self.goal_function._call_model([temp_text])
#                 print ('original_text',original_text)
#                 print ('temp_text',temp_text)
#                 sim_remove_unnecessary = self.sentence_encoder_use(original_text,temp_text)
#                 print ('sim_remove_unnecessary',sim_remove_unnecessary)
#                 sys.exit()

#                 # print ('model_outputs',model_outputs,GoalFunctionResultStatus.SUCCEEDED)
#                 current_goal_status = self.goal_function._get_goal_status(
#                     model_outputs[0], temp_text, check_skip=check_skip
#                 ) 
#                 if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
#                     perturbed_text = temp_text
                     
#         return perturbed_text

#     def push_words_towards_original(self, perturbed_text, original_text,check_skip=False):
#         for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
#             if perturbed_word != original_word: 

#                 sentence_replaced = self.get_transformations(original_text,original_text=original_text,indices_to_modify=[i])

#                 synonyms = []

#                 for s in sentence_replaced:
#                     s_words = s.words
#                     synonyms.append(s_words[i])
 
#                 embedding = WordEmbedding.counterfitted_GLOVE_embedding() 
                
#                 # synonyms.sort(key=lambda x: self.embedding.get_cos_sim(x, original_word), reverse=True) 

#                 synonyms_with_scores_and_transforms = []

#                 # Step 1: Compute the cosine similarity scores
#                 for synonym,sentence in zip(synonyms,sentence_replaced):
#                     # print ('synonym,sentence',synonym,sentence)
#                     cos_sim = self.embedding.get_cos_sim(synonym, original_word)
#                     synonyms_with_scores_and_transforms.append((synonym, cos_sim,sentence))

#                 # print ('synonyms_with_scores_and_transforms',synonyms_with_scores_and_transforms)
#                 # sys.exit()
#                 synonyms_with_scores_and_transforms.sort(key=lambda item: item[1], reverse=True)

#                 # get top K synonyms based on eucledian distance

#                 for s, (synonym, score, transformation) in enumerate(synonyms_with_scores_and_transforms):
#                     temp_text2 = perturbed_text.replace_word_at_index(i, synonym)
#                     temp_text = transformation# sentence_replaced[s_n]
#                     # print ('temp_text2',temp_text2,synonym)
#                     # print ('temp_text',temp_text, synonym)

#                     model_outputs = self.goal_function._call_model([temp_text]) 
#                     current_goal_status = self.goal_function._get_goal_status(
#                         model_outputs[0], temp_text, check_skip=check_skip
#                     ) 
#                     if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
#                         perturbed_text = temp_text
#                         break 
                        
#         return perturbed_text

#     def get_transformations(self, text, index):
#         return self.transformation(text, index)

#     def get_similarity(self, word1, word2):
#         return self.transformation.get_cosine_similarity(word1, word2)
    
#     @property
#     def is_black_box(self):
#         return True

import copy
class SSPAttackSearch(SearchMethod):
    def __init__(self, max_iterations=2,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.max_iterations = max_iterations
        self.embedding = WordEmbedding.counterfitted_GLOVE_embedding() 
        self.sentence_encoder_use = UniversalSentenceEncoder(window_size=15)
        
        self.number_of_queries = 0

    def perform_search(self, initial_result):
        self.number_of_queries = 0
        attacked_text = initial_result.attacked_text

        print ('goal_function',self.goal_function)
        
        # Step 1: Initialization
        number_samples = 2
        self.number_of_queries+=number_samples + 1 # checking the original sample if it's correct, then num samples perturbations to find adv

        perturbed_text = [self.random_initialization(attacked_text) for i in range(number_samples)]
        
        results, search_over = self.get_goal_results(perturbed_text)
        
         
        
        results_success = [result for result in results if result.ground_truth_output!=result.output] 

        results_success=[]
        perturbed_text_success = []
        for i,result in enumerate(results):
            if result.ground_truth_output!=result.output:
                results_success.append(result)
                perturbed_text_success.append(perturbed_text[i])


        print ('returnign failed?')
        if len(results_success) == 0:
            final_result = results[0]
            # final_result.num_queries = self.number_of_queries
            self.goal_function.num_queries = self.number_of_queries
            return final_result # return a random result that wasnt perturbed to show it failed.

        perturbed_text = perturbed_text_success[0]
        results = results_success[0]
 

        # check that sample is adversarial

        print ('attacked_text',attacked_text)
        print ('perturbed_text',perturbed_text)

                 

        
        
        # Main iteration loop
        for _ in range(self.max_iterations):
            # Step 2: Remove Unnecessary Replacement Words
            
            perturbed_text = self.remove_unnecessary_words(perturbed_text, attacked_text)
            print ('perturned+text',perturbed_text)

            # if attacked_text.words == perturbed_text.words:
            #     print ('should we skipp?')
            #     sys.exit() 

            # Step 3: Push Substitution Words towards Original Words
            perturbed_text = self.push_words_towards_original(perturbed_text, attacked_text)
            print ('perturned+text2',perturbed_text) 
            # if attacked_text == perturbed_text:
            #     print ('should we skipp 2?')
            #     sys.exit() 
            # Check if attack is successful
            results, search_over = self.get_goal_results([perturbed_text])
            # perturbed_result = initial_result.goal_function.call_model([perturbed_text])[0]
            # print ('results',results)

            # add semantic sim filter

            # this checks the generated test against the actual final use constraint
            
              

            final_result = results[0]

             

            if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                print ('attacked_text.text, final_result.attacked_text.text')
                print ('attk text',attacked_text.text)
                print ('final sre',final_result.attacked_text.text)
                sim_final_original, sim_final_pert = self.use_constraint.encode([attacked_text.text, final_result.attacked_text.text])

                if not isinstance(sim_final_original, torch.Tensor):
                    sim_final_original = torch.tensor(sim_final_original)

                if not isinstance(sim_final_pert, torch.Tensor):
                    sim_final_pert = torch.tensor(sim_final_pert)

                sim_score = self.sentence_encoder_use.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                print ('sim_score',sim_score, (1 - (args.similarity_threshold) / math.pi))
                if sim_score <  (1 - (args.similarity_threshold) / math.pi):
                    continue

                # final_result.num_queries = self.number_of_queries
                self.goal_function.num_queries = self.number_of_queries
                print ('final_result.num_queries',final_result.num_queries)
                # sys.exit()
                print ('final_result',final_result.attacked_text)
                print ('final_result.attacked_text.attack_attrs',final_result.attacked_text.attack_attrs)
                print ('final_result.attacked_text.attack_attrs[original_index_map]',final_result.attacked_text.attack_attrs['original_index_map'])
                # print ('final_result',final_result.original_text)
                # print ('final_result',final_result.perturbed_text)
                # print ('final_result.perturbed_result.attack_attrs',final_result.perturbed_result.attack_attrs)
                # if len(final_result.attacked_text.attack_attrs['newly_modified_indices']) == 0:
                #     final_result.attacked_text.attack_attrs['newly_modified_indices'] = {0}
                # if len(final_result.attacked_text.attack_attrs['modified_indices']) == 0:
                #     # final_result.attacked_text.attack_attrs['modified_indices'] = {0}
                #     return initial_result

                return final_result
        print ('just aviod everything')
        return initial_result

    def random_initialization(self, text):
        words = text.words
        tmp_text = text
        size_text = len(text.words)
        start_i = 0
        while start_i < size_text:
            # print ('start tmp text',tmp_text)
            words = tmp_text.words
            pos_tags = nltk.pos_tag(words)   
            # print ('pos_tags',pos_tags)
            if pos_tags[start_i][1].startswith(('VB', 'NN', 'JJ', 'RB')): 
                # print ('pos_tags[start_i][1]',pos_tags[start_i][1])
                replaced_with_synonyms = self.get_transformations(tmp_text, original_text=tmp_text,indices_to_modify=[start_i])
                # print ('replaced_with_synonyms',replaced_with_synonyms)
                if replaced_with_synonyms:
                    tmp_text = random.choice(replaced_with_synonyms)
                else:
                    pass
                
            start_i+=1
            
        adv_text = tmp_text
        return adv_text

 

    def remove_unnecessary_words(self, perturbed_text, original_text, check_skip=False):
        # Step 1: Identify words to replace back
        candidate_set = []
        word_importance_scores = [] 
        # print ('original_text',original_text)
        # print ('perturbed_text',perturbed_text)
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                # Replace perturbed_word with original_word
                temp_text = perturbed_text.replace_word_at_index(i, original_word)

                # Step 2: Check if still adversarial and calculate semantic similarity
                model_outputs = self.goal_function._call_model([temp_text])
                current_goal_status = self.goal_function._get_goal_status(
                    model_outputs[0], temp_text, check_skip=check_skip
                )
                self.number_of_queries+=1
                # print ('temp_text',temp_text,i,current_goal_status,GoalFunctionResultStatus.SUCCEEDED)
                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    candidate_set.append((i, temp_text))
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_text.text, temp_text.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    word_importance_scores.append((i, sim_score))

        # Step 3: Sort word importance scores in descending order and restore original words
        word_importance_scores.sort(key=lambda x: x[1], reverse=True)
        print ('attack_attrs ',perturbed_text.attack_attrs,perturbed_text  ) 
        print ('replace indexs',word_importance_scores)
        for idx, _ in word_importance_scores:
            temp_text2 = perturbed_text.replace_word_at_index(idx, original_text.words[idx])
            temp_text2.attack_attrs['modified_indices'].remove(idx)
            print ('temp_text2_word_imp',idx,temp_text2.attack_attrs,temp_text2)
            # print ('original_index_map',temp_text2.attack_attrs.original_index_map)
            
            model_outputs = self.goal_function._call_model([temp_text2])
            current_goal_status = self.goal_function._get_goal_status(
                model_outputs[0], temp_text2, check_skip=check_skip
            )
            self.number_of_queries+=1

            # print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)

 
            if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                # If perturbed_text is no longer adversarial, revert the last change
                perturbed_text = temp_text2
                # perturbed_text = perturbed_text.replace_word_at_index(idx, perturbed_text.words[idx])
            else:
                break
        # print ('original_text',original_text)
        # print ('perturbed_text',perturbed_text) 
        return perturbed_text

    def get_vector(self, embedding, word):
        if isinstance(word, str):
            if word in embedding._word2index:
                word_index = embedding._word2index[word]
            else:
                return None  # Word not found in the dictionary
        else:
            word_index = word

        vector = embedding.embedding_matrix[word_index]
        return torch.tensor(vector).to(textattack.shared.utils.device)

    def push_words_towards_original(self, perturbed_text, original_text, check_skip=False):
        # Step 1: Calculate Euclidean distances and sampling probabilities
        distances = []
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                # Using the get_vector function
                perturbed_vec = self.get_vector(self.embedding, perturbed_word)
                if perturbed_vec is None:
                    continue  # Skip to the next word
                original_vec = self.get_vector(self.embedding, original_word)
                if original_vec is None:
                    continue  # Skip to the next word
                distance = np.linalg.norm(perturbed_vec.cpu().numpy() - original_vec.cpu().numpy())
                distances.append((i, distance))

        if not distances:
            return perturbed_text

        # Normalize distances to get probabilities
        distances.sort(key=lambda x: x[1])
        indices, dist_values = zip(*distances)
        exp_dist_values = np.exp(dist_values)
        probabilities = exp_dist_values / np.sum(exp_dist_values)
        print ('probabilities',probabilities)

        # temp_perturbed_text = copy.deepcopy(perturbed_text)
        
        # Step 2: Iterate with sampling based on the probabilities 
        while len(indices) > 0:
            i = np.random.choice(indices, p=probabilities)
            print ('indices',indices,i)
            perturbed_word = perturbed_text.words[i]
            original_word = original_text.words[i]

            sentence_replaced = self.get_transformations(original_text, original_text=original_text, indices_to_modify=[i])
            synonyms = [s.words[i] for s in sentence_replaced]


            # Get top k synonyms
            k = 10  # Number of synonyms to sample
            top_k_synonyms_indexes  = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=k)
            top_k_synonyms = [self.embedding._index2word[index] for index in top_k_synonyms_indexes]

            # Find the best anchor synonym with the highest semantic similarity
            max_similarity = -float('inf')
            w_bar = None
            temp_text_bar = None
            filtered_synonyms = None
            print ('top_k_synonyms',top_k_synonyms)
            # temp_text2 = copy.deepcopy(perturbed_text)
            for synonym in top_k_synonyms:
                if perturbed_word == synonym:
                    continue # skip swapping the same word
                print ('synonym',i,synonym)
                # temp_text2 = copy.deepcopy(perturbed_text)
                temp_text2 = perturbed_text.replace_word_at_index(i, synonym)

                # Check if the substitution still results in an adversarial example
                model_outputs = self.goal_function._call_model([temp_text2])
                current_goal_status = self.goal_function._get_goal_status(
                    model_outputs[0], temp_text2, check_skip=check_skip
                )
                self.number_of_queries+=1

                print ('temp_text2_top_k_syn',i,synonym,temp_text2.attack_attrs,temp_text2,current_goal_status , GoalFunctionResultStatus.SUCCEEDED)

                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    # Compute semantic similarity at the word level
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0)).item()
                    print ('sim scores push towards orgin')
                    if sim_score > max_similarity:
                        max_similarity = sim_score
                        w_bar = synonym
                        temp_text_bar = temp_text2

            
            if w_bar:# is None:
                  # Skip this index if no suitable anchor synonym is found
            
                number_entries = len(self.embedding.nn_matrix[self.embedding._word2index[original_word]] )
                print ('num entries',number_entries)
                all_synonyms = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=number_entries)
                all_synonyms = [self.embedding._index2word[index] for index in all_synonyms]
            
                print ('all_synonyms',all_synonyms)
                filtered_synonyms = []
                for synonym in all_synonyms:
                    if perturbed_word == synonym or w_bar == synonym  :
                        continue # skip swapping/checking the same word and the anchor word
                    # Compute semantic similarity with w_bar and original_word
                    sim_w_bar, sim_synonym = self.sentence_encoder_use.encode([w_bar, synonym])
                    sim_org, sim_synonym_org = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_w_bar, torch.Tensor):
                        sim_w_bar = torch.tensor(sim_w_bar)
                    if not isinstance(sim_synonym, torch.Tensor):
                        sim_synonym = torch.tensor(sim_synonym)
                    if not isinstance(sim_org, torch.Tensor):
                        sim_org = torch.tensor(sim_org)
                    if not isinstance(sim_synonym_org, torch.Tensor):
                        sim_synonym_org = torch.tensor(sim_synonym_org)

                    sim_score_w_bar = self.sentence_encoder_use.sim_metric(sim_w_bar.unsqueeze(0), sim_synonym.unsqueeze(0)).item()
                    sim_score_org = self.sentence_encoder_use.sim_metric(sim_org.unsqueeze(0), sim_synonym_org.unsqueeze(0)).item()

                    if sim_score_w_bar > sim_score_org:
                        filtered_synonyms.append((sim_score_w_bar, synonym))

            if  filtered_synonyms:
                # continue  # Skip this index if no suitable synonym is found

                # Sort the filtered synonyms by their semantic similarity score in descending order
                filtered_synonyms.sort(key=lambda item: item[0], reverse=True)
                print ('filtered_synonyms',filtered_synonyms)
                
                

                print ('perturbed text',perturbed_text.attack_attrs,perturbed_text)
                for _, synonym in filtered_synonyms:
                    temp_text2 = perturbed_text.replace_word_at_index(i, synonym) 
                    # temp_text2.attack_attrs['modified_indices'].remove(i)
                    print ('temp_text2_filtered_syn',i,temp_text2.attack_attrs,temp_text2)
                    # Check if the substitution still results in an adversarial example
                    model_outputs = self.goal_function._call_model([temp_text2]) 
                    current_goal_status = self.goal_function._get_goal_status(
                        model_outputs[0], temp_text2, check_skip=check_skip
                    )
                    self.number_of_queries+=1
                    print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)
                    if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        perturbed_text = temp_text2
                        break
                    
            
            print ('perturbed_text',perturbed_text)  
            idx = indices.index(i)
            indices = indices[:idx] + indices[idx + 1:] 
            print ('indices2',indices,idx)
            
            probabilities = np.delete(probabilities, idx)
            probabilities /= np.sum(probabilities)   
        # sys.exit()
        return perturbed_text 

    def get_transformations(self, text, index):
        return self.transformation(text, index)

    def get_similarity(self, word1, word2):
        return self.transformation.get_cosine_similarity(word1, word2)
    
    @property
    def is_black_box(self):
        return True



from collections import defaultdict
class TextHoaxer(SearchMethod):
    def __init__(self, max_iterations=2,**kwargs): 
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.max_iterations = max_iterations
        self.embedding = WordEmbedding.counterfitted_GLOVE_embedding() 
        self.sentence_encoder_use = UniversalSentenceEncoder(window_size=15)
        self.download_synonym_file()

        

        self.number_of_queries = 0

    def download_synonym_file(self):
        import gdown
        import pickle
        # Create the full directory path if it doesn't exist
        full_path = os.path.join(self.cache_dir, 'texthoaxer')
        os.makedirs(full_path, exist_ok=True)

        # Define the URL or Google Drive ID
        file_id = '1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx'
        url = f'https://drive.google.com/uc?id={file_id}'

        # Define the file path where the file will be saved
        output_path = os.path.join(full_path, 'mat.txt')

        # Check if the file already exists
        if os.path.exists(output_path):
            print(f"File already exists at: {output_path}")
        else:
            # Download the file
            print(f"Downloading file to: {output_path}")
            gdown.download(url, output_path, quiet=False)
            print("Download complete.")
        
        with open(output_path, "rb") as fp:
            self.sim_lis = pickle.load(fp)
    def soft_threshold(self, alpha, beta):
        if beta > alpha:
            return beta - alpha
        elif beta < -alpha:
            return beta + alpha
        else:
            return 0
    def  get_pos(self,sent, tagset='universal'):
        '''
        :param sent: list of word strings
        tagset: {'universal', 'default'}
        :return: list of pos tags.
        Universal (Coarse) Pos tags has  12 categories
            - NOUN (nouns)
            - VERB (verbs)
            - ADJ (adjectives)
            - ADV (adverbs)
            - PRON (pronouns)
            - DET (determiners and articles)
            - ADP (prepositions and postpositions)
            - NUM (numerals)
            - CONJ (conjunctions)
            - PRT (particles)
            - . (punctuation marks)
            - X (a catch-all for other categories such as abbreviations or foreign words)
        '''
        if tagset == 'default':
            word_n_pos_list = nltk.pos_tag(sent)
        elif tagset == 'universal':
            word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
        print ('tagset',tagset)
        _, pos_list = zip(*word_n_pos_list)
        return pos_list

    def perform_search(self, initial_result):
        self.number_of_queries = 0
        attacked_text = initial_result.attacked_text

        print ('goal_function',self.goal_function)
        
        # Step 1: Initialization
        number_samples = 2
        self.number_of_queries+=number_samples + 1 # checking the original sample if it's correct, then num samples perturbations to find adv
        self.goal_function.num_queries = self.number_of_queries
        perturbed_text = [self.random_initialization(attacked_text) for i in range(number_samples)]
        
        results, search_over = self.get_goal_results(perturbed_text)
        
         
        
        # results_success = [result for result in results if result.ground_truth_output!=result.output] 

        results_success=[]
        perturbed_text_success = []
        for i,result in enumerate(results):
            if result.ground_truth_output!=result.output:
                results_success.append(result)
                perturbed_text_success.append(perturbed_text[i])


        print ('returnign failed?')
        if len(results_success) == 0:
            flag = 0
        else:
            flag = 1

            #     final_result = results[0]
                
            #     self.goal_function.num_queries = self.number_of_queries
            #     self.goal_function.model.reset_inference_steps()
            #     return final_result # return a random result that wasnt perturbed to show it failed.

            perturbed_text = perturbed_text_success[0]
            # results = results_success[0]
    

            # check that sample is adversarial

            print ('attacked_text',attacked_text)
            print ('perturbed_text',perturbed_text)

            random_text = perturbed_text.words

        # get_vector(self, self.embedding, word): # word can either be a str or the index eqivalant of embed_content[word_idx_dict[word] ]

        text_ls = attacked_text.words
        true_label = initial_result.ground_truth_output 
        orig_label = initial_result.output

        word_idx_dict = self.embedding._word2index
        embed_content = self.embedding
        idx2word = self.embedding._index2word
        word2idx = self.embedding._word2index
        criteria = self
        top_k_words = self.max_iter_i
        cos_sim = self.sim_lis
        budget =  self.query_budget - 1 # -1 because we have qrs > buget, but if we do a model call exacly on query results will be empty
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        # if len_text < sim_score_window:
        #     sim_score_threshold = 0.1
        # half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))
        print ('words_perturb',words_perturb)

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]
        print ('words_perturb',words_perturb) 
        words_perturb_idx= []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                # print ('[float(num) for num in embed_content[ word_idx_dict[word] ]', embed_content[ word_idx_dict[word]] )
                words_perturb_embed.append([float(num) for num in embed_content[ word_idx_dict[word] ]])
                # print ('words_perturb_embed',words_perturb_embed) 

        words_perturb_embed_matrix = np.asarray(words_perturb_embed)


        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        # print ('synonym_words',synonym_words)
        print ('synonyms_all',synonyms_all)
        ###################### erlier code ##########################
        # qrs = 1 # qrs start at 1 because we already had to use 1 query to detect if sample gets classified correctly
        # num_changed = 0
        # flag = 0
        # th = 0
        # while qrs < len(text_ls):
        #     print ('qrs1',qrs)
        #     random_text = text_ls[:]
        #     for i in range(len(synonyms_all)):
        #         idx = synonyms_all[i][0]
        #         syn = synonyms_all[i][1]
        #         random_text[idx] = random.choice(syn)
        #         if i >= th:
        #             break
        #     print ('random_text 1',random_text)
        #     print ('attacked_text 1',attacked_text)
        #     print ('text_ls 1',text_ls)
        #     # random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
        #     # model_outputs = self.goal_function._call_model([random_text_joint])
        #     # current_goal_status = self.goal_function._get_goal_status(
        #     #     model_outputs[0], random_text_joint, check_skip=False
        #     # )
        #     # self.number_of_queries+=1
            
        #     random_text_joint = attacked_text.generate_new_attacked_text(random_text)
        #     results_adv_initial, search_over = self.get_goal_results([random_text_joint])
        #     # pr = get_attack_result([random_text], predictor, orig_label, batch_size)
        #     if search_over: 
        #         self.goal_function.model.reset_inference_steps()
        #         return initial_result

        #     qrs+=1
        #     self.number_of_queries+=1
        #     self.goal_function.num_queries = self.number_of_queries
        #     th +=1
        #     if th > len_text:
        #         break
        #     # if np.sum(pr)>0:
        #     # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #     # print ('results first',results)
        #     # if len(results) == 0: 
        #     #     print ('returning initial result 1', initial_result)
        #     #     return initial_result

        #     if results_adv_initial[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #         flag = 1
        #         break
        # old_qrs = qrs
        # print ('old_qrs',old_qrs)
        ######################################################
        qrs = number_samples + 1
        old_qrs = qrs  
        print ('old_qrs',old_qrs)

        # while qrs < old_qrs + 2500 and flag == 0:

        # we remove this part because althou if tries all possible queries to find a solution, to make sure we stay in buget we 
        # focus only on examples that we found an adversarial example quickly then optimize it's semantic similarity.
        # while qrs < budget and flag == 0:
        #     print ('qrs2',qrs)
        #     random_text = text_ls[:]
        #     for j in range(len(synonyms_all)):
        #         idx = synonyms_all[j][0]
        #         syn = synonyms_all[j][1]
        #         random_text[idx] = random.choice(syn)
        #         if j >= len_text:
        #             break
        #     # pr = get_attack_result([random_text], predictor, orig_label, batch_size) 
        #     print ('random_text 2',random_text)
        #     print ('attacked_text 2',attacked_text)
        #     print ('text_ls 2',text_ls)
        #     # random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
        #     # model_outputs = self.goal_function._call_model([random_text_joint])
        #     # current_goal_status = self.goal_function._get_goal_status(
        #     #     model_outputs[0], random_text_joint, check_skip=False
        #     # )
        #     # self.number_of_queries+=1

        #     random_text_joint = attacked_text.generate_new_attacked_text(random_text)
        #     results_adv_initial, search_over = self.get_goal_results([random_text_joint])
        #     if search_over: 
        #         self.goal_function.model.reset_inference_steps()
        #         return initial_result
        #     qrs+=1
        #     self.number_of_queries+=1
        #     self.goal_function.num_queries = self.number_of_queries
        #     # if np.sum(pr)>0:
        #     # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #     print ('results_adv_initial second',results_adv_initial)


        #     # print ('returning failed result because flag==0')
        #     # results, search_over = self.get_goal_results([attacked_text])
        #     # return results[0]

        #     # if len(results) == 0: 
        #     #     print ('returning initial result 2', initial_result)
        #     #     return initial_result

        #     if results_adv_initial[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #         flag = 1
        #         break
            
            
        print ('flag',flag) 
        if flag == 1:
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1
             

            print ('original_random_text',random_text)
            while True:
                choices = []

                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        # print ('text_ls, [new_text], -1, sim_score_window, sim_predictor',text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        
                        


                        new_text_joint = attacked_text.generate_new_attacked_text(new_text) 


                        print ('attacked_text',attacked_text)
                        print ('new_text_joint',new_text_joint)
                        if attacked_text.text == new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        

                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, new_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        semantic_sims = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        
                        print ('semantic_sims1',semantic_sims)
                        # semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        # model_outputs = self.goal_function._call_model([new_text_joint])
                        # current_goal_status = self.goal_function._get_goal_status(
                        #     model_outputs[0], new_text_joint, check_skip=False
                        # )
                        # self.number_of_queries+=1
                        # random_text_joint = attacked_text.generate_new_attacked_text(new_text_joint)
                        results, search_over = self.get_goal_results([new_text_joint])
                        if search_over: 
                            self.goal_function.model.reset_inference_steps()
                            return initial_result
                        print ('qrs3',qrs)
                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            choices.append((i,semantic_sims[0]))
 
                        # qrs+=1
                        # pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        # if np.sum(pr) > 0:
                        #     choices.append((i,semantic_sims[0]))

                print ('choices', choices) 
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]

                        # new_text_joint = attacked_text.generate_new_attacked_text(new_text) 
                        # model_outputs = self.goal_function._call_model([new_text_joint])
                        # current_goal_status = self.goal_function._get_goal_status(
                        #     model_outputs[0], new_text_joint, check_skip=False
                        # )
                        # self.number_of_queries+=1

                        new_text_joint = attacked_text.generate_new_attacked_text(new_text)
                        if attacked_text.text == new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue

                        results, search_over = self.get_goal_results([new_text_joint])
                        if search_over: 
                            self.goal_function.model.reset_inference_steps()
                            return initial_result

                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        # if current_goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            break
                        random_text[choices[i][0]] = text_ls[choices[i][0]]

                        # pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        # qrs+=1
                        # if pr[0] == 0:
                        #     break
                        # random_text[choices[i][0]] = text_ls[choices[i][0]]

                if len(choices) == 0:
                    break
            
            print ('after choices random_text',random_text) 
            changed_indices = [] 
            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed_indices.append(i)
                    num_changed+=1
            print(str(num_changed)+" "+str(qrs))

            new_random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
            print ('attacked_text',attacked_text)
            print ('new_random_text_joint',new_random_text_joint)
            
            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, new_random_text_joint.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            random_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            random_sim = random_sim.item()
            print ('random_sim1',random_sim)


            # random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]

            print ('qrs budget 1',qrs)
            if qrs > budget:
                # return fail 
                random_text_qrs_joint = attacked_text.generate_new_attacked_text(random_text) 

                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, random_text_qrs_joint.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                failed_sem_sim = failed_sem_sim.item()
                print ('failed_sem_sim0',failed_sem_sim)

                if failed_sem_sim <  (1 - (args.similarity_threshold) / math.pi):
                    print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                    # results_inner, search_over = self.get_goal_results([attacked_text])
                    # return results_inner[0]
                    self.goal_function.model.reset_inference_steps()
                    return initial_result
                else:
                    print ('out of queries, random_text_qrs_joint',random_text_qrs_joint)
                    results_inner, search_over = self.get_goal_results([random_text_qrs_joint])
                    self.goal_function.model.reset_inference_steps()
                    return results_inner[0]
                
                

                 
                # return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                #     orig_label, torch.argmax(predictor([random_text])), qrs, random_sim, random_sim

            if num_changed == 1:
                random_text_num_changed_joint = attacked_text.generate_new_attacked_text(random_text)  
                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, random_text_num_changed_joint.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                failed_sem_sim = failed_sem_sim.item()
                print ('failed_sem_sim1',failed_sem_sim)

                if failed_sem_sim <  (1 - (args.similarity_threshold) / math.pi):
                    print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                    # results_inner, search_over = self.get_goal_results([attacked_text])
                    # return results_inner[0]
                    self.goal_function.model.reset_inference_steps()
                    return initial_result
                else:
                    print ('out of queries, random_text_num_changed_joint',random_text_num_changed_joint)
                    results_inner, search_over = self.get_goal_results([random_text_num_changed_joint])
                    self.goal_function.model.reset_inference_steps()
                    return results_inner[0]
                    
                # return failed
                # return ' '.join(random_text), 1, 1, \
                #     orig_label, torch.argmax(predictor([random_text])), qrs, random_sim, random_sim

            

            best_attack = random_text
            # best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)
            best_sim = random_sim



            gamma = 0.3*np.ones([words_perturb_embed_matrix.shape[0], 1])
            l1 = 0.1
            l2_lambda = 0.1
            for t in range(100):

                theta_old_text = best_attack
                sim_old= best_sim 
                old_adv_embed = []
                for idx in words_perturb_doc_idx:
                    # old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]].strip().split()[1:]])
                    old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]]])
                old_adv_embed_matrix = np.asarray(old_adv_embed)

                theta_old = old_adv_embed_matrix-words_perturb_embed_matrix
               
                u_vec = np.random.normal(loc=0.0, scale=1,size=theta_old.shape)
                theta_old_neighbor = theta_old+0.5*u_vec
                print ('theta_old_neighbor',theta_old_neighbor)
                # Check if theta_old_neighbor is a 2D array
                if theta_old_neighbor.ndim != 2:
                    print('theta_old_neighbor not a 2D array. Skipping this iteration.')
                    continue
                theta_perturb_dist = np.sum((theta_old_neighbor)**2, axis=1)
                nonzero_ele = np.nonzero(np.linalg.norm(theta_old,axis = -1))[0].tolist()
                perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])

                theta_old_neighbor_text = text_ls[:]
                for perturb_idx in range(len(nonzero_ele)):
                    perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                    word_dict_idx = words_perturb_idx[perturb_word_idx]
                    
                    perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_old_neighbor[perturb_word_idx]
                    syn_feat_set = []
                    for syn in synonyms_all[perturb_word_idx][1]:
                        # syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                        syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                        syn_feat_set.append(syn_feat)

                    perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                    perturb_syn_order = np.argsort(perturb_syn_dist)
                    replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                    
                    theta_old_neighbor_text[synonyms_all[perturb_word_idx][0]] = replacement

                    theta_old_neighbor_text_joint = attacked_text.generate_new_attacked_text(theta_old_neighbor_text)
                    print ('attacked_text',attacked_text)
                    print ('theta_old_neighbor_text_joint',theta_old_neighbor_text_joint)
                    if attacked_text.text == theta_old_neighbor_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                        continue 
                    # model_outputs = self.goal_function._call_model([theta_old_neighbor_text_joint])
                    # current_goal_status = self.goal_function._get_goal_status(
                    #     model_outputs[0], theta_old_neighbor_text_joint, check_skip=False
                    # )
                    # self.number_of_queries+=1
                    results, search_over = self.get_goal_results([theta_old_neighbor_text_joint])
                    if search_over: 
                        self.goal_function.model.reset_inference_steps()
                        return initial_result
                    print ('qrs budget 2',qrs)
                    qrs+=1
                    self.number_of_queries+=1
                    self.goal_function.num_queries = self.number_of_queries
                    if qrs > budget:
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_old_neighbor_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        failed_sem_sim = failed_sem_sim.item()
                        print ('failed_sem_sim2',failed_sem_sim)

                        if failed_sem_sim <  (1 - (args.similarity_threshold) / math.pi): 
                            print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                            # results_inner, search_over = self.get_goal_results([attacked_text])
                            # return results_inner[0]
                            self.goal_function.model.reset_inference_steps()
                            return initial_result
                        else:
                            print ('out of queries, theta_old_neighbor_text_joint',theta_old_neighbor_text_joint)
                            # results_inner, search_over = self.get_goal_results([theta_old_neighbor_text_joint])
                            # return results_inner[0]
                            self.goal_function.model.reset_inference_steps()
                            return results[0]
                        
                        
                        # return results[0]   
                        # return ' '.join(best_attack), max_changes, len(changed_indices), \
                        #     orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        break



                    # pr = get_attack_result([theta_old_neighbor_text], predictor, orig_label, batch_size)
                    # qrs+=1

                    # if qrs > budget:
                    #     sim = best_sim[0]
                    #     max_changes = 0
                    #     for i in range(len(text_ls)):
                    #         if text_ls[i]!=best_attack[i]:
                    #             max_changes+=1

                    #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                    #         orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

                    # if np.sum(pr)>0:
                    #     break

                # if np.sum(pr)>0:

                if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:

                     
                    theta_old_neighbor_text_joint = attacked_text.generate_new_attacked_text(theta_old_neighbor_text) 
                    

                    print ('attacked_text',attacked_text)
                    print ('theta_old_neighbor_text_joint',theta_old_neighbor_text_joint)
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_old_neighbor_text_joint.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    sim_new = sim_new.item()
                    print ('sim_new',sim_new)

                    # sim_new = calc_sim(text_ls, [theta_old_neighbor_text], -1, sim_score_window, sim_predictor)
                    derivative = (sim_old-sim_new)/0.5

                    g_hat = derivative*u_vec

                    theta_new = theta_old-0.3*(g_hat+2*l2_lambda*theta_old)

                    if sim_new > sim_old:
                        best_attack = theta_old_neighbor_text
                        best_sim = sim_new

                    theta_perturb_dist = np.sum((theta_new)**2, axis=1)
                    nonzero_ele = np.nonzero(np.linalg.norm(theta_new,axis = -1))[0].tolist()
                    perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])

                    theta_new_text = text_ls[:]
                    for perturb_idx in range(len(nonzero_ele)):
                        perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                        word_dict_idx = words_perturb_idx[perturb_word_idx]
                        
                        perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_new[perturb_word_idx]
                        syn_feat_set = []
                        for syn in synonyms_all[perturb_word_idx][1]:
                            # syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                            syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                            syn_feat_set.append(syn_feat)

                        perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                        perturb_syn_order = np.argsort(perturb_syn_dist)
                        replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                        
                        theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement


                        theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text)
                        print ('attacked_text',attacked_text)
                        print ('theta_new_text_joint',theta_new_text_joint)
                        if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue  
                        results, search_over = self.get_goal_results([theta_new_text_joint])
                        if search_over: 
                            self.goal_function.model.reset_inference_steps()
                            return initial_result

                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        print ('qrs budget 3',qrs)
                        if qrs > budget:
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            failed_sem_sim = failed_sem_sim.item()
                            print ('failed_sem_sim3',failed_sem_sim)

                            if failed_sem_sim <  (1 - (args.similarity_threshold) / math.pi):
                                print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                                # results_inner, search_over = self.get_goal_results([attacked_text])
                                # return results_inner[0]
                                self.goal_function.model.reset_inference_steps()
                                return initial_result
                            else:
                                # print ('out of queries, theta_new_text_joint',theta_new_text_joint)
                                # results_inner, search_over = self.get_goal_results([theta_new_text_joint])
                                # return results_inner[0]
                                self.goal_function.model.reset_inference_steps()
                                return results[0]
                            # return results[0]
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            break


                        # pr = get_attack_result([theta_new_text], predictor, orig_label, batch_size)
                        # qrs+=1

                        # if qrs > budget:
                        #     sim = best_sim[0]
                        #     max_changes = 0
                        #     for i in range(len(text_ls)):
                        #         if text_ls[i]!=best_attack[i]:
                        #             max_changes+=1

                        #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                        #         orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

                        # if np.sum(pr)>0:
                        #     break
                    # if np.sum(pr)>0:
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                        print ('attacked_text',attacked_text)
                        print ('theta_new_text_joint',theta_new_text_joint)
                        if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        sim_theta_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        sim_theta_new = sim_theta_new.item()
                        print ('sim_theta_new',sim_theta_new)
                        # sim_theta_new = calc_sim(text_ls, [theta_new_text], -1, sim_score_window, sim_predictor)
                        if sim_theta_new > best_sim:
                            best_attack = theta_new_text
                            best_sim = sim_theta_new

                    
                    # if np.sum(pr)>0:
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        gamma_old_text = theta_new_text

                        gamma_old_text_joint = attacked_text.generate_new_attacked_text(gamma_old_text) 
                        print ('attacked_text',attacked_text)
                        print ('gamma_old_text_joint',gamma_old_text_joint)
                        if attacked_text.text == gamma_old_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, gamma_old_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        gamma_sim_full = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        gamma_sim_full = gamma_sim_full.item()
                        print ('gamma_sim_full',gamma_sim_full)



                        # gamma_sim_full = calc_sim(text_ls, [gamma_old_text], -1, sim_score_window, sim_predictor)
                        gamma_old_adv_embed = []
                        for idx in words_perturb_doc_idx:
                            # gamma_old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[gamma_old_text[idx]]].strip().split()[1:]])
                            gamma_old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[gamma_old_text[idx]]]])
                        gamma_old_adv_embed_matrix = np.asarray(gamma_old_adv_embed)

                        gamma_old_pert= gamma_old_adv_embed_matrix-words_perturb_embed_matrix
                        gamma_old_pert_divided =gamma_old_pert/gamma
                        perturb_gradient = []
                        for i in range(gamma.shape[0]):
                            idx = words_perturb_doc_idx[i]
                            replaceback_text = gamma_old_text[:]
                            replaceback_text[idx] = text_ls[idx]

                            replaceback_text_joint = attacked_text.generate_new_attacked_text(replaceback_text) 
                            print ('attacked_text',attacked_text)
                            print ('replaceback_text_joint',replaceback_text_joint)
                            if attacked_text.text == replaceback_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, replaceback_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            replaceback_sims = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            replaceback_sims = replaceback_sims.item()
                            print ('replaceback_sims',replaceback_sims)

                            # replaceback_sims = calc_sim(text_ls, [replaceback_text], -1, sim_score_window, sim_predictor)
                            gradient_2 = self.soft_threshold(l1,gamma[i][0])
                            gradient_1 = -((gamma_sim_full-replaceback_sims)/(gamma[i]+1e-4))[0]
                            gradient = gradient_1+gradient_2
                            gamma[i]=gamma[i]-0.05*gradient


                        theta_new = gamma_old_pert_divided * gamma
                        theta_perturb_dist = np.sum((theta_new)**2, axis=1)
                        nonzero_ele = np.nonzero(np.linalg.norm(theta_new,axis = -1))[0].tolist()
                        perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])
                        theta_new_text = text_ls[:]
                        for perturb_idx in range(len(nonzero_ele)):
                            perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                            word_dict_idx = words_perturb_idx[perturb_word_idx]
                            
                            perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_new[perturb_word_idx]
                            syn_feat_set = []
                            for syn in synonyms_all[perturb_word_idx][1]:
                                # syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                                syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                                syn_feat_set.append(syn_feat)

                            perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                            
                            theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement


                            theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text)
                            print ('attacked_text',attacked_text)
                            print ('theta_new_text_joint',theta_new_text_joint)
                            if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue 
                            # model_outputs = self.goal_function._call_model([theta_new_text_joint])
                            # current_goal_status = self.goal_function._get_goal_status(
                            #     model_outputs[0], theta_new_text_joint, check_skip=False
                            # )
                            # self.number_of_queries+=1
                            results, search_over = self.get_goal_results([theta_new_text_joint])
                            if search_over:
                                self.goal_function.model.reset_inference_steps() 
                                return initial_result

                            qrs+=1 
                            self.number_of_queries+=1
                            self.goal_function.num_queries = self.number_of_queries
                            print ('qrs budget 4',qrs)
                            if qrs > budget:
                                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                                failed_sem_sim = failed_sem_sim.item()
                                print ('failed_sem_sim4',failed_sem_sim)

                                if failed_sem_sim <  (1 - (args.similarity_threshold) / math.pi):
                                    print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                                    # results_inner, search_over = self.get_goal_results([attacked_text])
                                    # return results_inner[0]
                                    self.goal_function.model.reset_inference_steps()
                                    return initial_result
                                else:
                                    print ('out of queries, theta_new_text_joint',theta_new_text_joint)
                                    # results_inner, search_over = self.get_goal_results([theta_new_text_joint])
                                    # return results_inner[0]
                                    self.goal_function.model.reset_inference_steps()
                                    return results[0]
                                # sim = best_sim[0]
                                # max_changes = 0
                                # for i in range(len(text_ls)):
                                #     if text_ls[i]!=best_attack[i]:
                                #         max_changes+=1

                                # return results[0]
                                # return ' '.join(best_attack), max_changes, len(changed_indices), \
                                #     orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
                            # if current_goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            #     break
                            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                break

                            # pr = get_attack_result([theta_new_text], predictor, orig_label, batch_size)
                            
                            
                            # qrs+=1

                            # if qrs > budget:
                            #     sim = best_sim[0]
                            #     max_changes = 0
                            #     for i in range(len(text_ls)):
                            #         if text_ls[i]!=best_attack[i]:
                            #             max_changes+=1

                            #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                            #         orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

                            # if np.sum(pr)>0:
                            #     break

                    
                        # if np.sum(pr)>0:
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                            print ('attacked_text',attacked_text)
                            print ('theta_new_text_joint 2',theta_new_text_joint)
                            if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            sim_theta_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            sim_theta_new = sim_theta_new.item()
                            print ('sim_theta_new',sim_theta_new)
                            # sim_theta_new = calc_sim(text_ls, [theta_new_text], -1, sim_score_window, sim_predictor)
                            if sim_theta_new > best_sim:
                                best_attack = theta_new_text
                                best_sim = sim_theta_new


            best_attack_joint = attacked_text.generate_new_attacked_text(best_attack)


            print ('attacked_text',attacked_text)
            print ('best_attack_joint',best_attack_joint)
            
            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, best_attack_joint.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            best_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            best_sem_sim = best_sem_sim.item()
            print ('best_sem_sim',best_sem_sim)

            if best_sem_sim <  (1 - (args.similarity_threshold) / math.pi):
                print ('returning failed result because best_sem_sim too low', best_sem_sim)
                # results, search_over = self.get_goal_results([attacked_text])
                # return results[0]
                self.goal_function.model.reset_inference_steps()
                return initial_result



            print ('best sim meets threshod,best_attack_joint',best_attack_joint)
            results, search_over = self.get_goal_results([best_attack_joint])
            self.number_of_queries+=1
            self.goal_function.num_queries = self.number_of_queries
            if search_over: 
                self.goal_function.model.reset_inference_steps() 
                return initial_result
            print ('results best attack',results)
            self.goal_function.model.reset_inference_steps()
            return results[0]

            # sim = best_sim[0]
            # print ('last sim',sim)
            # max_changes = 0
            # for i in range(len(text_ls)):
            #     if text_ls[i]!=best_attack[i]:
            #         max_changes+=1
            # print ('best_attack',best_attack)
            
            # print ('return everything ',' '.join(best_attack), max_changes, len(changed_indices),  orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim)
            
            # sys.exit()
            # return ' '.join(best_attack), max_changes, len(changed_indices), \
            #       orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

            

        else:
            print ('returning failed result because flag==0')
            # results, search_over = self.get_goal_results([attacked_text])
            # return results[0]
            self.goal_function.model.reset_inference_steps()
            return initial_result
            # print("Not Found")
            # return '', 0,0, orig_label, orig_label, 0, 0, 0
        
        
        sys.exit()
        # Att_sen_new_sentence = AttackedText(new_sentence) 
        # print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
        # print ('att sen new words',Att_sen_new_sentence.words, len(Att_sen_new_sentence.words) ) 
        # Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
        # Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
        # # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(range(len(Att_sen_new_sentence.words)))
        # # Att_sen_new_sentence.attack_attrs['original_index_map'] = original_index_map
        # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
        # print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)

        # attacked_text.generate_new_attacked_text(new_words)




        
        # Main iteration loop
        for _ in range(self.max_iterations):
            # Step 2: Remove Unnecessary Replacement Words
            
            perturbed_text = self.remove_unnecessary_words(perturbed_text, attacked_text)
            print ('perturned+text',perturbed_text)

            # if attacked_text.words == perturbed_text.words:
            #     print ('should we skipp?')
            #     sys.exit() 

            # Step 3: Push Substitution Words towards Original Words
            perturbed_text = self.push_words_towards_original(perturbed_text, attacked_text)
            print ('perturned+text2',perturbed_text) 
            # if attacked_text == perturbed_text:
            #     print ('should we skipp 2?')
            #     sys.exit() 
            # Check if attack is successful
            results, search_over = self.get_goal_results([perturbed_text])
            # perturbed_result = initial_result.goal_function.call_model([perturbed_text])[0]
            # print ('results',results)

            # add semantic sim filter

            # this checks the generated test against the actual final use constraint
            
              

            final_result = results[0]

             

            if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                print ('attacked_text.text, final_result.attacked_text.text')
                print ('attk text',attacked_text.text)
                print ('final sre',final_result.attacked_text.text)
                sim_final_original, sim_final_pert = self.use_constraint.encode([attacked_text.text, final_result.attacked_text.text])

                if not isinstance(sim_final_original, torch.Tensor):
                    sim_final_original = torch.tensor(sim_final_original)

                if not isinstance(sim_final_pert, torch.Tensor):
                    sim_final_pert = torch.tensor(sim_final_pert)

                sim_score = self.sentence_encoder_use.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                print ('sim_score',sim_score, (1 - (args.similarity_threshold) / math.pi))
                if sim_score <  (1 - (args.similarity_threshold) / math.pi):
                    continue

                # final_result.num_queries = self.number_of_queries
                self.goal_function.num_queries = self.number_of_queries
                print ('final_result.num_queries',final_result.num_queries)
                # sys.exit()
                print ('final_result',final_result.attacked_text)
                print ('final_result.attacked_text.attack_attrs',final_result.attacked_text.attack_attrs)
                print ('final_result.attacked_text.attack_attrs[original_index_map]',final_result.attacked_text.attack_attrs['original_index_map'])
                # print ('final_result',final_result.original_text)
                # print ('final_result',final_result.perturbed_text)
                # print ('final_result.perturbed_result.attack_attrs',final_result.perturbed_result.attack_attrs)
                # if len(final_result.attacked_text.attack_attrs['newly_modified_indices']) == 0:
                #     final_result.attacked_text.attack_attrs['newly_modified_indices'] = {0}
                # if len(final_result.attacked_text.attack_attrs['modified_indices']) == 0:
                #     # final_result.attacked_text.attack_attrs['modified_indices'] = {0}
                #     return initial_result

                return final_result
        print ('just aviod everything')
        return initial_result

    def random_initialization(self, text):
        words = text.words
        tmp_text = text
        size_text = len(text.words)
        start_i = 0
        while start_i < size_text:
            # print ('start tmp text',tmp_text)
            words = tmp_text.words
            pos_tags = nltk.pos_tag(words)   
            # print ('pos_tags',pos_tags)
            if pos_tags[start_i][1].startswith(('VB', 'NN', 'JJ', 'RB')): 
                # print ('pos_tags[start_i][1]',pos_tags[start_i][1])
                replaced_with_synonyms = self.get_transformations(tmp_text, original_text=tmp_text,indices_to_modify=[start_i])
                # print ('replaced_with_synonyms',replaced_with_synonyms)
                if replaced_with_synonyms:
                    tmp_text = random.choice(replaced_with_synonyms)
                else:
                    pass
                
            start_i+=1
            
        adv_text = tmp_text
        return adv_text

 

    def remove_unnecessary_words(self, perturbed_text, original_text, check_skip=False):
        # Step 1: Identify words to replace back
        candidate_set = []
        word_importance_scores = [] 
        # print ('original_text',original_text)
        # print ('perturbed_text',perturbed_text)
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                # Replace perturbed_word with original_word
                temp_text = perturbed_text.replace_word_at_index(i, original_word)

                # Step 2: Check if still adversarial and calculate semantic similarity
                model_outputs = self.goal_function._call_model([temp_text])
                current_goal_status = self.goal_function._get_goal_status(
                    model_outputs[0], temp_text, check_skip=check_skip
                )
                self.number_of_queries+=1
                # print ('temp_text',temp_text,i,current_goal_status,GoalFunctionResultStatus.SUCCEEDED)
                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    candidate_set.append((i, temp_text))
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_text.text, temp_text.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    
                    word_importance_scores.append((i, sim_score))

        # Step 3: Sort word importance scores in descending order and restore original words
        word_importance_scores.sort(key=lambda x: x[1], reverse=True)
        print ('attack_attrs ',perturbed_text.attack_attrs,perturbed_text  ) 
        print ('replace indexs',word_importance_scores)
        for idx, _ in word_importance_scores:
            temp_text2 = perturbed_text.replace_word_at_index(idx, original_text.words[idx])
            temp_text2.attack_attrs['modified_indices'].remove(idx)
            print ('temp_text2_word_imp',idx,temp_text2.attack_attrs,temp_text2)
            # print ('original_index_map',temp_text2.attack_attrs.original_index_map)
            
            model_outputs = self.goal_function._call_model([temp_text2])
            current_goal_status = self.goal_function._get_goal_status(
                model_outputs[0], temp_text2, check_skip=check_skip
            )
            self.number_of_queries+=1

            # print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)

 
            if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                # If perturbed_text is no longer adversarial, revert the last change
                perturbed_text = temp_text2
                # perturbed_text = perturbed_text.replace_word_at_index(idx, perturbed_text.words[idx])
            else:
                break
        # print ('original_text',original_text)
        # print ('perturbed_text',perturbed_text) 
        return perturbed_text

    def get_vector(self, embedding, word):
        if isinstance(word, str):
            if word in embedding._word2index:
                word_index = embedding._word2index[word]
            else:
                return None  # Word not found in the dictionary
        else:
            word_index = word

        vector = embedding.embedding_matrix[word_index]
        return torch.tensor(vector).to(textattack.shared.utils.device)

    def push_words_towards_original(self, perturbed_text, original_text, check_skip=False):
        # Step 1: Calculate Euclidean distances and sampling probabilities
        distances = []
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                # Using the get_vector function
                perturbed_vec = self.get_vector(self.embedding, perturbed_word)
                if perturbed_vec is None:
                    continue  # Skip to the next word
                original_vec = self.get_vector(self.embedding, original_word)
                if original_vec is None:
                    continue  # Skip to the next word
                distance = np.linalg.norm(perturbed_vec.cpu().numpy() - original_vec.cpu().numpy())
                distances.append((i, distance))

        if not distances:
            return perturbed_text

        # Normalize distances to get probabilities
        distances.sort(key=lambda x: x[1])
        indices, dist_values = zip(*distances)
        exp_dist_values = np.exp(dist_values)
        probabilities = exp_dist_values / np.sum(exp_dist_values)
        print ('probabilities',probabilities)

        # temp_perturbed_text = copy.deepcopy(perturbed_text)
        
        # Step 2: Iterate with sampling based on the probabilities 
        while len(indices) > 0:
            i = np.random.choice(indices, p=probabilities)
            print ('indices',indices,i)
            perturbed_word = perturbed_text.words[i]
            original_word = original_text.words[i]

            sentence_replaced = self.get_transformations(original_text, original_text=original_text, indices_to_modify=[i])
            synonyms = [s.words[i] for s in sentence_replaced]


            # Get top k synonyms
            k = 10  # Number of synonyms to sample
            top_k_synonyms_indexes  = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=k)
            top_k_synonyms = [self.embedding._index2word[index] for index in top_k_synonyms_indexes]

            # Find the best anchor synonym with the highest semantic similarity
            max_similarity = -float('inf')
            w_bar = None
            temp_text_bar = None
            filtered_synonyms = None
            print ('top_k_synonyms',top_k_synonyms)
            # temp_text2 = copy.deepcopy(perturbed_text)
            for synonym in top_k_synonyms:
                if perturbed_word == synonym:
                    continue # skip swapping the same word
                print ('synonym',i,synonym)
                # temp_text2 = copy.deepcopy(perturbed_text)
                temp_text2 = perturbed_text.replace_word_at_index(i, synonym)

                # Check if the substitution still results in an adversarial example
                model_outputs = self.goal_function._call_model([temp_text2])
                current_goal_status = self.goal_function._get_goal_status(
                    model_outputs[0], temp_text2, check_skip=check_skip
                )
                self.number_of_queries+=1

                print ('temp_text2_top_k_syn',i,synonym,temp_text2.attack_attrs,temp_text2,current_goal_status , GoalFunctionResultStatus.SUCCEEDED)

                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    # Compute semantic similarity at the word level
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0)).item()
                    print ('sim scores push towards orgin')
                    if sim_score > max_similarity:
                        max_similarity = sim_score
                        w_bar = synonym
                        temp_text_bar = temp_text2

            
            if w_bar:# is None:
                  # Skip this index if no suitable anchor synonym is found
            
                number_entries = len(self.embedding.nn_matrix[self.embedding._word2index[original_word]] )
                print ('num entries',number_entries)
                all_synonyms = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=number_entries)
                all_synonyms = [self.embedding._index2word[index] for index in all_synonyms]
            
                print ('all_synonyms',all_synonyms)
                filtered_synonyms = []
                for synonym in all_synonyms:
                    if perturbed_word == synonym or w_bar == synonym  :
                        continue # skip swapping/checking the same word and the anchor word
                    # Compute semantic similarity with w_bar and original_word
                    sim_w_bar, sim_synonym = self.sentence_encoder_use.encode([w_bar, synonym])
                    sim_org, sim_synonym_org = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_w_bar, torch.Tensor):
                        sim_w_bar = torch.tensor(sim_w_bar)
                    if not isinstance(sim_synonym, torch.Tensor):
                        sim_synonym = torch.tensor(sim_synonym)
                    if not isinstance(sim_org, torch.Tensor):
                        sim_org = torch.tensor(sim_org)
                    if not isinstance(sim_synonym_org, torch.Tensor):
                        sim_synonym_org = torch.tensor(sim_synonym_org)

                    sim_score_w_bar = self.sentence_encoder_use.sim_metric(sim_w_bar.unsqueeze(0), sim_synonym.unsqueeze(0)).item()
                    sim_score_org = self.sentence_encoder_use.sim_metric(sim_org.unsqueeze(0), sim_synonym_org.unsqueeze(0)).item()

                    if sim_score_w_bar > sim_score_org:
                        filtered_synonyms.append((sim_score_w_bar, synonym))

            if  filtered_synonyms:
                # continue  # Skip this index if no suitable synonym is found

                # Sort the filtered synonyms by their semantic similarity score in descending order
                filtered_synonyms.sort(key=lambda item: item[0], reverse=True)
                print ('filtered_synonyms',filtered_synonyms)
                
                

                print ('perturbed text',perturbed_text.attack_attrs,perturbed_text)
                for _, synonym in filtered_synonyms:
                    temp_text2 = perturbed_text.replace_word_at_index(i, synonym) 
                    # temp_text2.attack_attrs['modified_indices'].remove(i)
                    print ('temp_text2_filtered_syn',i,temp_text2.attack_attrs,temp_text2)
                    # Check if the substitution still results in an adversarial example
                    model_outputs = self.goal_function._call_model([temp_text2]) 
                    current_goal_status = self.goal_function._get_goal_status(
                        model_outputs[0], temp_text2, check_skip=check_skip
                    )
                    self.number_of_queries+=1
                    print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)
                    if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        perturbed_text = temp_text2
                        break
                    
            
            print ('perturbed_text',perturbed_text)  
            idx = indices.index(i)
            indices = indices[:idx] + indices[idx + 1:] 
            print ('indices2',indices,idx)
            
            probabilities = np.delete(probabilities, idx)
            probabilities /= np.sum(probabilities)   
        # sys.exit()
        return perturbed_text 

    def get_transformations(self, text, index):
        return self.transformation(text, index)

    def get_similarity(self, word1, word2):
        return self.transformation.get_cosine_similarity(word1, word2)
    
    @property
    def is_black_box(self):
        return True


class GreedySearch(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """
    def __init__(self,beam_width=1,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # def __init__(self,index_order_technique,goal_function, beam_width=1):
    #     self.index_order_technique = index_order_technique
    #     self.goal_function = goal_function
        self.beam_width = beam_width
        self.previeous_beam = [] 
    

    def _get_index_order(self, initial_text, max_len=-1):
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        elif self.index_order_technique  == 'prompt_top_k': 
            # ground_truth = 'positive'
            K = '' # number of important words to return 
            print ('self.ground_truth_output',self.goal_function.ground_truth_output) # should test with and without ground truth
            
            label_list = self.dataset.label_names
            label_index = self.goal_function.ground_truth_output
            expected_prediction, other_classes = self.predictor.prompt_class._identify_correct_incorrect_labels( label_index)
            
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('initial_text',initial_text)
            print ('len text indeces to order',len_text,indices_to_order)
            examples = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            
            # prompt = f"""{self.start_prompt_header}return the most important words for the task of {task} where the text is '{initial_text.text}' and is classified as {ground_truth}.
            # Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            # Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            # The top {K} words are: {self.end_prompt_footer}"""

            prompt = f"""{self.start_prompt_header}return the most important words (in descending order of importance) for the following text '{initial_text.text}' which is classified as {expected_prediction}.
            Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            The top {K} words in descending order are: {self.end_prompt_footer}"""
            # print ('prompt',prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        

            generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**generate_args)
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            print("Generated Text word order:", generated_text)
            print ('words of original text',initial_text.words)
            words_list = [word.strip().lower() for word in generated_text.split(',')]
            initial_words_list = [word.lower() for word in initial_text.words]
            print ('word_list',words_list, set(initial_words_list), set(words_list)  )
            # find set of words that are in both, then for each word in words list that is in set find index in initial_text.words
            interesection_words = set(initial_words_list) & set(words_list) 
            print ('interesection_words',interesection_words)
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('len_text, indices_to_order',len_text, indices_to_order)
            
            
            indices_to_order = []

            # for i,w in enumerate(initial_words_list):
            #     if w in interesection_words:
            #         indices_to_order.append(i)

            initial_word_index_pair = {j:i for i,j in enumerate(initial_words_list) }
            print ('initial_word_index_pair',initial_word_index_pair)
            for i,w in enumerate(words_list):
                if w in interesection_words:
                    indices_to_order.append(initial_word_index_pair[w])# initial_words_list.index(w)) # potntially use miaos ranking explenation for theory behond this?


            print('indices_to_order',indices_to_order) 

            # len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            search_over = False 

            if len(index_order) == 0:
                len_text, indices_to_order = self.get_indices_to_order(initial_text)
                index_order = indices_to_order
                np.random.shuffle(index_order)
                search_over = False

            search_over = False
            return index_order, search_over
        elif self.index_order_technique == 'delete': 
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            print ('leave_one_texts',leave_one_texts)
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            print ('leave_one_results, search_over',leave_one_results, search_over)

            index_scores = np.array([result.score for result in leave_one_results])
            print ('index_scores',index_scores, 'search_over',search_over)
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            return  index_order, search_over

    def perform_search(self, initial_result): 
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None 
        # pick two indexes and modify them
        while i < len(index_order) and not search_over:
            if i > self.max_iter_i:
                break
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter')
            print ('null_label',null_label)
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            # print ('results_after_null_filter',results)
            results = sorted(results, key=lambda x: -x.score)
            print ('results_after_sorted',results)
            if len(results) == 0:
                continue
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score: 
                cur_result = results[0] 
                 
            else:
                continue 
        self.goal_function.model.reset_inference_steps() # so far only used by dirischlet plots
        return cur_result


         

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]



class GreedySearch_USE(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """
    def __init__(self,beam_width=1,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # def __init__(self,index_order_technique,goal_function, beam_width=1):
    #     self.index_order_technique = index_order_technique
    #     self.goal_function = goal_function
        self.beam_width = beam_width
        self.previeous_beam = [] 
    

    def _get_index_order(self, initial_text, max_len=-1):
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        elif self.index_order_technique  == 'prompt_top_k': 
            # ground_truth = 'positive'
            K = '' # number of important words to return 
            print ('self.ground_truth_output',self.goal_function.ground_truth_output) # should test with and without ground truth
            
            label_list = self.dataset.label_names
            label_index = self.goal_function.ground_truth_output
            expected_prediction, other_classes = self.predictor.prompt_class._identify_correct_incorrect_labels( label_index)
            
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('initial_text',initial_text)
            print ('len text indeces to order',len_text,indices_to_order)
            examples = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            
            # prompt = f"""{self.start_prompt_header}return the most important words for the task of {task} where the text is '{initial_text.text}' and is classified as {ground_truth}.
            # Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            # Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            # The top {K} words are: {self.end_prompt_footer}"""

            prompt = f"""{self.start_prompt_header}return the most important words (in descending order of importance) for the following text '{initial_text.text}' which is classified as {expected_prediction}.
            Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            The top {K} words in descending order are: {self.end_prompt_footer}"""
            # print ('prompt',prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        

            generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**generate_args)
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            print("Generated Text word order:", generated_text)
            print ('words of original text',initial_text.words)
            words_list = [word.strip().lower() for word in generated_text.split(',')]
            initial_words_list = [word.lower() for word in initial_text.words]
            print ('word_list',words_list, set(initial_words_list), set(words_list)  )
            # find set of words that are in both, then for each word in words list that is in set find index in initial_text.words
            interesection_words = set(initial_words_list) & set(words_list) 
            print ('interesection_words',interesection_words)
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('len_text, indices_to_order',len_text, indices_to_order)
            
            
            indices_to_order = []

            # for i,w in enumerate(initial_words_list):
            #     if w in interesection_words:
            #         indices_to_order.append(i)

            initial_word_index_pair = {j:i for i,j in enumerate(initial_words_list) }
            print ('initial_word_index_pair',initial_word_index_pair)
            for i,w in enumerate(words_list):
                if w in interesection_words:
                    indices_to_order.append(initial_word_index_pair[w])# initial_words_list.index(w)) # potntially use miaos ranking explenation for theory behond this?


            print('indices_to_order',indices_to_order) 

            # len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            search_over = False 

            if len(index_order) == 0:
                len_text, indices_to_order = self.get_indices_to_order(initial_text)
                index_order = indices_to_order
                np.random.shuffle(index_order)
                search_over = False

            search_over = False
            return index_order, search_over
        elif self.index_order_technique == 'delete': 
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            print ('leave_one_texts',leave_one_texts)
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            print ('leave_one_results, search_over',leave_one_results, search_over)

            index_scores = np.array([result.score for result in leave_one_results])
            print ('index_scores',index_scores, 'search_over',search_over)
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            return  index_order, search_over
    def perform_search(self, initial_result): 
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None  
        best_result = None
        max_similarity = -float("inf")
        # pick two indexes and modify them
        while i < len(index_order) and not search_over:
            if i > self.max_iter_i:
                break
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter')
            print ('null_label',null_label)
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            # print ('results_after_null_filter',results)

            # a self.track_result_score
            # can put all scores here so that we can access them later
            # for each i we can save a list of scores, then do the max in each, we expect for each i to increase
            # we can then show how this increases by number of perturbations by definition a low and high score
            # are high confidences, while a mid score are medium confidences

            results = sorted(results, key=lambda x: -x.score)
            print ('results_after_sorted',results)
            if len(results) == 0:
                continue
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score: 
                cur_result = results[0] 
                 
            else:
                continue 
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                candidate = cur_result.attacked_text
                try:
                    similarity_score = candidate.attack_attrs["similarity_score"]
                except KeyError:
                    break
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_result = cur_result

        self.goal_function.model.reset_inference_steps()
        if best_result: 
            return best_result
        else:
            return cur_result 

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]


class GreedySearch_USE_Hardlabel(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """
    def __init__(self,beam_width=1,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # def __init__(self,index_order_technique,goal_function, beam_width=1):
    #     self.index_order_technique = index_order_technique
    #     self.goal_function = goal_function
        self.beam_width = beam_width
        self.previeous_beam = [] 
    

    def _get_index_order(self, initial_text, max_len=-1):
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        elif self.index_order_technique  == 'prompt_top_k': 
            # ground_truth = 'positive'
            K = '' # number of important words to return 
            print ('self.ground_truth_output',self.goal_function.ground_truth_output) # should test with and without ground truth
            
            label_list = self.dataset.label_names
            label_index = self.goal_function.ground_truth_output
            expected_prediction, other_classes = self.predictor.prompt_class._identify_correct_incorrect_labels( label_index)
            
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('initial_text',initial_text)
            print ('len text indeces to order',len_text,indices_to_order)
            examples = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            
            # prompt = f"""{self.start_prompt_header}return the most important words for the task of {task} where the text is '{initial_text.text}' and is classified as {ground_truth}.
            # Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            # Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            # The top {K} words are: {self.end_prompt_footer}"""

            prompt = f"""{self.start_prompt_header}return the most important words (in descending order of importance) for the following text '{initial_text.text}' which is classified as {expected_prediction}.
            Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            The top {K} words in descending order are: {self.end_prompt_footer}"""
            # print ('prompt',prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        

            generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**generate_args)
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            print("Generated Text word order:", generated_text)
            print ('words of original text',initial_text.words)
            words_list = [word.strip().lower() for word in generated_text.split(',')]
            initial_words_list = [word.lower() for word in initial_text.words]
            print ('word_list',words_list, set(initial_words_list), set(words_list)  )
            # find set of words that are in both, then for each word in words list that is in set find index in initial_text.words
            interesection_words = set(initial_words_list) & set(words_list) 
            print ('interesection_words',interesection_words)
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('len_text, indices_to_order',len_text, indices_to_order)
            
            
            indices_to_order = []

            # for i,w in enumerate(initial_words_list):
            #     if w in interesection_words:
            #         indices_to_order.append(i)

            initial_word_index_pair = {j:i for i,j in enumerate(initial_words_list) }
            print ('initial_word_index_pair',initial_word_index_pair)
            for i,w in enumerate(words_list):
                if w in interesection_words:
                    indices_to_order.append(initial_word_index_pair[w])# initial_words_list.index(w)) # potntially use miaos ranking explenation for theory behond this?


            print('indices_to_order',indices_to_order) 

            # len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            search_over = False 

            if len(index_order) == 0:
                len_text, indices_to_order = self.get_indices_to_order(initial_text)
                index_order = indices_to_order
                np.random.shuffle(index_order)
                search_over = False

            search_over = False
            return index_order, search_over
        elif self.index_order_technique == 'delete': 
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            print ('leave_one_texts',leave_one_texts)
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            print ('leave_one_results, search_over',leave_one_results, search_over)

            index_scores = np.array([result.score for result in leave_one_results])
            print ('index_scores',index_scores, 'search_over',search_over)
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            return  index_order, search_over
    def perform_search(self, initial_result): 
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None  
        best_result = None
        max_similarity = -float("inf")
        # pick two indexes and modify them
        while i < len(index_order) and not search_over:
            if i > self.max_iter_i:
                break
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter')
            print ('null_label',null_label)
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            # print ('results_after_null_filter',results)

            # a self.track_result_score
            # can put all scores here so that we can access them later
            # for each i we can save a list of scores, then do the max in each, we expect for each i to increase
            # we can then show how this increases by number of perturbations by definition a low and high score
            # are high confidences, while a mid score are medium confidences
 

            
            print ('results_after_sorted',results)
            if len(results) == 0:
                continue

            # for result in results:
            #     if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            #         print ('successful adv sample ', self.goal_function.num_queries, self.number_of_queries )
            #         self.goal_function.num_queries += self.number_of_queries # if adv sample found we query model N times to find a suitable transformation then N times to check it's actually adv
            #         return result

            # print ('ending queries',self.goal_function.num_queries,self.number_of_queries )
            # self.goal_function.num_queries += self.number_of_queries # no succesful adv samples found, we still query model N times to generate transformations
        
                
            for result in results:
                cur_result = result
                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    candidate = cur_result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = cur_result

        self.goal_function.model.reset_inference_steps()
        if best_result: 
            return best_result
        else:
            return cur_result 

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]


class GreedySearch_Margin(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """
    def __init__(self,beam_width=1,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # def __init__(self,index_order_technique,goal_function, beam_width=1):
    #     self.index_order_technique = index_order_technique
    #     self.goal_function = goal_function
        self.beam_width = beam_width
        self.previeous_beam = [] 
    

    def _get_index_order(self, initial_text, max_len=-1):
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        elif self.index_order_technique  == 'prompt_top_k': 
            # ground_truth = 'positive'
            K = '' # number of important words to return 
            print ('self.ground_truth_output',self.goal_function.ground_truth_output) # should test with and without ground truth
            
            label_list = self.dataset.label_names
            label_index = self.goal_function.ground_truth_output
            expected_prediction, other_classes = self.predictor.prompt_class._identify_correct_incorrect_labels( label_index)
            
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('initial_text',initial_text)
            print ('len text indeces to order',len_text,indices_to_order)
            examples = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            
            # prompt = f"""{self.start_prompt_header}return the most important words for the task of {task} where the text is '{initial_text.text}' and is classified as {ground_truth}.
            # Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            # Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            # The top {K} words are: {self.end_prompt_footer}"""

            prompt = f"""{self.start_prompt_header}return the most important words (in descending order of importance) for the following text '{initial_text.text}' which is classified as {expected_prediction}.
            Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            The top {K} words in descending order are: {self.end_prompt_footer}"""
            # print ('prompt',prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        

            generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**generate_args)
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            print("Generated Text word order:", generated_text)
            print ('words of original text',initial_text.words)
            words_list = [word.strip().lower() for word in generated_text.split(',')]
            initial_words_list = [word.lower() for word in initial_text.words]
            print ('word_list',words_list, set(initial_words_list), set(words_list)  )
            # find set of words that are in both, then for each word in words list that is in set find index in initial_text.words
            interesection_words = set(initial_words_list) & set(words_list) 
            print ('interesection_words',interesection_words)
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('len_text, indices_to_order',len_text, indices_to_order)
            
            
            indices_to_order = []

            # for i,w in enumerate(initial_words_list):
            #     if w in interesection_words:
            #         indices_to_order.append(i)

            initial_word_index_pair = {j:i for i,j in enumerate(initial_words_list) }
            print ('initial_word_index_pair',initial_word_index_pair)
            for i,w in enumerate(words_list):
                if w in interesection_words:
                    indices_to_order.append(initial_word_index_pair[w])# initial_words_list.index(w)) # potntially use miaos ranking explenation for theory behond this?


            print('indices_to_order',indices_to_order) 

            # len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            search_over = False 

            if len(index_order) == 0:
                len_text, indices_to_order = self.get_indices_to_order(initial_text)
                index_order = indices_to_order
                np.random.shuffle(index_order)
                search_over = False

            search_over = False
            return index_order, search_over
        elif self.index_order_technique == 'delete': 
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            print ('leave_one_texts',leave_one_texts)
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            print ('leave_one_results, search_over',leave_one_results, search_over)

            index_scores = np.array([result.score for result in leave_one_results])
            print ('index_scores',index_scores, 'search_over',search_over)
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            return  index_order, search_over
    def perform_search(self, initial_result): 
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None  
        best_result = None
        max_similarity = -float("inf")
        # pick two indexes and modify them
        while i < len(index_order) and not search_over:
            if i > 5:
                break
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter')
            print ('null_label',null_label)
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            # print ('results_after_null_filter',results)

            # a self.track_result_score
            # can put all scores here so that we can access them later
            # for each i we can save a list of scores, then do the max in each, we expect for each i to increase
            # we can then show how this increases by number of perturbations by definition a low and high score
            # are high confidences, while a mid score are medium confidences

            results = sorted(results, key=lambda x: -x.score)
            print ('results_after_sorted',results)
            if len(results) == 0:
                continue
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score: 
                cur_result = results[0] 
                 
            else:
                continue 
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                candidate = cur_result.attacked_text
                try:
                    similarity_score = candidate.attack_attrs["similarity_score"]
                except KeyError:
                    break
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_result = cur_result

                best_result = cur_result

        if best_result:
            self.goal_function.model.reset_inference_steps()
            return best_result
        self.goal_function.model.reset_inference_steps()
        return cur_result 

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]



class GreedySearch_WithMin_USE(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """
    def __init__(self,beam_width=1,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # def __init__(self,index_order_technique,goal_function, beam_width=1):
    #     self.index_order_technique = index_order_technique
    #     self.goal_function = goal_function
        self.beam_width = beam_width
        self.previeous_beam = [] 
        self.sentence_encoder_use = UniversalSentenceEncoder(window_size=15)
    

    def _get_index_order(self, initial_text, max_len=-1):
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        elif self.index_order_technique  == 'prompt_top_k':
            task = 'sentiment classification'
            ground_truth = 'positive'
            K = 3 # number of important words to return
            print ('self.ground_truth_output',self.goal_function.ground_truth_output) # should test with and without ground truth
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('initial_text',initial_text)
            print ('len text indeces to order',len_text,indices_to_order)
            examples = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            
            prompt = f"""{self.start_prompt_header}return the most important words for the task of {task} where the text is '{initial_text.text}' and is classified as {ground_truth}.
            Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            The top {K} words are: {self.end_prompt_footer}"""
            # print ('prompt',prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        

            generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**generate_args)
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            print("Generated Text word order:", generated_text)
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over

    def perform_search(self, initial_result):  
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None  
        best_result = None
        max_similarity = -float("inf")
        # pick two indexes and modify them
        while i < 5 and not search_over:
            if i > 5:
                break
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=None,
            )
            i += 1
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter')
            print ('null_label',null_label)
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            # print ('results_after_null_filter',results)
            results = sorted(results, key=lambda x: -x.score)
            print ('results_after_sorted',results)
            if len(results) == 0:
                continue
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score: 
                cur_result = results[0] 
                
                 
            else:
                continue 
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                candidate = cur_result.attacked_text 
                
                try:
                    similarity_score = candidate.attack_attrs["similarity_score"]
                except KeyError:
                    break
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_result = cur_result
        print ('starting the use exploration')

        #checking it's actually adv
        model_outputs = self.goal_function._call_model([cur_result.attacked_text]) 
        current_goal_status = self.goal_function._get_goal_status(
                    model_outputs[0], cur_result.attacked_text, check_skip=False
        )  
        print ('should be true',model_outputs, current_goal_status, GoalFunctionResultStatus.SUCCEEDED)
        if best_result:
                result_text = best_result.attacked_text
                print ('best result',result_text)
        else:
            result_text = cur_result.attacked_text
            print ('current result',result_text)

        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([result_text.text, initial_result.attacked_text.text])

        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

        original_sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
        print ('original_sim_score',original_sim_score)
 
        for i in range(5):
            

            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([result_text.text, initial_result.attacked_text.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            original_sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            
            # calling llm('give me the meaning between 0 and 1 of how the meanig is preserved between x and x' ')
            
            print ('original_sim_score',original_sim_score)

            print ('best_result.attacked_text,initial_result.attacked_text',result_text,initial_result.attacked_text)
            transformations_sem_sim =  self.transformation._maximise_semantic_sim(initial_result.attacked_text, result_text)
            print ('transformations_sem_sim',transformations_sem_sim)
            semantic_ranking = []
            for k,result in enumerate(transformations_sem_sim):
                temp_text = result
                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([temp_text.text, initial_result.attacked_text.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                print ('sim_score',sim_score) 

                model_outputs = self.goal_function._call_model([result]) 
                current_goal_status = self.goal_function._get_goal_status(
                    model_outputs[0], result, check_skip=False
                )  

                semantic_ranking.append((k,result, current_goal_status == GoalFunctionResultStatus.SUCCEEDED ,  sim_score))
                # filter out and rank

                print ('result status etc',model_outputs, current_goal_status, GoalFunctionResultStatus.SUCCEEDED)
                print ('semantic_ranking',semantic_ranking)
                # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                #     print ('successful goalresultstatus')
                #     pass
            filtered_tuples = [tup for tup in semantic_ranking if tup[2] == True]
            print ('filtered_tuples',filtered_tuples)
            if filtered_tuples:
                sorted_tuples = sorted(filtered_tuples, key=lambda x: x[3], reverse=True)
            else:
                sorted_tuples = []
            print ('sorted_tuples',sorted_tuples)
            if len(sorted_tuples) != 0:
                if sorted_tuples[0][3] > original_sim_score:
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([sorted_tuples[0][1].text, initial_result.attacked_text.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    original_sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    print ('original_sim_score_just_before_update',original_sim_score)
                    result_text = sorted_tuples[0][1]
                    print ('result_text.attack_attrs',result_text.attack_attrs)
                    
                else:
                    pass
            print ('result_text',result_text)

        # need to build result result_text
        final_result, search_over = self.get_goal_results([result_text])
        return final_result[0]
        # print ('final_result',final_result)
        # sys.exit()
        # if best_result:
        #     self.goal_function.model.reset_inference_steps()
        #     return best_result
        # self.goal_function.model.reset_inference_steps()
        # return cur_result 

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]

if args.search_method == 'black_box':
    search_method = BlackBoxSearch(**vars(args))#num_transformations)
# elif args.search_method == 's1_black_box' :
#     search_method = BlackBoxSearch(**vars(args))#num_transformations)
elif args.search_method == 'greedy_search': #'s1' or args.search_method =='2step' or args.search_method == 'empirical' or args.search_method == 'k_pred_avg':
    search_method = GreedySearch(**vars(args))#index_order_technique,goal_function)#   GreedyWordSwapWIR(wir_method="delete")
elif  args.search_method =='greedy_search_use':
    search_method = GreedySearch_USE(**vars(args))
elif args.search_method == 'greedy_search_use_hardlabel':
    search_method = GreedySearch_USE_Hardlabel(**vars(args))
elif args.search_method=='sspattack':
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    search_method = SSPAttackSearch(**vars(args))
elif args.search_method=='texthoaxer':
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    search_method = TextHoaxer(**vars(args))
elif args.search_method=='greedy_search_withmin_use':
    search_method = GreedySearch_WithMin_USE(**vars(args))

# search_method_class_1 = AlzantotGeneticAlgorithm(pop_size=60, max_iters=20, post_crossover_check=False)

# Create the attack
greedy_attack = Attack(goal_function, constraints, transformation, search_method)
# genetic_attack = Attack(goal_function, constraints, transformation, search_method_class_1)








print ('Saving to:',args.test_folder,name_of_test )
attack_args = AttackArgs(
    num_examples=args.num_examples,
    log_to_csv=os.path.join(args.test_folder, f'normal_{name_of_test}.csv'),  # Adjusted to save in test_folder
    checkpoint_interval=1000,
    checkpoint_dir="checkpoints",
    disable_stdout=True,
    parallel=False,
    num_workers_per_device=8, 
    csv_coloring_style=None,
    enable_advance_metrics=True,
)

# Run attack for class 0
attacker_greedy_class_0 = Attacker(greedy_attack, dataset_class, attack_args)

import time

# Start the timer
start_time = time.time()
 
results = attacker_greedy_class_0.attack_dataset()
end_time = time.time()


 
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
hours, minutes = divmod(minutes, 60)

# Print the elapsed time
print(f"Execution Time: {int(hours):03}:{int(minutes):02}:{int(seconds):02}")
args.logging.info(f"Execution Time: {int(hours):03}:{int(minutes):02}:{int(seconds):02}")

from src.logging import log_results, log_results_extension, evaluate_results

log_results(results, test_folder=args.test_folder, name_of_test = name_of_test, name_log = 'normal', args = args)

# ppl_avg = textattack.metrics.quality_metrics.Perplexity().calculate(results)
# usem_avg = textattack.metrics.quality_metrics.USEMetric().calculate(results)

# import pandas as pd 
# from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult


log_results_extension(results, test_folder=args.test_folder, name_of_test = name_of_test, name_log = 'normal', args = args )

# file_path = os.path.join(test_folder, f'normal_{name_of_test}.csv')
# df = pd.read_csv(file_path)


# if 'original_perplexity' not in df.columns:
#     df['original_perplexity'] = None
# if 'attack_perplexity' not in df.columns:
#     df['attack_perplexity'] = None
# if 'attack_use_score' not in df.columns:
#     df['attack_use_score'] = None
# if 'num_words_perturbed' not in df.columns:
#     df['num_words_perturbed'] = None

# def count_successful_samples_quereis(results):
#     total_queries = 0
#     succ_samples = 0
#     for i, result in enumerate(results):
#         if isinstance(result, SuccessfulAttackResult):
#             total_queries+=result.num_queries
#             succ_samples+=1
#     return total_queries/succ_samples


# successful_samples_quereis = 0


 

# for i, result in enumerate(results):    

#     print ('result queries', result.num_queries)
#     try:
#         ppl = textattack.metrics.quality_metrics.Perplexity().calculate([result])
#         df.at[i, 'original_perplexity'] = ppl['avg_original_perplexity']
#         df.at[i, 'attack_perplexity'] = ppl['avg_attack_perplexity']
#     except Exception as e:
#         df.at[i, 'original_perplexity'] = -1
#         df.at[i, 'attack_perplexity'] = -1
#         print('Error calculating perplexity:', e)

#     # Calculate USE Metric
#     try:
#         usem = textattack.metrics.quality_metrics.USEMetric().calculate([result])
#         df.at[i, 'attack_use_score'] = usem['avg_attack_use_score']
#     except Exception as e:
#         df.at[i, 'attack_use_score'] = -1
#         print('Error calculating USE score:', e)


#     try:
#         print ('attack_attrs',result.original_result.attacked_text )
#         print ('attack_attrs',result.perturbed_result.attacked_text )
#         original_text = result.original_result.attacked_text
#         perturbed_text = result.perturbed_result.attacked_text
#         num_perturbed_words = original_text.words_diff_num(perturbed_text)
#         df.at[i, 'num_words_perturbed']  = num_perturbed_words
#     except Exception as e:
#         df.at[i, 'num_words_perturbed'] = -1
#         print('Error calculating number words perturbed:', e)


# df.to_csv(file_path, index=False)
 




evaluate_results(results, args)

import pickle
name_log = 'results'
file_path = os.path.join(args.test_folder, f'{name_log}_{name_of_test}.pkl')

# Saving the `results` list to a file
with open(file_path, 'wb') as file:
    pickle.dump(results, file)

print(f'Results saved to {file_path}')

# To load the `results` list from the file
with open(file_path, 'rb') as file:
    loaded_results = pickle.load(file)

for result in loaded_results:
    original_text = result.original_result.attacked_text
    perturbed_text = result.perturbed_result.attacked_text
    print ('original_text',original_text)
    print ('perturbed_text',perturbed_text)



#is it possible to first save results as pickle objects in the test folder?

# with result we have the orignal text and perturbed text and ground truth
# we can have a list [prompting_class1, prompting_class2 etc..]

# we then iterate through each result in results and for each item in promptingclass we call predict_and_confidence to get 
# the guess, probs, confidence

# then we would have e.g multiple guess e.g predictions, for 5 prompitngclasses [1,1,0,1,1], so a vector of different predictions
# we can then check out of how many of these we get the right prediction and do other extended evaluations. peraps we can have an accuracy for each
# classification method



# original_result_true_labels = []
# original_result_predictions = []

# result_true_labels = [] # assume [0,1,0,1,0,0] have ground truths
# result_predictions = []

# for i, result in enumerate(results):  
#     # print ('prediction orig',result.original_result.output)
#     # print ('true label orig', result.original_result.ground_truth_output)
#     # if result.original_result.output == args.n_classes:
#     #     filtered_results.append(result)
#     original_result_true_labels.append(result.original_result.ground_truth_output)
#     original_result_predictions.append(result.original_result.output)

#     # print ('prediction pert',result.perturbed_result.output)
#     # print ('true label pert', result.perturbed_result.ground_truth_output)
#     result_true_labels.append(result.perturbed_result.ground_truth_output)
#     result_predictions.append(result.perturbed_result.output)



# # Output the confusion matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# print ('original_result_true_labels',original_result_true_labels)
# print ('original_result_predictions',original_result_predictions)

# orig_conf_matrix = confusion_matrix(original_result_true_labels, original_result_predictions, labels=list(range(args.n_classes+1)))
# print("Orig Confusion Matrix:")
# print(orig_conf_matrix)
# orig_acc = accuracy_score(original_result_true_labels, original_result_predictions)*100.0
# print("Orig Accuracy:", orig_acc)
# print ('original_result_true_labels',original_result_true_labels)
# print ('original_result_predictions',original_result_predictions)
# null_class = args.n_classes
# # Filter out null class instances
# orig_filtered_predictions = [pred for pred in original_result_predictions if pred != null_class]
# orig_filtered_true_labels = [true_label for true_label, pred in zip(original_result_true_labels, original_result_predictions) if pred != null_class]
# print ('orig_filtered_true_labels',orig_filtered_true_labels)
# print ('orig_filtered_predictions',orig_filtered_predictions)
# # Calculate the accuracy ignoring the null class
# orig_filtered_accuracy = accuracy_score(orig_filtered_true_labels, orig_filtered_predictions)*100.0

# print(f'Orig Filtered Accuracy: {orig_filtered_accuracy:.4f}')





# conf_matrix = confusion_matrix(result_true_labels, result_predictions, labels=list(range(args.n_classes+1)))
# print("Confusion Matrix:")
# print(conf_matrix)
# acc = accuracy_score(result_true_labels, result_predictions)*100.0
# print("Accuracy:", acc )
# print ('result_true_labels',result_true_labels)
# print ('result_predictions',result_predictions)
# null_class = args.n_classes
# # Filter out null class instances
# filtered_predictions = [pred for pred in result_predictions if pred != null_class]
# filtered_true_labels = [true_label for true_label, pred in zip(result_true_labels, result_predictions) if pred != null_class]
# print ('filtered_true_labels',filtered_true_labels)
# print ('filtered_predictions',filtered_predictions)
# # Calculate the accuracy ignoring the null class
# filtered_accuracy = accuracy_score(filtered_true_labels, filtered_predictions)*100.0

# print(f'Filtered Accuracy: {filtered_accuracy:.4f}')

# successful_attacks = [true_label for true_label, pred in zip(filtered_true_labels, filtered_predictions) if true_label != pred] 
# filtered_attack_success_rate = (len(successful_attacks)*100.0)/len(filtered_predictions)
# print ('filtered attack success rate', filtered_attack_success_rate)

# successful_samples_queries = count_successful_samples_quereis(results)
# print ('Successful samples queries:',successful_samples_queries)
 

 
  
filtered_results = []
for i, result in enumerate(results):   
    if result.original_result.output != args.n_classes: 
        filtered_results.append(result) 

log_results(filtered_results, test_folder=args.test_folder, name_of_test = name_of_test, name_log = 'filtered', args = args)

# print ('filtered_results',filtered_results,len(filtered_results))
# # Generate summary
# attack_log_manager.add_output_file(filename = os.path.join(test_folder, f'filtered_res_{name_of_test}.txt'), color_method = "file")
# attack_log_manager.add_output_csv(filename = os.path.join(test_folder, f'filtered_csv{name_of_test}.csv'), color_method = "file"):
 
# attack_log_manager.enable_stdout()
# attack_log_manager.log_summary() 






# print ('orig_filtered_true_labels for roc',orig_filtered_true_labels)
# print ('orig_filtered_predictions for roc',orig_filtered_predictions)


# from src.utils.shared import plot_and_calculate_roc_metrics
# plot_and_calculate_roc_metrics(true_labels = orig_filtered_true_labels,
#                                 probabilities = orig_filtered_predictions,
#                                 name_plot = 'attack_roc', args = args)


# ppl_avg = textattack.metrics.quality_metrics.Perplexity().calculate(filtered_results)
# usem_avg = textattack.metrics.quality_metrics.USEMetric().calculate(filtered_results)
# print ('ppx',ppl_avg,'usem',usem_avg)
# # Define attack arguments
# attack_args = AttackArgs(
#     num_examples=1000,  # Adjust based on the number of examples from each class you wish to attack
#     log_to_csv=os.path.join(test_folder,"log_greedy_class1.csv"),
#     checkpoint_interval=1000,
#     checkpoint_dir="checkpoints",
#     disable_stdout=False,
#     parallel=False,
#     num_workers_per_device=8,
# )
# # Optionally, run attack for class 1 with the same or a different configuration
# attacker_greedy_class_1 = Attacker(greedy_attack, dataset_class_1_ta, attack_args)
# attacker_greedy_class_1.attack_dataset()



# ## inverted


# # Define attack arguments
# attack_args = AttackArgs(
#     num_examples=1000,  # Adjust based on the number of examples from each class you wish to attack
#     log_to_csv=os.path.join(test_folder,"log_greedy_class1.csv"),
#     checkpoint_interval=1000,
#     checkpoint_dir="checkpoints",
#     disable_stdout=False,
#     parallel=False,
#     num_workers_per_device=8,
# )

# End of Selection