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
from textattack.transformations import WordSwapEmbedding, WordSwapWordNet
from textattack.goal_functions import UntargetedClassification
from textattack.shared import AttackedText
from textattack.attack import Attack

from textattack.search_methods import GreedyWordSwapWIR, GreedySearch, BeamSearch

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

from textattack.constraints.grammaticality import PartOfSpeech
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
# Load model, tokenizer, and model wrapper
import os
os.environ["HF_HOME"] = "/mnt/hdd/brian/"


model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
# model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
# model_name = "meta-llama/Llama-2-7b-chat-hf"# "gpt2"  # Example placeholder, change to your actual model
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
# model_name = "TheBloke/Llama-2-70B-AWQ"

# tokenizer = AutoTokenizer.from_pretrained("./finetuned_sentiment_model")
# model = AutoModelForCausalLM.from_pretrained("./finetuned_sentiment_model")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer's pad token is set
print ('tokenizer.pad_token',tokenizer.pad_token )
print ( 'tokenizer.pad_token_id', tokenizer.pad_token_id)
model.config.pad_token_id = tokenizer.pad_token_id



from textattack.models.wrappers import ModelWrapper
import re



# Load and process dataset
dataset = HuggingFaceDataset("glue", "sst2", split="train", shuffle=True) 

# For dataset_class_1, only include sentences with 3 or more characters and label == 1
dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1 and len(text['sentence']) >= 3]
# dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1]
dataset_class_1_t = dataset_class_1[3000:3250]
incontext_dataset_class_1 = dataset_class_1[-5:]
# For dataset_class_0, only include sentences with 3 or more characters and label == 0
dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0 and len(text['sentence']) >= 3]
# dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0]
dataset_class_0_t = dataset_class_0[3000:3250]
incontext_dataset_class_0 = dataset_class_0[-5:]

dataset_class = dataset_class_1_t + dataset_class_0_t






         

PATTERNS = {"sst2": [
        ("Review:\n{sentence}\nIs this movie review sentence negative or positive?\n{options_}", "{answer}"),
        ("Short movie review: {sentence}\nDid the critic thinking positively or negatively of the movie?\n{options_}", "{answer}"),
        ("Sentence from a movie review: {sentence}\nWas the movie seen positively or negatively based on the preceding review?\n\n{options_}", "{answer}"),
        ("\"{sentence}\"\nHow would the sentiment of this sentence be perceived? \n\n{options_}", "{answer}"),
        ("Is the sentiment of the following sentence positive or negative? \n{sentence}\n{options_}", "{answer}"),
        ("What is the sentiment of the following movie review sentence? \n{sentence}\n{options_}", "{answer}"),
        ("Would the following phrase be considered positive or negative? \n\n{sentence}\n{options_}", "{answer}"),
        ("Does the following review have a positive or negative opinion of the movie?\n\n{sentence}\n{options_}", "{answer}"),
        # ("Write a {answer} movie review.", "{sentence}"),
        # ("Generate a short movie review that has {answer} sentiment.", "{sentence}"),
    ],}
 


import random
class HuggingFaceLLMWrapper(ModelWrapper):
    """A wrapper around HuggingFace for LLMs.

    Args:
        model: A HuggingFace pretrained LLM
        tokenizer: A HuggingFace pretrained tokenizer
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generated_max_length = 2

    def predict_sentiment_and_confidence(self,text,expected_sentiment):
        # for llama2 7 b
        num_classes = 2
        # prompt = f"[INST]What is the sentiment of the following movie review sentence, and provide your prediction confidence as a percentage? \n{t}\npositive, negative[/INST]"
        # for quantiside llama 70b
        prompt = f"""[INST]Provide your best guess and the probability that it is correct (0% to 100%) for
                the following sentiment. Give ONLY the guess and probability, no other words or
                explanation. For example:\n\nGuess: <most likely guess, either positive or negative; not
                a complete sentence, just the guess!>\n Probability: <the probability between 0%
                and 100% that your guess is correct, without any extra commentary whatsoever; just
                the probability!>\n\nThe question is: ${text} Sentiment & confidence:[/INST]"""
         

        #. “lowest”, “low”, “medium”,“high”, “highest”
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.model_device)
        
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

            # concatenated_logits = torch.tensor([probs[1], probs[0],0.0], device=model_device)
            # print ('softmax_probabilities',softmax_probabilities) 
            # logit_list.append(concatenated_logits)
             
            expected_sentiment = expected_sentiment
            confidence_empirical = confidence_result
            # return sentiment_result, probs ,confidence_result
            return expected_sentiment,confidence_empirical,sentiment_confidences
        else: 
            # temp_logit = torch.zeros(num_classes+1, dtype=torch.float)  
            # temp_logit[-1] = 1.0  
            # logit_list.append(torch.tensor(temp_logit, device=model_device))
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
        
    def predict_sentiment_and_verbal_confidence(self,text,expected_sentiment):
        # for llama2 7 b
        num_classes = 2
        # prompt = f"[INST]What is the sentiment of the following movie review sentence, and provide your prediction confidence as a percentage? \n{t}\npositive, negative[/INST]"
        # for quantiside llama 70b
        prompt = f"""[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') for
            for the following sentiment. Give ONLY the guess and verbal confidence, no other words or
            explanation. For example:\n\nGuess: <most likely guess, either positive or negative; not
            a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just
            the confidence!>\n\nThe sentiment is: ${text} Sentiment & confidence:[/INST]"""
         

        #. 'lowest', 'low', 'medium','high', 'highest'
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.model_device)
        
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
            # outputs = model.generate(**inputs, max_new_tokens=200,temperature = 0.7 ,top_k= 1)
            outputs = model.generate(**generate_args)
        
            # full_text  = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
        # Display the generated text from the model
        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
        print("Generated Text:", generated_text) 
        # Use regex to extract sentiment and confidence
        match_sentiment = re.search(r'positive|POSITIVE|Positive|negative|NEGATIVE|Negative', generated_text)
        # match_confidence = re.search(r"[-+]?\d*\.?\d+", generated_text) 
        match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
        print ('match_sentiment',match_sentiment,'match_confidence',match_verbal_confidence)
        
        confidence_map = {
            'lowest': 0,
            'low': 25,
            'medium': 50,
            'high': 75,
            'highest': 100
        } 


        if match_sentiment and match_verbal_confidence:
            sentiment_result = match_sentiment.group(0).lower()
            match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
            # confidence_result = 100 if confidence_result > 100 else confidence_result
            confidence_result =  float(match_confidence) /100   # Append '%' 
            if sentiment_result == 'positive':
                # probs = np.array([1-confidence_result,confidence_result])
                sentiment_confidences = { 'negative': 1-confidence_result, 'positive': confidence_result, 'null': 0.0 }
            else:
                sentiment_confidences = { 'negative': confidence_result, 'positive': 1- confidence_result, 'null': 0.0 }
                # probs = np.array([confidence_result,1-confidence_result])

            # concatenated_logits = torch.tensor([probs[1], probs[0],0.0], device=model_device)
            # print ('softmax_probabilities',softmax_probabilities) 
            # logit_list.append(concatenated_logits)
             
            expected_sentiment = expected_sentiment
            confidence_empirical = confidence_result
            # return sentiment_result, probs ,confidence_result
            return expected_sentiment,confidence_empirical,sentiment_confidences
        else: 
            # temp_logit = torch.zeros(num_classes+1, dtype=torch.float)  
            # temp_logit[-1] = 1.0  
            # logit_list.append(torch.tensor(temp_logit, device=model_device)) 
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
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model_device)
        
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

    def perform_multiple_predictions(self,text, n=20):
        results = []
        for _ in range(n):
            result = self.predict_sentiment(text)
            results.append(result)
        return results

    def analyze_results(self,text, expected_sentiment):
        results = self.perform_multiple_predictions(text)
        correct_predictions = sum(1 for sentiment in results if sentiment == expected_sentiment)
        confidence_empirical = (correct_predictions / len(results)) * 100

        sentiment_counts = { 'positive': 0, 'negative': 0, 'null': 0 }
        for sentiment in results:
            sentiment_counts[sentiment] += 1
        
        sentiment_confidences = { 'positive': 0, 'negative': 0, 'null': 0 }
        for sentiment, number_of_results in sentiment_counts.items():
            sentiment_confidences[sentiment] = (number_of_results / len(results))

        # average_confidence = sum(confidence for _, confidence in results) / len(results)

        print(f"Results for '{text}':")
        print(f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Null: {sentiment_counts['null']}")
        # print(f"Average model confidence: {average_confidence}%")
        print(f"Empirical confidence: {confidence_empirical}%")
        max_class = max(sentiment_counts, key=sentiment_counts.get)
        print ('max_class',max_class,expected_sentiment)
        return expected_sentiment,confidence_empirical,sentiment_confidences

    def __call__(self, text_input_list,ground_truth_output):
        """Returns a list of responses to the given input list.""" # here we add our promp

        
        
        # prompt = f"""Please perform a sentiment analysis on the provided text. Your task is to classify the sentiment of the text as either positive or negative. After analyzing the text, output only one of the following tokens: [Positive] or [Negative], based on the sentiment expressed in the text. Do not provide additional explanations or text.

        #     Text under analysis: {text_input_list}

        #     Sentiment:"""
        self.model_device = next(self.model.parameters()).device
        print ('model_device',self.model_device)
        logit_list = []
        for t in text_input_list:
            num_classes = 2

            print ('ground_truth_output',ground_truth_output)
            expected_prediction = 'negative' if ground_truth_output == 0 else 'positive'

            match_sentiment, match_confidence, confidences  = self.predict_sentiment_and_verbal_confidence(t, expected_sentiment=expected_prediction)
            print ('match_sentiment',match_sentiment)
            

            
            # can set the correct confidence that includes - null, but set null to 0, so that we never attack towards this
            print ('match_sentiment',match_sentiment,match_confidence,confidences)
            
            if match_sentiment != 'null':
                # sentiment_result = match_sentiment.group(0).lower()
                confidence_result = float(match_confidence) /100   # Append '%'
                sentiment_result = match_sentiment
                
                # if sentiment_result == 'positive':
                #     probs = np.array([1-confidence_result,confidence_result])
                # else:
                #     probs = np.array([confidence_result,1-confidence_result])
                probs =  np.array([confidences['negative'],confidences['positive'],confidences['null']]) 
                concatenated_logits = torch.tensor([probs[0], probs[1],probs[2]], device=self.model_device)
                print ('concatenated_logits',concatenated_logits) 
                logit_list.append(concatenated_logits)


                # return sentiment_result, probs ,confidence_result
            else: 
                # temp_logit = torch.zeros(num_classes+1, dtype=torch.float) 
                temp_logit =  np.array([confidences['negative'],confidences['positive'],confidences['null']])
                # print ('temp_logit',temp_logit)
                # temp_logit[-1] = 1.0 
                # print ('temp_logit',temp_logit)
                logit_list.append(torch.tensor(temp_logit, device=self.model_device))
                # probs = np.array([0.5,0.5])
                
                # return 'null', np.array([0,0]) , 0.0

            # prompt = f"[INST]What is the sentiment of the following movie review sentence, and provide your prediction confidence as a percentage? \n{t}\npositive, negative[/INST]"
            # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model_device)
            
            # # Get model predictions as text
            # generate_args = {
            #         "input_ids": inputs['input_ids'],
            #         "attention_mask": inputs['attention_mask'],
            #         "do_sample": True,  # enable sampling
            #         "top_k": 40,  # top-k sampling
            #         "top_p": 0.92,  # nucleus sampling probability
            #         "temperature": 0.7,  # sampling temperature
            #         "max_new_tokens":200
            #     }
        
            # with torch.no_grad():
            #     outputs = model.generate(**inputs, max_new_tokens=200,temperature = 0.01 ,top_k= 1)
            
            #     # full_text  = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # # Display the generated text from the model
            # prompt_length = len(inputs['input_ids'][0])
            # generated_tokens = outputs[0][prompt_length:]
            # generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            # print("Generated Text:", generated_text)

            # # Use regex to extract sentiment and confidence
            # match_sentiment = re.search(r'positive|POSITIVE|Positive|negative|NEGATIVE|Negative', generated_text)
            # match_confidence = re.search(r"[-+]?\d*\.?\d+", generated_text)
            # print ('match_sentiment',match_sentiment,'match_confidence',match_confidence)



            # if match_sentiment and match_confidence:
            #     sentiment_result = match_sentiment.group(0).lower()
            #     confidence_result = float(match_confidence.group(0)) /100   # Append '%'
                
                
            #     if sentiment_result == 'positive':
            #         probs = np.array([1-confidence_result,confidence_result])
            #     else:
            #         probs = np.array([confidence_result,1-confidence_result])

            #     concatenated_logits = torch.tensor([probs[1], probs[0],0.0], device=model_device)
            #     # print ('softmax_probabilities',softmax_probabilities) 
            #     logit_list.append(concatenated_logits)


            #     # return sentiment_result, probs ,confidence_result
            # else: 
            #     temp_logit = torch.zeros(num_classes+1, dtype=torch.float) 
            #     # print ('temp_logit',temp_logit)
            #     temp_logit[-1] = 1.0 
            #     # print ('temp_logit',temp_logit)
            #     logit_list.append(torch.tensor(temp_logit, device=model_device))
            #     # probs = np.array([0.5,0.5])
                
            #     # return 'null', np.array([0,0]) , 0.0
             
            # continue

        # After you're done collecting all logits in logit_list
        # Stack the list of tensors to create a single tensor
        print ('logit list:',logit_list)
        logit_tensor = torch.stack(logit_list)
        print('logit_tensor:', logit_tensor)
        return logit_tensor




model_wrapper = HuggingFaceLLMWrapper(model, tokenizer)





print ('dataset_class_1',len(dataset_class_1_t))
print ('dataset_class_0',len(dataset_class_0_t))
# sys.exit()


# from textattack.transformations import Transformation

# class PromptFocusedTransformation(Transformation):
#     def __init__(self, ...):
#         # Initialization code


transformation = WordSwapWordNet() #WordSwapEmbedding(max_candidates=50)


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






# Define constraints (optional but recommended to refine the search space)
stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
constraints = [RepeatModification()]
 

# Define the search method

# Define the goal function
from textattack.goal_functions import ClassificationGoalFunction


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
        



goal_function = CustomConfidenceGoalFunction(model_wrapper,query_budget=500)
# goal_function = UntargetedClassification(model_wrapper)


"""
Beam Search
===============

"""

import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class BeamSearchLocal(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=1):
        self.beam_width = beam_width
        self.previeous_beam = [] 

    def _get_index_order(self, initial_text, max_len=-1):
        len_text, indices_to_order = self.get_indices_to_order(initial_text)
        index_order = indices_to_order
        np.random.shuffle(index_order)
        search_over = False
        return index_order, search_over

    def perform_search(self, initial_result):
        # beam = [initial_result.attacked_text] 
        # self.previeous_beam = beam
        # initial_beam = initial_result
        # best_result = initial_result
        # counter = 0


        # raw beam search with stopping if beam is equal, seems to take ages
        # while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #     potential_next_beam = []
        #     for text in beam:
        #         transformations = self.get_transformations(
        #             text, original_text=initial_result.attacked_text
        #         )
        #         potential_next_beam += transformations

        #     if len(potential_next_beam) == 0:
        #         # If we did not find any possible perturbations, give up.
        #         return best_result
        #     results, search_over = self.get_goal_results(potential_next_beam)
        #     scores = np.array([r.score for r in results])
        #     best_result = results[scores.argmax()]
        #     if search_over:
        #         return best_result

        #     # Refill the beam. This works by sorting the scores
        #     # in descending order and filling the beam from there.
        #     best_indices = (-scores).argsort()[: self.beam_width]
        #     beam = [potential_next_beam[i] for i in best_indices]

        #     if self.previeous_beam == potential_next_beam:
        #         break

        # return best_result
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None
        result_path = []
        while i < len(index_order) and not search_over:
            if i > 6:
                break
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            print ('results',results)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                result_path.append(cur_result)
                cur_result = results[0] 
                 
            else:
                continue
        result_path.append(cur_result)
        if len(result_path)>2:
            print ('res path',result_path)
            scores_path = [result.score for result in result_path]
            print ('scores_path',scores_path)
            if 0.5 in scores_path:
                sys.exit()
        
        return cur_result
        # while best_result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
        #     # Generate transformations for the current best text only.
        #     current_text = best_result.attacked_text
        #     transformations = self.get_transformations(current_text, original_text=initial_result.attacked_text,indices_to_modify=[index_order])

        #     # If no transformations were found, give up.
        #     if not transformations:
        #         return best_result

        #     print('Transformations:', transformations)

        #     # Evaluate all potential transformations from the current text.
        #     results, search_over = self.get_goal_results(transformations)
        #     if search_over:
        #         return best_result

        #     # Gather the scores for each result.
        #     scores = np.array([result.score for result in results])
        #     print('Scores:', scores)

        #     # Find the best transformation result (maximize or minimize the score).
        #     max_score_index = np.argmax(scores)

        #     # Compare the best transformation's score with the current best score.
        #     if scores[max_score_index] > best_result.score:
        #         # Move to the best transformation found (climb the hill).
        #         best_result = results[max_score_index]
        #         print(f"Updated best result with a score of {best_result.score}")
        #     else:
        #         # No transformation is better than the current one.
        #         print("No better transformations found, stopping search.")
        #         break

        #     # Optional: Constrain the number of iterations to prevent infinite loops.
        #     # counter += 1
        #     # if counter == max_iterations:  # Define max_iterations as needed
        #     #     print("Maximum iterations reached, stopping search.")
        #     #     break
        # return best_result


    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]


search_method_class_0 = BeamSearchLocal()#   GreedyWordSwapWIR(wir_method="delete")


search_method_class_1 = AlzantotGeneticAlgorithm(pop_size=60, max_iters=20, post_crossover_check=False)

# Create the attack
greedy_attack = Attack(goal_function, constraints, transformation, search_method_class_0)
# genetic_attack = Attack(goal_function, constraints, transformation, search_method_class_1)




from textattack.datasets import Dataset

class SimpleDataset(Dataset):
    def __init__(self, examples, label_names=None):
        """
        args:
            examples: list of tuples where each tuple is (text, label)
            label_names: list of strings representing label names (optional)
        """
        self.examples = examples
        self.label_names = label_names
        self.shuffled = False  # Set to True if you shuffle the examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Convert filtered datasets into TextAttack Dataset format
dataset_class_ta =  SimpleDataset(dataset_class) 

import os
# Define the folder where logs should be saved
test_folder = 'llm_log_folder'  # Replace 'desired_log_folder' with your folder name

# Ensure the directory exists; if not, create it
if not os.path.exists(test_folder):
    os.makedirs(test_folder)


# Define attack arguments
attack_args = AttackArgs(
    num_examples=500,
    log_to_csv=os.path.join(test_folder, "log_greedy_500_trying_to_ignore_null.csv"),  # Adjusted to save in test_folder
    checkpoint_interval=1000,
    checkpoint_dir="checkpoints",
    disable_stdout=False,
    parallel=False,
    num_workers_per_device=8,
)

# Run attack for class 0
attacker_greedy_class_0 = Attacker(greedy_attack, dataset_class_ta, attack_args)
attacker_greedy_class_0.attack_dataset()

 
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

# # Run attack for class 0
# attacker_greedy_class_1 = Attacker(greedy_attack, dataset_class_1_ta, attack_args)
# attacker_greedy_class_1.attack_dataset()

# # Define attack arguments
# attack_args = AttackArgs(
#     num_examples=1000,
#     log_to_csv=os.path.join(test_folder, "log_genetic_class0.csv"),  # Path adjusted
#     checkpoint_interval=1000,
#     checkpoint_dir="checkpoints",
#     disable_stdout=False,
#     parallel=False,
#     num_workers_per_device=8,
# )
# # Optionally, run attack for class 1 with the same or a different configuration
# attacker_genetic_class_0 = Attacker(genetic_attack, dataset_class_0_ta, attack_args)
# attacker_genetic_class_0.attack_dataset()