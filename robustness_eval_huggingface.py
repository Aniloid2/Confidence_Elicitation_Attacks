
import os 

import numpy as np 
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import re
import matplotlib.pyplot as plt 
from numpy.random import dirichlet  
import json 

import random

# os.environ['HF_DATASETS_CACHE'] =  '/mnt/hdd/brian/datasets'

from src.arg_parser.arg_config import get_args
args = get_args()  
from src.arg_parser.set_cache import set_huggingface_cache
set_huggingface_cache(args)

# os.environ['HF_DATASETS_CACHE'] =  '/mnt/hdd/brian/datasets'
# print (os.environ['HF_DATASETS_CACHE']) 

# cache_dir = args.cache_transformers # "/mnt/hdd/brian/hub", '/mnt/hdd/brian/' 
# os.environ['HF_DATASETS_CACHE'] = cache_dir + 'datasets'
# print (os.environ['HF_DATASETS_CACHE'])
 
from src.utils.shared.misc import environment_setup
args = environment_setup(args) 
# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def calculate_ece(true_labels, pred_probs, n_bins=10):
#     """
#     Calculate Expected Calibration Error (ECE).
#     """
#     true_labels = np.array(true_labels)
#     pred_probs = np.array(pred_probs)
#     bin_boundaries = np.linspace(0, 1, n_bins + 1)
#     ece = 0

#     for i in range(n_bins):
#         bin_mask = (pred_probs >= bin_boundaries[i]) & (pred_probs < bin_boundaries[i + 1])
#         if np.any(bin_mask):
#             bin_prob_true = np.mean(true_labels[bin_mask])
#             bin_prob_pred = np.mean(pred_probs[bin_mask])
#             bin_size = np.sum(bin_mask)
#             ece += (bin_size / len(true_labels)) * np.abs(bin_prob_true - bin_prob_pred)

#     return ece



# def predict_sentiment_2(text):

#     # insert prompt here
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits[:, -1, :]
#     label_0_prob = logits[:, tokenizer.vocab['▁negative']].item()
#     label_1_prob = logits[:, tokenizer.vocab['▁positive']].item()
#     probs_normalized = torch.softmax(torch.tensor([label_0_prob, label_1_prob]), dim=0)
#     sentiment = 1 if label_1_prob > label_0_prob else 0
#     sentiment_ret = 'positive' if sentiment == 1 else 'negative'
#     print ('entiment, probs_normalized.numpy(),max(probs_normalized)',sentiment, probs_normalized.numpy(),max(probs_normalized))
#     return sentiment_ret, probs_normalized.numpy(),max(probs_normalized)  # Return probabilities along with sentiment


# def predict_sentiment_and_confidence(text):
    

#     prompt = f"""[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') 
#     the following sentiment. Give ONLY the guess and verbal confidence, no other words or
#     explanation. For example:\n\nGuess: <most likely guess, either positive or negative; not
#     a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just
#     the confidence!>\n\nThe sentiment is: ${text} Sentiment & confidence:[/INST]"""


#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
     
#     # Get model predictions as text
#     generate_args = {
#             "input_ids": inputs['input_ids'],
#             "attention_mask": inputs['attention_mask'],
#             "do_sample": True,  # enable sampling
#             "top_k": 40,  # top-k sampling
#             "top_p": 0.92,  # nucleus sampling probability
#             "temperature": 0.7,  # sampling temperature
#             "max_new_tokens":200
#         }
 
#     with torch.no_grad():
#         # outputs = model.generate(**inputs, max_new_tokens=200,temperature = 0.7 ,top_k= 1)
#         outputs = model.generate(**generate_args)
     
#         # full_text  = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Display the generated text from the model
#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
#     print("Generated Text:", generated_text)

#     # Use regex to extract sentiment and confidence
#     match_sentiment = re.search(r'positive|POSITIVE|Positive|negative|NEGATIVE|Negative', generated_text)
#     match_confidence = re.search(r"[-+]?\d*\.?\d+", generated_text)
#     print ('match_sentiment',match_sentiment,'match_confidence',match_confidence)



#     if match_sentiment and match_confidence:
#         sentiment_result = match_sentiment.group(0).lower()
#         confidence_result = float(match_confidence.group(0)) /100   # Append '%'
        
        
#         if sentiment_result == 'positive':
#             probs = np.array([1-confidence_result,confidence_result])
#         else:
#             probs = np.array([confidence_result,1-confidence_result])
#         return sentiment_result, probs ,confidence_result
#     else: 
#         return 'null', np.array([0,0]) , 0.0


# def scale_vector_by_confidence(vector, confidence, margin=0.02):
#         # Normalize the confidence value to a range of 0 to 1
#         print ('vector',vector)
#         normalized_confidence = confidence #/ 100.0

#         # Create an almost equalized vector with a slight margin
#         half_margin = margin / 2.0
#         equalized_vector = [0.5 + half_margin if i == 0 else 0.5 - half_margin for i in range(len(vector))]

#         # Interpolate between the equalized vector and the original vector
#         scaled_vector = [
#             (1 - normalized_confidence) * equalized_value + normalized_confidence * original_value
#             for equalized_value, original_value in zip(equalized_vector, vector)
#         ]

#         # Ensure the scaled vector sums to 1 by re-normalizing
#         sum_scaled_vector = sum(scaled_vector)
#         normalized_scaled_vector = [value / sum_scaled_vector for value in scaled_vector]

#         return normalized_scaled_vector

# def predict_sentiment_and_verbal_confidence_k_pred_avg(text,task, expected_prediction):
#     k_pred = 10
#     if task not in ['sst2', 'ag_news']:
#         raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

#     if task == 'sst2':
#         guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
#         class_number  = 2
#         label_list = label_names# ['positive', 'negative']
#     elif task == 'ag_news':
#         guess_pattern = r'(world|business|tech|science|sports)'
#         class_number  = 4
#         label_list = ['world','business', 'tech','science','tech/sci', 'sports']

#     task_dictionary_counts = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
#     task_dictionary_confidences = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
        



#     if task == 'sst2':
#         prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a coma, for example [Negative, Positive, Positive, Negative ...]>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""
    
#     elif task == 'ag_news':
        
#         options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
        
#         prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses!>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""
        
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

#     # Get model predictions as text
#     generate_args = {
#         "input_ids": inputs['input_ids'],
#         "attention_mask": inputs['attention_mask'],
#         "do_sample": True,  # enable sampling
#         "top_k": 40,  # top-k sampling
#         "top_p": 0.92,  # nucleus sampling probability
#         "temperature": 0.7,  # sampling temperature
#         "max_new_tokens": 200,
#         'pad_token_id': tokenizer.eos_token_id
#     }

#     with torch.no_grad():
#         outputs = model.generate(**generate_args)

#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#     print ('prompt', prompt)
#     print("Generated Prediction Text:", generated_text) 
#     # Use regex to extract the guess (sentiment or category) and verbal confidence
#     # match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
#     # if match_guess:
#     #     guess_result = match_guess.group(0).lower()
#     #     if guess_result not in label_list:
#     #         guess_result = 'null'
#     # else:
#     #     probs = np.zeros(class_number+1)
#     #     probs[-1] = 1.0  # Only null has confidence 1
#     #     return 'null', probs, 1.0

#     # Extract guesses, assuming they're separated by commas and ignoring case
#     results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
#     results = [result for result in results if result in label_list]
#     # If fewer results than k_pred, fill with 'null'
#     results.extend(['null'] * (k_pred - len(results)))
#     print ('results',results,expected_prediction)
#     correct_predictions = sum(1 for sentiment in results if sentiment == expected_prediction)
#     confidence_empirical = (correct_predictions / len(results)) * 100
#     # Manually count occurrences

#     # sentiment_counts = {k:0 for k in label_list}
#     # sentiment_counts['null'] = 0
#     # print ('count',sentiment_counts)
#     # for guess in results:
#     #     if guess in sentiment_counts:
#     #         sentiment_counts[guess] += 1
#     #     else:
#     #         sentiment_counts[guess] = 1

#     # # percentages = {key: (value / k_pred) * 100 for key, value in sentiment_counts.items()} 
#     # # print ('percentages',percentages)
 
#     # sentiment_confidences = {k:0 for k in label_list}
#     # sentiment_confidences['null'] = 0
#     # for sentiment, number_of_results in sentiment_counts.items():
#     #     sentiment_confidences[sentiment] = (number_of_results / len(results))

 
#     # print(f"Results for '{text}':")
#     # print(f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Null: {sentiment_counts['null']}")
#     # # print(f"Average model confidence: {average_confidence}%")
#     # print(f"Empirical confidence: {confidence_empirical}%")
#     # max_class = max(sentiment_counts, key=sentiment_counts.get)
#     # print ('max_class',max_class,expected_prediction)

#     for sentiment in results:
#         task_dictionary_counts[task][sentiment] += 1
    
#     # sentiment_confidences = { 'positive': 0, 'negative': 0, 'null': 0 }
#     for sentiment, number_of_results in task_dictionary_counts[task].items():
#         task_dictionary_confidences[task][sentiment] = (number_of_results / len(results))

#     # average_confidence = sum(confidence for _, confidence in results) / len(results)

#     print(f"Results for '{text}':")
#     print(f"Counter: {task_dictionary_counts[task]}")
#     # print(f"Average model confidence: {average_confidence}%")
#     print(f"Empirical confidence: {confidence_empirical}%")
#     max_class = max(task_dictionary_counts[task], key=task_dictionary_counts[task].get)
#     print ('max_class',max_class,expected_prediction)
#     print ('empricial:',expected_prediction,confidence_empirical,task_dictionary_confidences[task])


#     guess_result = max_class
#     confidence_result = confidence_empirical/100
#     if guess_result != 'null':
#         if task == 'sst2':
#             probs = np.array([task_dictionary_confidences[task]['negative'],
#                 task_dictionary_confidences[task]['positive'],
#                 task_dictionary_confidences[task]['null']])
#             # if guess_result == 'positive':
#                 # probs = np.array([task_dictionary_confidences[self.task]['negative'],task_dictionary_confidences[self.task]['positive'],task_dictionary_confidences[self.task]['null']])
#                 # probs = np.array([1-confidence_result,confidence_result,0.0])
#             # else:
#             #     probs = np.array([task_dictionary_confidences[self.task]['negative'],task_dictionary_confidences[self.task]['positive'],task_dictionary_confidences[self.task]['null']])
#                 # probs = np.array([confidence_result,1-confidence_result,0.0])
#             return guess_result, probs ,confidence_result
#         elif task == 'ag_news':
#             # granularity = 4 # how many confidence labels do we have in confidence elicitation
#             # ignore_classes  = 1 # ignore null and the current class
#             # other_level_candidates = class_number
            
#             # confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
#             # confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
#             probs = np.array([task_dictionary_confidences[task]['world'],
#             task_dictionary_confidences[task]['sports'],
#             task_dictionary_confidences[task]['business'],
#             task_dictionary_confidences[task]['sci/tech'],
#             task_dictionary_confidences[task]['null'],
#             ])
#             print ('probs',probs)
#             # if guess_result == 'world':
#             #     probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
#             # elif guess_result == 'sports' :
#             #     probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
#             # elif guess_result == 'business':
#             #     probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
#             # if guess_result == 'tech' or 'science' or 'tech/science':
#             #     guess_result = 'tech/sci'
#                 # probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
#             return guess_result, probs ,confidence_result
#     else:        
#         probs = np.zeros(class_number+1)
#         probs[-1] = 1.0  # Only null has confidence 1
#         return 'null', probs, 1.0
        
    
# def predict_sentiment_and_verbal_confidence_2step_k_pred_avg(text,task, expected_prediction):
#         k_pred = 20
#         if task not in ['sst2', 'ag_news']:
#             raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

#         if task == 'sst2':
#             guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
#             class_number  = 2
#             label_list = label_names# ['positive', 'negative']
#         elif task == 'ag_news':
#             guess_pattern = r'(world|business|tech|science|sports)'
#             class_number  = 4
#             label_list = label_names# ['positive', 'negative'] #label_list = ['world','business', 'tech','science','tech/sci', 'sports']

#         task_dictionary_counts = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
#         task_dictionary_confidences = {'sst2':{ 'negative': 0, 'positive': 0, 'null': 0 }, 'ag_news':{'world':0,'sports':0,'business':0,'sci/tech':0,'null':0}}
            



#         if task == 'sst2':
#             prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a coma, for example [Negative, Positive, Positive, Negative ...]>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""
        
#         elif task == 'ag_news':
            
#             options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
            
#             prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses! Separated by a coma, for example [Sport, Business, Sport, Politics, Sci/Tech ...]>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""
            
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

#         # Get model predictions as text
#         generate_args = {
#             "input_ids": inputs['input_ids'],
#             "attention_mask": inputs['attention_mask'],
#             "do_sample": True,  # enable sampling
#             "top_k": 40,  # top-k sampling
#             "top_p": 0.92,  # nucleus sampling probability
#             "temperature": 0.7,  # sampling temperature
#             "max_new_tokens": 200,
#             'pad_token_id': tokenizer.eos_token_id
#         }

#         with torch.no_grad():
#             outputs = model.generate(**generate_args)

#         prompt_length = len(inputs['input_ids'][0])
#         generated_tokens = outputs[0][prompt_length:]
#         generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#         print ('prompt', prompt)
#         print("Generated Prediction Text:", generated_text) 
        

#         # Extract guesses, assuming they're separated by commas and ignoring case
#         results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
#         results = [result for result in results if result in label_list]
#         # If fewer results than k_pred, fill with 'null'
#         results.extend(['null'] * (k_pred - len(results)))
#         print ('results',results,expected_prediction)
#         correct_predictions = sum(1 for sentiment in results if sentiment == expected_prediction)
#         confidence_empirical = (correct_predictions / len(results)) * 100
        

#         for sentiment in results:
#             task_dictionary_counts[task][sentiment] += 1
        
#         # sentiment_confidences = { 'positive': 0, 'negative': 0, 'null': 0 }
#         for sentiment, number_of_results in task_dictionary_counts[task].items():
#             task_dictionary_confidences[task][sentiment] = (number_of_results / len(results))

#         # average_confidence = sum(confidence for _, confidence in results) / len(results)

#         print(f"Results for '{text}':")
#         print(f"Counter: {task_dictionary_counts[task]}")
#         # print(f"Average model confidence: {average_confidence}%")
#         print(f"Empirical confidence: {confidence_empirical}%")
#         max_class = max(task_dictionary_counts[task], key=task_dictionary_counts[task].get)
#         print ('max_class',max_class,expected_prediction)
#         print ('empricial:',expected_prediction,confidence_empirical,task_dictionary_confidences[task])


#         # if null majority, we return null as main
        
#         # if max_class != 'null':
#         #     guess_result = max_class
#         #     if guess_result not in label_list:
#         #         guess_result = 'null'
#         # else:
#         #     probs = np.zeros(class_number+1)
#         #     probs[-1] = 1.0  # Only null has confidence 1
#         #     return 'null', probs, 1.0

#         guess_result = max_class
#         guesses_output = results
        
#         if task == 'sst2':
#             # confidence_prompt = f"""{start_prompt_header}Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[self.confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [Low, Highest, Medium, Lowest, High, High ...]> Confidences:{end_prompt_footer}"""
#             confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [Low, Highest, Medium, Lowest, High, High ...]> Confidences:{end_prompt_footer}"""
        
#         elif task == 'ag_news':
#             confidence_prompt = f"""{start_prompt_header}Provide your {k_pred} best guesses for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either {options}; not a complete sentence, just the guesses!>\n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidencees! Separated by a coma, for example [Low, Highest, Medium, Lowest, High, High ...]> Confidences:{end_prompt_footer}"""


#         print ('confidence_prompt',confidence_prompt)
#         inputs = tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
#         generate_args = {
#             "input_ids": inputs['input_ids'],
#             "attention_mask": inputs['attention_mask'],
#             "do_sample": True,
#             "top_k": 40,
#             "top_p": 0.92,
#             "temperature": 0.7,
#             "max_new_tokens": 200,
#             'pad_token_id': tokenizer.eos_token_id
#         }
#         with torch.no_grad():
#             outputs = model.generate(**generate_args)

#         prompt_length = len(inputs['input_ids'][0])
#         generated_tokens = outputs[0][prompt_length:]
#         generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#         print("Generated Confidence Text:", generated_text) 
#         # sorted_confidence_options = sorted(CONFIDENCE_LEVELS[self.confidence_type], key=len, reverse=True) 
#         # print ('sorted_confidence_options',sorted_confidence_options)


#         confidence_options = '|'.join(CONFIDENCE_LEVELS[confidence_type]) 
#         confidence_guesses = re.findall(confidence_options, generated_text, flags=re.IGNORECASE)
#         confidence_guesses =  [match.lower() for match in confidence_guesses]
#         # confidence_guesses = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
#         print ('confidence_guesses',confidence_guesses)
#         confidence_list = CONFIDENCE_LEVELS[confidence_type]
#         print ('confidence_list',confidence_list)
#         confidence_results = [result for result in confidence_guesses if result in confidence_list]
#         # confidence_results=[]
#         print ('confidence_results',confidence_results)
#         # If fewer results than k_pred, fill with 'null'
#         confidence_results.extend(['null'] * (k_pred - len(confidence_results)))
        
        
#         confidence_map = CONFIDENCE_MAP[confidence_type]

#         # can count number of occurances or other stuff when computing the confidence 
#         # confidence_numerical_results = [confidence_map[result] for result in confidence_results  min(list(confidence_map.values())) if result == 'null' ]
        
#         confidence_numerical_results = [
#             confidence_map[result] if result != 'null' else min(confidence_map.values()) 
#             for result in confidence_results
#         ] 
#         # confidence_numerical_results.extend(([min(list(confidence_map.values()))]) * (k_pred - len(confidence_results)))
         
#         # confidence_numerical_results.extend([min(list(confidence_map.values()))] * (k_pred - len(confidence_results)))
        
#         print ('confidence_numerical_results',confidence_numerical_results) 
#         print ('guesses_output',guesses_output)

#         counts = task_dictionary_counts[task]   
#         weighted_counts = { 'negative': 0.0,'positive': 0.0,'null':0.0} 
#         for sentiment, confidence in zip(guesses_output, confidence_numerical_results):
#             print ('confidence',confidence)
#             if confidence:
#                 weighted_counts[sentiment] += confidence
#             else:
#                 weighted_counts[sentiment] += 1

#         print ('weighted_counts',weighted_counts) 
#         alpha_prior = 1.0

#         alpha = { 'negative': weighted_counts['negative'] + alpha_prior,'positive': weighted_counts['positive'] + alpha_prior,'null': weighted_counts['null'] + alpha_prior}

#         # alpha = { 'negative': counts['negative'] + alpha_prior,'positive': counts['positive'] + alpha_prior,'null': counts['null'] + alpha_prior}

#         alpha_total = sum(alpha.values())
#         from utils.shared.plotting import ternary_plot, ternary_mean_plot
#         sample_size=1000
#         alpha_values = list(alpha.values())
#         dirichlet_distribution = dirichlet(alpha_values, size=sample_size)
#         # print ('dirichlet_distribution',dirichlet_distribution)
#         # Normalized probabilities from the Dirichlet distribution
#         samples_ternary = [(p[0], p[1], p[2]) for p in dirichlet_distribution]  
#         # print ('dirichlet_distribution',samples_ternary)
#         empirical_means = np.mean(dirichlet_distribution, axis=0)
#         empirical_means_ternary = (empirical_means[0], empirical_means[1], empirical_means[2])
#         print ('empirical_means',empirical_means) 
#         # if current_sample == 1:
#         #     infrence_step +=1
#         #     ternary_plot_file = os.path.join(test_folder, f'dirichlet_cs{current_sample}_is{infrence_step}_a({alpha_values})_n{str(sample_size)}')
#         #     ternary_mean_plot(samples_ternary,alpha_values,empirical_means_ternary,ternary_plot_file)

#         probabilities = dirichlet_distribution[0]

#         def dirichlet_variance(alpha):
#             alpha_0 = sum(alpha)
#             variances = [(alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1)) for alpha_i in alpha]
#             return variances

#         alpha_vector = [ alpha['negative'],alpha['positive'],alpha['null']]
#         second_order_uncertainty = dirichlet_variance(alpha_vector)

#         print("Counts:", counts)
#         print("Numerical Confidences:", confidence_numerical_results)
#         print("Weighted Counts:", weighted_counts)
#         print("Alpha Vector:", alpha_vector)
#         print("Probabilities:", probabilities)
#         print("Second Order Uncertainty:", second_order_uncertainty)
#         probs = probabilities
#         confidence_result = max(probabilities)
#         print ('guess_result',guess_result, 'probs',probs,'confidence_result',confidence_result   )
        
#         return guess_result, probs ,confidence_result
    
# import json
# def predict_sentiment_and_verbal_confidence_2step_k_pred_avg(datapoint,label_names=None):
#     k_pred = 1
#     # if task not in ['sst2', 'ag_news', 'popQA']:
#     #     raise ValueError("Unsupported task. Please choose 'sst2', 'ag_news', or 'popQA'.")

#     # Custom processing for PopQA
#     if task == 'popQA': 
#         data_point = {'question':datapoint[0], 'possible_answers':datapoint[1]}
#         text = data_point['question']

#         possible_answers = json.loads(data_point['possible_answers'])
#         label_names = possible_answers['correct'] + possible_answers['incorrect']
#         print ('label_names', label_names)

#         expected_prediction = possible_answers['correct']
#         incorrect_answers = possible_answers['incorrect']
#         guess_pattern = '|'.join(label_names)
#         class_number = len(label_names)

#         # Create task-specific dictionaries for PopQA
#         task_dictionary_counts_correct = {'popQA':{label: 0 for label in expected_prediction}}
#         task_dictionary_counts_incorrect = {'popQA':{label: 0 for label in incorrect_answers}}

#         task_dictionary_confidences = {'popQA':{label: 0 for label in label_names}}
#         task_dictionary_confidences['null'] = 0

#         task_dictionary_counts = {'popQA':{label: 0 for label in label_names}}
#         task_dictionary_counts['null'] = 0

#         # Create a sample list for examples in the prompt
#         sampled_answers = random.sample(label_names, min(5, len(label_names)))
#         sampled_answers_str = ', '.join(label_names)
#         sampled_confidences_str =', '.join(CONFIDENCE_LEVELS[confidence_type])
#         print ('sampled_confidences_str',sampled_confidences_str)
#         prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following question. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses; not a complete sentence, just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: ${data_point['question']} Guesses:{end_prompt_footer}"""
#     else:
#         print ('datapoint',datapoint)
#         if task == 'sst2':
#             guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
#             class_number = 2
#             label_list = label_names  # ['positive', 'negative']
#             label_index = datapoint[1]
#         elif task == 'ag_news':
#             guess_pattern = r'(world|business|tech|science|sports)'
#             class_number = 4
#             label_list = label_names  # ['positive', 'negative']
#             label_index = datapoint[1]
#         elif task == 'strategyQA':
#             guess_pattern = r'(true|TRUE|True|false|FALSE|False)'
#             class_number = 2
#             label_list = ['false', 'true']
#             label_index = 1 if datapoint[1] else 0

#         print ('label_index',label_index) 
#         text = datapoint[0]
#         expected_prediction = [label_list[label_index]]
#         print ('sets',set(label_list) , set([label_list[label_index]])) 
#         incorrect_answers = list (set(label_list) -  set([label_list[label_index]])) 
#         print ('text', text)
#         print ('expec', expected_prediction) 
#         print ('expected_prediction',expected_prediction)
#         print ('incorrect_answers',incorrect_answers)
#         task_dictionary_counts = {task: {label: 0 for label in label_list}}
#         task_dictionary_counts[task]['null'] = 0

#         task_dictionary_confidences = {task: {label: 0 for label in label_list}}
#         task_dictionary_confidences[task]['null'] = 0
#         # Create task-specific dictionaries for PopQA
#         task_dictionary_counts_correct = {task :{label: 0 for label in expected_prediction}}
#         task_dictionary_counts_incorrect = {task:{label: 0 for label in incorrect_answers}}

#         sampled_confidences_str =', '.join(CONFIDENCE_LEVELS[confidence_type])
#         print ('sampled_confidences_str',sampled_confidences_str)
        
#         if task == 'sst2':
#             prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a comma, for example [Negative, Positive, Positive, Negative ...]>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""

#         elif task == 'ag_news':
#             options = 'world, business, tech, science, sports'  # we separate tech/science into two different prediction categories, but treat them as one label
#             prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses! Separated by a comma, for example [Sport, Business, Sport, Politics, Sci/Tech ...]>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""
#         elif task == 'strategyQA':
#             # prompt = f"""{start_prompt_header}Provide your {k_pred} best guess for the following text (false, true). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either true orfalse; not a complete sentence, just the guesses! Separated by a comma, for example [True, False, False, True ...]>\n\nThe text is:${text} Guesses:{end_prompt_footer}"""
#             # prompt = f"""{start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {k_pred} guesses, for example [True, False, False, True ...]. Only output your answer nothing else!\n\nStatement: {text} Answer:{end_prompt_footer}"""
#             # prompt = f"""{start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {k_pred} guesses. Only output your answer nothing else!\n\nStatement: {text} Answer:{end_prompt_footer}"""
            
#             prompt = f"""{start_prompt_header}You are a factual question answering model, Is the following statement true or false?\n\nStatement: {text} Answer:{end_prompt_footer}"""
    
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

#     generate_args = {
#         "input_ids": inputs['input_ids'],
#         "attention_mask": inputs['attention_mask'],
#         "do_sample": True,  # enable sampling
#         "top_k": 40,  # top-k sampling
#         "top_p": 0.92,  # nucleus sampling probability
#         "temperature": 0.7,  # sampling temperature
#         "max_new_tokens": 200,
#         'pad_token_id': tokenizer.eos_token_id
#     }

#     with torch.no_grad():
#         outputs = model.generate(**generate_args)

#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#     print('prompt', prompt)
#     print("Generated Prediction Text:", generated_text) 


#     # Regex to find 'true' or 'false', case-insensitive, ensuring full word match
#     pattern = re.compile(r'\btrue\b|\bfalse\b', re.IGNORECASE)
#     # Find all matches in the text
#     matches = pattern.findall(generated_text)
#     # Convert all matches to lowercase (optional, for consistency)
#     results = [match.lower() for match in matches]



#     # # Extract guesses, assuming they're separated by commas and ignoring case
#     # results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
#     # results = [result for result in results if result in label_list]# else 'null' for result in results]
#     # # If fewer results than k_pred, fill with 'null'
#     results.extend(['null'] * (k_pred - len(results)))



#     print('results', results, expected_prediction)
#     correct_predictions = sum(1 for sentiment in results if sentiment in expected_prediction)
#     confidence_empirical = (correct_predictions / len(results)) * 100
#     print('correct_predictions',correct_predictions)
#     for result in results:
#         if result in task_dictionary_counts[task]:
#             task_dictionary_counts[task][result] += 1
#         else:
#             task_dictionary_counts[task]['null'] +=1
        
#     for sentiment, number_of_results in task_dictionary_counts[task].items():
#         task_dictionary_confidences[sentiment] = (number_of_results / len(results))

    
    

#     for result in results:
#         if result in task_dictionary_counts_correct[task]:
#             task_dictionary_counts_correct[task][result]+=1
#         elif result in task_dictionary_counts_incorrect[task]:
#             task_dictionary_counts_incorrect[task][result]+=1

#     print(f"Results for '{text}':")
#     print(f"Counter: {task_dictionary_counts}")
#     print(f"Empirical confidence: {confidence_empirical}%")
#     max_class = max(task_dictionary_counts[task], key=task_dictionary_counts[task].get)
#     print('max_class', max_class, expected_prediction)
#     print ('task_dictionary_counts_correct[task]',task_dictionary_counts_correct[task])
#     print ('task_dictionary_counts_incorrect[task]', task_dictionary_counts_incorrect[task])

#     guess_result = max_class
#     guesses_output = results
    
#     if task == 'sst2':
#         # confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [{sampled_confidences_str} ...]> Confidences:{end_prompt_footer}"""
#         confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{end_prompt_footer}"""
    
#     elif task == 'ag_news':
#         confidence_prompt = f"""{start_prompt_header}Provide your {k_pred} best guesses for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: \n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences:  Confidences:{end_prompt_footer}"""
#     elif task == 'popQA':
#         confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${data_point['question']}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [{sampled_confidences_str} ...]> Confidences:{end_prompt_footer}"""
#     elif task == 'strategyQA':
#         # confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (false, true). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either true or false; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{end_prompt_footer}"""
#         # confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {k_pred} guesses. Only output your answer nothing else!\n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{end_prompt_footer}"""
#         # confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {k_pred} guesses. \n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. \n\nFor example:\n\Confidences: <the confidences, from either {CONFIDENCE_LEVELS[confidence_type]} that your guesses are correct, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{end_prompt_footer}"""
#         confidence_prompt = f"""{start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {k_pred} guess.\n\nStatement: {text}$. The guess were: {guesses_output}, given the guess provide the verbal confidences that your guess us correct. \n\nFor example:\n\Confidence: <the confidence, from either {CONFIDENCE_LEVELS[confidence_type]} that your guess is correct, for example [{sampled_confidences_str} ...]; just the confidence!> Confidence:{end_prompt_footer}"""
    
#     print('confidence_prompt', confidence_prompt)
#     inputs = tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
#     generate_args = {
#         "input_ids": inputs['input_ids'],
#         "attention_mask": inputs['attention_mask'],
#         "do_sample": True,
#         "top_k": 40,
#         "top_p": 0.92,
#         "temperature": 0.7,
#         "max_new_tokens": 200,
#         'pad_token_id': tokenizer.eos_token_id
#     }

#     with torch.no_grad():
#         outputs = model.generate(**generate_args)

#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#     print("Generated Confidence Text:", generated_text) 

#     confidence_options = '|'.join(CONFIDENCE_LEVELS[confidence_type]) 
#     confidence_guesses = re.findall(confidence_options, generated_text, flags=re.IGNORECASE)
#     confidence_guesses = [match.lower() for match in confidence_guesses]
#     print('confidence_guesses', confidence_guesses)
#     confidence_list = CONFIDENCE_LEVELS[confidence_type]
#     print('confidence_list', confidence_list)
#     confidence_results = [result for result in confidence_guesses if result in confidence_list]
#     confidence_results.extend(['null'] * (k_pred - len(confidence_results)))
#     confidence_map = CONFIDENCE_MAP[confidence_type]

#     confidence_numerical_results = [
#         confidence_map[result] if result != 'null' else min(confidence_map.values()) 
#         for result in confidence_results
#     ] 
#     print('confidence_numerical_results', confidence_numerical_results) 
#     print('guesses_output', guesses_output)

#     weighted_counts = {label: 0.0 for label in label_list}
#     weighted_counts['null'] = 0.0


#     for sentiment, confidence in zip(guesses_output, confidence_numerical_results):
#         if confidence:
            
#             weighted_counts[sentiment] += confidence
#         else:
#             weighted_counts[sentiment] += 1
    

#     for result, confidence in zip(guesses_output, confidence_numerical_results):
#         if result in task_dictionary_counts_correct[task]:
#             task_dictionary_counts_correct[task][result]+=confidence
#         elif result in task_dictionary_counts_incorrect[task]:
#             task_dictionary_counts_incorrect[task][result]+=confidence
    
#     weighted_counts_binary = {'incorrect':sum(task_dictionary_counts_incorrect[task].values()),
#                             'correct':sum(task_dictionary_counts_correct[task].values()),
#                             'null':weighted_counts['null']}
#     #for all datasets have a correct, incorrect and null bucket

#     def compute_dirichlet_statistics(weighted_counts, label_list):
#         print('weighted_counts', weighted_counts)

#         # You mentioned 'weighted_counts_binary' in your request, but it seems missing.
#         # Assuming it's another dictionary similar to weighted_counts. For now, we skip this.
#         # print('weighted_counts_binary', weighted_counts_binary)  

#         alpha_prior = 1.0
#         alpha = {label: weighted_counts[label] + alpha_prior for label in label_list}
#         alpha['null'] = weighted_counts['null'] + alpha_prior
#         {pos:9, neg:8, null:1}
#         alpha_values = list(alpha.values())
#         sample_size = 1000
#         dirichlet_distribution = dirichlet(alpha_values, size=sample_size)
#         samples_ternary = [(p[0], p[1], p[2]) for p in dirichlet_distribution]  
#         empirical_means = np.mean(dirichlet_distribution, axis=0)
#         empirical_means_ternary = (empirical_means[0], empirical_means[1], empirical_means[2])
#         print('empirical_means', empirical_means) 

#         def dirichlet_variance(alpha):
#             alpha_0 = sum(alpha)
#             variances = [(alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1)) for alpha_i in alpha]
#             return variances

#         alpha_vector = list(alpha.values())
#         second_order_uncertainty = dirichlet_variance(alpha_vector)
#         probabilities = dirichlet_distribution[0] 
        
#         print("Counts:", task_dictionary_counts)
#         print("Numerical Confidences:", confidence_numerical_results)
#         print("Weighted Counts:", weighted_counts)
#         print("Alpha Vector:", alpha_vector)
#         print("Probabilities:", probabilities)
#         print("Second Order Uncertainty:", second_order_uncertainty) 
#         return alpha, dirichlet_distribution, empirical_means, second_order_uncertainty, probabilities 

#     alpha, dirichlet_distribution, empirical_means, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts,weighted_counts.keys())
#     # alpha, dirichlet_distribution, empirical_means, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts_binary,weighted_counts_binary.keys())
    
#     confidence_result = max(probabilities)


#     # sys.exit()
#     return guess_result, probabilities, confidence_result

# def predict_sentiment_and_verbal_confidence_2steps(text, expected_sentiment):
#     if task not in ['sst2', 'ag_news']:
#         raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

#     if task == 'sst2':
#         guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
#         class_number  = 2
#         label_list = ['positive', 'negative']
#     elif task == 'ag_news':
#         guess_pattern = r'(world|business|tech|science|sports)'
#         class_number  = 4
#         label_list = ['world','business', 'tech','science','tech/sci', 'sports']




#     if task == 'sst2':
#         prompt = f"""{start_prompt_header}Provide your best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{end_prompt_footer}"""
    
#     elif task == 'ag_news':
        
#         options = 'world, business, tech, science, sports'  # we separeate tech/science into two different prediction categories, but treat them as one label
        
#         prompt = f"""{start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{end_prompt_footer}"""
        
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

#     # Get model predictions as text
#     generate_args = {
#         "input_ids": inputs['input_ids'],
#         "attention_mask": inputs['attention_mask'],
#         "do_sample": True,  # enable sampling
#         "top_k": 40,  # top-k sampling
#         "top_p": 0.92,  # nucleus sampling probability
#         "temperature": 0.7,  # sampling temperature
#         "max_new_tokens": 200,
#         'pad_token_id': tokenizer.eos_token_id
#     }

#     with torch.no_grad():
#         outputs = model.generate(**generate_args)

#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     print("Generated Prediction Text:", generated_text)

#     # Use regex to extract the guess (sentiment or category) and verbal confidence
#     match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)
#     if match_guess:
#         guess_result = match_guess.group(0).lower()
#         if guess_result not in label_list:
#             guess_result = 'null'
#     else:
#         probs = np.zeros(class_number+1)
#         probs[-1] = 1.0  # Only null has confidence 1
#         return 'null', probs, 1.0

#     # confidence_prompt = f"""[INST]Provide the verbal confidence that your guess is correct ('lowest', 'low', 'medium','high', 'highest') Give ONLY the verbal confidence, no other words or explanation. For example: Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!> The text is: "{text}" with guess: "{guess_result}" Confidence:[/INST]"""
#     # if task == 'sst2':
#     #     confidence_prompt = f"""{start_prompt_header}Provide your best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess: {guess_result} Provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!> Confidence:{end_prompt_footer}"""
    
#     # elif task == 'ag_news':
#     #     confidence_prompt = f"""{start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess: {guess_result} Provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!> Confidence:{end_prompt_footer}"""

#     if task == 'sst2':
#         confidence_prompt = f"""{start_prompt_header}Provide your best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess: {guess_result} Provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, either {CONFIDENCE_LEVELS[confidence_type]} that your guess is correct, without any extra commentary whatsoever; just the confidence!> Confidence:{end_prompt_footer}"""
#     elif task == 'ag_news':
#         confidence_prompt = f"""{start_prompt_header}Provide your best guess for the following news article ({options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess: {guess_result} Provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, either {CONFIDENCE_LEVELS[confidence_type]} that your guess is correct, without any extra commentary whatsoever; just the confidence!> Confidence:{end_prompt_footer}"""



#     inputs = tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)
#     generate_args = {
#         "input_ids": inputs['input_ids'],
#         "attention_mask": inputs['attention_mask'],
#         "do_sample": True,
#         "top_k": 40,
#         "top_p": 0.92,
#         "temperature": 0.7,
#         "max_new_tokens": 200,
#         'pad_token_id': tokenizer.eos_token_id
#     }
#     with torch.no_grad():
#         outputs = model.generate(**generate_args)

#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     print("Generated Confidence Text:", generated_text) 
#     # sorted_confidence_options = sorted(CONFIDENCE_LEVELS[confidence_type], key=len, reverse=True) 
#     # print ('sorted_confidence_options',sorted_confidence_options)
#     confidence_options = '|'.join(CONFIDENCE_LEVELS[confidence_type])
#     # confidence_options = '|'.join(sorted_confidence_options) 
#     match_verbal_confidence = re.search(fr'\b({confidence_options})\b', generated_text, flags=re.IGNORECASE)

#     # match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
#     print('match_guess', match_guess, 'match_confidence', match_verbal_confidence)

#     # confidence_map = {
#     #     'lowest': 0,
#     #     'low': 25,
#     #     'medium': 50,
#     #     'high': 75,
#     #     'highest': 100
#     # }
#     confidence_map = CONFIDENCE_MAP[confidence_type]

#     if not match_verbal_confidence:
#         guess_result = match_guess.group(0).lower()
#         # match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
#         # confidence_result = float(match_confidence) / 100  # Normalize confidence
#         confidence_result = 1.0


#         if guess_result not in label_list:
#             guess_result = 'null'

#         # Handle probabilities for different categories
#         probs = np.zeros(class_number)
#         if guess_result != 'null':
#             if task == 'sst2':
#                 if guess_result == 'positive':

#                     probs = np.array([0.0,1.0,0.0]) 
#                 else:
#                     probs = np.array([1.0,0.0,0.0])
#                 return guess_result, probs ,confidence_result
#             elif task == 'ag_news':
#                 granularity = 4 # how many confidence labels do we have in confidence elicitation
#                 ignore_classes  = 1 # ignore null and the current class
#                 other_level_candidates = class_number
                
#                 confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
#                 confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
#                 if guess_result == 'world':
#                     probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
#                 elif guess_result == 'sports' :
#                     probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
#                 elif guess_result == 'business':
#                     probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
#                 elif guess_result == 'tech' or 'science' or 'tech/science':
#                     guess_result = 'tech/sci'
#                     probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
#                 return guess_result, probs ,confidence_result

#     guess_result = match_guess.group(0).lower()
#     match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
#     confidence_result = float(match_confidence) / 100  # Normalize confidence


#     if guess_result not in label_list:
#         guess_result = 'null'

#     # Handle probabilities for different categories
#     probs = np.zeros(class_number)
#     if guess_result != 'null':
#         if task == 'sst2':
#             if guess_result == 'positive':
#                 probs = np.array(scale_vector_by_confidence([0,1],confidence_result) + [0.0])
#                 # probs = np.array([1-confidence_result,confidence_result,0.0])
#             else:
#                 probs = np.array(scale_vector_by_confidence([1,0],confidence_result) + [0.0])
#                 # probs = np.array([confidence_result,1-confidence_result,0.0])
#             print ('probs 2step',probs)
#             return guess_result, probs ,confidence_result
#         elif task == 'ag_news':
#             granularity = 4 # how many confidence labels do we have in confidence elicitation
#             ignore_classes  = 1 # ignore null and the current class
#             other_level_candidates = class_number
            
#             confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
#             confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
#             if guess_result == 'world':
#                 conf_vec = scale_vector_by_confidence([1.0,0.0,0.0,0.0],confidence_result)
#                 probs = np.array(conf_vec+[0.0])
#                 # probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
#             elif guess_result == 'sports' :
#                 conf_vec = scale_vector_by_confidence([0.0,1.0,0.0,0.0],confidence_result)
#                 probs = np.array(conf_vec+[0.0])
#                 # probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
#             elif guess_result == 'business':
#                 conf_vec = scale_vector_by_confidence([0.0,0.0,1.0,0.0],confidence_result)
#                 probs = np.array(conf_vec+[0.0])
#                 # probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
#             elif guess_result == 'tech' or 'science' or 'tech/science':
#                 guess_result = 'tech/sci'
#                 conf_vec = scale_vector_by_confidence([0.0,0.0,0.0,1.0],confidence_result)
#                 probs = np.array(conf_vec+[0.0])
#                 # probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
#             return guess_result, probs ,confidence_result
        

#     return guess_result, probs, confidence_result



# def predict_sentiment_and_verbal_confidence(text,task):
#     if task not in ['sst2', 'ag_news']:
#         raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

#     if task == 'sst2':
#         guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
#         class_number  = 2
#         label_list = ['positive', 'negative']
#     elif task == 'ag_news':
#         guess_pattern = r'(world|business|tech|science|sports)'
#         class_number  = 4
#         label_list = ['world','business', 'tech','science','tech/sci', 'sports']

  
  

#     if task == 'sst2':
#         prompt = f"""[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') for the following sentiment. Give ONLY the guess and verbal confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!>\n\nThe text is: ${text} Sentiment & confidence:[/INST]"""
#     elif task == 'ag_news':
#         prompt = f"""[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium','high', 'highest') for the following news article. Give ONLY the guess and verbal confidence, no other words or explanation. For example:\n\nGuess: <most likely guess, either world, sports, business or tech/science; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', 'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the confidence!>\n\nThe text is: ${text} News type & confidence:[/INST]"""

    


#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(device)

#     # Get model predictions as text
#     generate_args = {
#         "input_ids": inputs['input_ids'],
#         "attention_mask": inputs['attention_mask'],
#         "do_sample": True,  # enable sampling
#         "top_k": 40,  # top-k sampling
#         "top_p": 0.92,  # nucleus sampling probability
#         "temperature": 0.7,  # sampling temperature
#         "max_new_tokens": 200,
#         'pad_token_id': tokenizer.eos_token_id
#     }

#     with torch.no_grad():
#         outputs = model.generate(**generate_args)

#     prompt_length = len(inputs['input_ids'][0])
#     generated_tokens = outputs[0][prompt_length:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     print("Generated Text:", generated_text)

#     # Use regex to extract the guess (sentiment or category) and verbal confidence
#     match_guess = re.search(guess_pattern, generated_text, flags=re.IGNORECASE)

#     confidence_options = '|'.join(CONFIDENCE_LEVELS[confidence_type])
#     match_verbal_confidence = re.search(fr'\b({confidence_options})\b', generated_text, flags=re.IGNORECASE)
#     # match_verbal_confidence = re.search(r'\b(lowest|low|medium|high|highest)\b', generated_text, flags=re.IGNORECASE)
 

#     print('match_guess', match_guess, 'match_confidence', match_verbal_confidence)

#     confidence_map = {
#         'lowest': 0,
#         'low': 25,
#         'medium': 50,
#         'high': 75,
#         'highest': 100
#     }

#     if match_guess and match_verbal_confidence:
#         guess_result = match_guess.group(0).lower()
#         match_confidence = confidence_map[match_verbal_confidence.group(0).lower()]
#         confidence_result = float(match_confidence) / 100  # Normalize confidence


#         if guess_result not in label_list:
#             guess_result = 'null'

#         # Handle probabilities for different categories
#         probs = np.zeros(class_number)
#         if guess_result != 'null':
#             if task == 'sst2':
#                 if guess_result == 'positive':

#                     probs = np.array([1-confidence_result,confidence_result,0.0])
#                 else:
#                     probs = np.array([confidence_result,1-confidence_result,0.0])
#                 return guess_result, probs ,confidence_result
#             elif task == 'ag_news':
#                 granularity = 4 # how many confidence labels do we have in confidence elicitation
#                 ignore_classes  = 1 # ignore null and the current class
#                 other_level_candidates = class_number
                
#                 confidence_split = ((100-((100/granularity)*ignore_classes))/other_level_candidates)/100
#                 confidence_split = (1 - confidence_result)/ 3 # 1-conf to see how much confidene we have left
#                 if guess_result == 'world':
#                     probs = np.array([confidence_result,confidence_split,confidence_split,confidence_split,0.0])
#                 elif guess_result == 'sports' :
#                     probs = np.array([confidence_split,confidence_result,confidence_split,confidence_split,0.0])
#                 elif guess_result == 'business':
#                     probs = np.array([confidence_split,confidence_split,confidence_result,confidence_split,0.0])
#                 elif guess_result == 'tech' or 'science' or 'tech/science':
#                     probs = np.array([confidence_split,confidence_split,confidence_split,confidence_result,0.0])
#                 return guess_result, probs ,confidence_result
            

#         return guess_result, probs, confidence_result
#     else:
#         probs = np.zeros(class_number+1)
#         probs[-1] = 1.0  # Only null has confidence 1
#         return 'null', probs, 1.0


 
# def load_data(dataset_name): 
#     task = dataset_name
#     if task == 'sst2':
#         dataset = HuggingFaceDataset("glue", "sst2", split="validation", shuffle=True) 
#         label_names = dataset.label_names

#         # For dataset_class_1, only include sentences with 3 or more characters and label == 1
#         dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1 and len(text['sentence']) >= 0]
#         # dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1]
#         # used to take 3000:3250
#         dataset_class_1_t = dataset_class_1[:50]#[3000:3250]
#         incontext_dataset_class_1 = dataset_class_1[-5:]
#         # For dataset_class_0, only include sentences with 3 or more characters and label == 0
#         dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0 and len(text['sentence']) >= 0]
#         # dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0]
#         dataset_class_0_t = dataset_class_0[:50]#[3000:3250]
#         incontext_dataset_class_0 = dataset_class_0[-5:]
#         # label_names = ['negative','positive']
#         dataset_class =   dataset_class_1_t + dataset_class_0_t
#         return dataset_class,label_names
#     elif task == 'ag_news':
#         # dataset = load_dataset('ag_news', split='train').shuffle(seed=42)
#         dataset = HuggingFaceDataset('ag_news', split="test", shuffle=True) 
#         label_names = dataset.label_names
#         # For dataset_class_1, only include documents with 3 or more characters and label == 1
#         dataset_class_1 = [(text['text'], label) for (text, label) in dataset if label == 1 and len(text['text']) >= 0]
#         dataset_class_1_t = dataset_class_1[:250]
#         incontext_dataset_class_1 = dataset_class_1[-5:]

#         # For dataset_class_0, only include documents with 3 or more characters and label == 0
#         dataset_class_0 = [(text['text'], label) for (text, label) in dataset if label == 0 and len(text['text']) >= 0]
#         dataset_class_0_t = dataset_class_0[:250]
#         incontext_dataset_class_0 = dataset_class_0[-5:]

#         # You can extend for classes 2 and 3 similarly if needed
#         dataset_class_2 = [(text['text'], label) for (text, label) in dataset if label == 2 and len(text['text']) >= 0]
#         dataset_class_2_t = dataset_class_2[:250]
#         incontext_dataset_class_2 = dataset_class_2[-5:]

#         dataset_class_3 = [(text['text'], label) for (text, label) in dataset if label == 3 and len(text['text']) >= 0]
#         dataset_class_3_t = dataset_class_3[:250]
#         incontext_dataset_class_3 = dataset_class_3[-5:]

#         # label_names = ['world','sport','business','tech/sci']

#         # Combine datasets from different classes
#         dataset_class =   dataset_class_0_t + dataset_class_1_t + dataset_class_2_t + dataset_class_3_t

#         print(f'Total filtered dataset size: {len(dataset_class)}')
#         print(f'In-context samples for class 1: {incontext_dataset_class_1}')
#         print(f'In-context samples for class 0: {incontext_dataset_class_0}')
#         print(f'In-context samples for class 2: {incontext_dataset_class_2}')
#         print(f'In-context samples for class 3: {incontext_dataset_class_3}')
#         print (dataset_class) 
#         return dataset_class,label_names
#     else:
#         print("Task not supported.")



# def load_data(dataset_name):
#     task = dataset_name
#     if task == 'sst2':
#         dataset = HuggingFaceDataset("glue", "sst2", split="validation", shuffle=True)
#         label_names = dataset.label_names

#         # For dataset_class_1, only include sentences with 3 or more characters and label == 1
#         dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1 and len(text['sentence']) >= 3]
#         dataset_class_1_t = dataset_class_1[:250]
#         incontext_dataset_class_1 = dataset_class_1[-5:]

#         # For dataset_class_0, only include sentences with 3 or more characters and label == 0
#         dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0 and len(text['sentence']) >= 3]
#         dataset_class_0_t = dataset_class_0[:250]
#         incontext_dataset_class_0 = dataset_class_0[-5:]

#         dataset_class = dataset_class_1_t + dataset_class_0_t
#         return dataset_class, label_names

#     elif task == 'ag_news':
#         dataset = HuggingFaceDataset('ag_news', split="test", shuffle=True)
#         label_names = dataset.label_names

#         # For dataset_class_1, only include documents with 3 or more characters and label == 1
#         dataset_class_1 = [(text['text'], label) for (text, label) in dataset if label == 1 and len(text['text']) >= 3]
#         dataset_class_1_t = dataset_class_1[:250]
#         incontext_dataset_class_1 = dataset_class_1[-5:]

#         # For dataset_class_0, only include documents with 3 or more characters and label == 0
#         dataset_class_0 = [(text['text'], label) for (text, label) in dataset if label == 0 and len(text['text']) >= 3]
#         dataset_class_0_t = dataset_class_0[:250]
#         incontext_dataset_class_0 = dataset_class_0[-5:]

#         # For dataset_class_2
#         dataset_class_2 = [(text['text'], label) for (text, label) in dataset if label == 2 and len(text['text']) >= 3]
#         dataset_class_2_t = dataset_class_2[:250]
#         incontext_dataset_class_2 = dataset_class_2[-5:]

#         # For dataset_class_3
#         dataset_class_3 = [(text['text'], label) for (text, label) in dataset if label == 3 and len(text['text']) >= 3]
#         dataset_class_3_t = dataset_class_3[:250]
#         incontext_dataset_class_3 = dataset_class_3[-5:]

#         # Combine datasets from different classes
#         dataset_class = dataset_class_0_t + dataset_class_1_t + dataset_class_2_t + dataset_class_3_t

#         print(f'Total filtered dataset size: {len(dataset_class)}')
#         print(f'In-context samples for class 1: {incontext_dataset_class_1}')
#         print(f'In-context samples for class 0: {incontext_dataset_class_0}')
#         print(f'In-context samples for class 2: {incontext_dataset_class_2}')
#         print(f'In-context samples for class 3: {incontext_dataset_class_3}')
#         print (dataset_class)
#         return dataset_class, label_names 

#     elif task == 'popQA':
#         from datasets import load_dataset
#         import json
#         from collections import defaultdict

#         split = "test"
#         print(f"Loading PopQA {split} dataset from 'akariasai/PopQA'...")
#         raw_dataset = load_dataset("akariasai/PopQA", split=split).select(range(5))
#         print(f"PopQA {split} dataset loaded successfully", type(raw_dataset),'column names',raw_dataset.column_names)

#         # add a new column, and for each datasample given raw_dataset['prop'] (e.g capital, business etc) extract all other rows that have the same raw_dataset['prop'] and add their answers which are accessed raw_dataset['possible_answers'] and add them to a new column raw_dataset['wrong_possible_answers'] 
#         # Create a dictionary to hold all possible answers for each 'prop'
#         # prop_to_answers = defaultdict(list)
#         # # Populate the dictionary
#         # for sample in raw_dataset:
#         #     prop_to_answers[sample['prop']].append(sample['possible_answers'])


#         prop_to_answers = {}
#         # Populate the dictionary
#         for sample in raw_dataset:
#             # print ('sample',sample)
#             if sample['prop'] in prop_to_answers:
#                 prop_to_answers[sample['prop']] |= set(json.loads(sample['possible_answers']))
#             else:
#                 prop_to_answers[sample['prop']] = set(json.loads(sample['possible_answers']))
                
 

#         # Create the new column 'wrong_possible_answers'
#         wrong_possible_answers_col = []
#         possible_correct_incorrect_answers_col = []
#         for sample in raw_dataset:
#             current_prop = sample['prop']
#             # current_answers = sample['possible_answers']
#             current_answers = set(json.loads(sample['possible_answers']))
#             print ('current_answers',current_answers)

#             # Get all possible answers for the current 'prop'
#             # all_answers_for_prop = prop_to_answers[current_prop]
#             all_answers_for_prop =  prop_to_answers[current_prop]
#             print ('all_answers_for_prop',all_answers_for_prop)

#             # Perform set difference to find wrong possible answers
#             wrong_answers = all_answers_for_prop - current_answers
#             print ('wrong_answers',wrong_answers)
#             # Convert back to list
#             wrong_possible_answers_col.append(json.dumps(list(wrong_answers))) 

#             possible_correct_incorrect_answers_col.append(json.dumps({
#                 'correct': list(current_answers),
#                 'incorrect': list(wrong_answers)
#             }))
#             # # Create the list of wrong possible answers by excluding the current sample's answers
#             # wrong_answers = [ans for ans in all_answers_for_prop if ans != current_answers]

#             # # Flatten the list of lists
#             # wrong_answers = [item for sublist in wrong_answers for item in sublist]

#             # wrong_possible_answers_col.append(wrong_answers) 
#         # Add the new column to the dataset
#         raw_dataset = raw_dataset.add_column("wrong_possible_answers", wrong_possible_answers_col)
#         raw_dataset = raw_dataset.add_column("possible_correct_incorrect_answers", possible_correct_incorrect_answers_col)

#         print ('raw_dataset',raw_dataset)
#         print (raw_dataset['wrong_possible_answers']) 
#         print (raw_dataset['possible_correct_incorrect_answers']) 
#         # with   create all possible answers, in possible_answers, we insert a dictionary = {'correct':{},'incorrect':{}} 
#         raw_dataset = raw_dataset.rename_column("possible_correct_incorrect_answers", "label")
#         raw_dataset = raw_dataset.rename_column("question", "text") 
#         # Create the PyTorch dataset
#         dataset = HuggingFaceDataset(raw_dataset,split="test", shuffle=True)
#         print(f"Converted to PyTorch Dataset", type(dataset))
#         label_names = None
 
#         # Collect data for dataset_class_t and incontext_dataset_class
#         dataset_class = [] 
#         dataset_class = [(text['text'], label) for (text, label) in dataset]
#         dataset_class_t = dataset_class[:50]
#         incontext_dataset_class = dataset_class[-5:]

#         print(f'Total filtered dataset size for PopQA: {len(dataset_class)}')
#         print(f'In-context samples for PopQA: {incontext_dataset_class}')
        
        
#         return dataset_class_t, label_names
#     elif task == 'strategyQA':
#         from datasets import load_dataset 
#         print(f"Loading StrategyQA dataset from 'ChilleD/StrategyQA'...")
#         strategy_dataset = load_dataset('ChilleD/StrategyQA', split='test').select(range(500)) # example: taking 50 samples
#         print(f"StrategyQA dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)

#         strategy_dataset = strategy_dataset.rename_column("question", "text")
#         strategy_dataset = strategy_dataset.rename_column("answer", "label")

#         strategy_dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True)
#         print(f"Converted StrategyQA to PyTorch Dataset", type(strategy_dataset))

#         strategy_dataset_class = []
#         strategy_dataset_class = [(text['text'], label) for (text, label) in strategy_dataset]
#         strategy_dataset_class_t = strategy_dataset_class[:500]
#         strategy_incontext_dataset_class = strategy_dataset_class[-5:]
#         label_names = ['false','true']
#         print(f'Total filtered dataset size for StrategyQA: {len(strategy_dataset)}')
#         print(f'In-context samples for StrategyQA: {strategy_incontext_dataset_class}')
#         return strategy_dataset_class_t, label_names
     
#     else:
#         print("Task not supported.")


  
import argparse
from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP,TASK_N_CLASSES,MODEL_INFO # ,TASK_LABEL_TO_NAME, TASK_NAME_TO_LABEL



# # Define the argument parser
# parser = argparse.ArgumentParser(description='Argument parser for model configuration')

# # Define the arguments
# parser.add_argument('--model_type', type=str, default='llama3', choices=['llama2','mistral','llama3','llama2_13b', 'other_model_types_here'],
#                     help='Type of the model to use')
# parser.add_argument('--task', type=str, default='strategyQA', choices=['sst2','ag_news','popQA','strategyQA', 'other_tasks_here'],
#                     help='Task to perform')
# parser.add_argument('--confidence_type', type=str, default='weighted_confidence', choices=['verbal_confidence','weighted_confidence','sherman_kent', 'yougov','single_token_mix'],
#                     help='type of confidence levels')
# parser.add_argument('--prompting_type', type=str, default='step2_k_pred_avg', choices=['s1','w1','s1_black_box', '2step', 'empirical','k_pred_avg','step2_k_pred_avg' ,'other_prompting_types_here','sspattack','e_guided_paraphrasing'],
#                     help='Type of prompting to use')
# parser.add_argument('--prompt_shot_type', type=str, default='fs', choices=['fs','zs', 'other_shot_types_here'],
#                     help='Type of prompt shot to use')
# parser.add_argument('--k_pred', type=int, default=20,
#                     help='Number of predictions to perform')
# parser.add_argument('--similarity_techneque', type=str, default='USE',
#                     help='similarity technique (USE or BERTScore)')
# parser.add_argument('--num_transformations', type=int, default=20,
#                     help='Number of transformations to perform')
# parser.add_argument('--index_order_technique', type=str, default='prompt_top_k', choices=['prompt_top_k','random' ,'other_techniques_here'],
#                     help='Index order technique to use')
# parser.add_argument('--cache_transformers', type=str, default='/mnt/hdd/brian/hub',
#                     help='Directory for transformers cache')
# parser.add_argument('--experiment_name_folder', type=str, default='attack_calibrated_model',
#                     help='Folder name for the experiment')
# # Parse the arguments
# args = parser.parse_args()

# from src.utils.shared.arg_config import get_args

# args = get_args()

# # Function to print Namespace in a human-readable format
# def print_args_human_readable(namespace):
#     print("Arguments Namespace:")
#     for key, value in vars(namespace).items():
#         print(f"  - {key.replace('_', ' ').capitalize()}: {value}")

# # Print the Namespace
# print_args_human_readable(args)
  
# # dataset = load_dataset("glue", "sst2") 
# # task = args.task 
# # task = 'ag_news'
# # prompting_type = args.prompting_type #'step2_k_pred_avg' #'2step'
# # model_type = args.model_type #'llama3'

# high_level_folder = args.experiment_name_folder
# test_folder = os.path.join(high_level_folder, f'{args.model_type}_{args.task}_log_folder') 
# args.test_folder = test_folder 
# os.makedirs(test_folder, exist_ok=True)

# cache_dir = args.cache_transformers#  "/mnt/hdd/brian/hub" # args.cache_transformers
# os.environ['TFHUB_CACHE_DIR'] = cache_dir
# # texts, true_labels = load_data(task)

# from src.utils.shared.misc import environment_setup
# args = environment_setup()


args.n_classes =  TASK_N_CLASSES[args.task] 
args.confidence_type_dict = CONFIDENCE_LEVELS[args.confidence_type] 
args.confidence_map_dict = CONFIDENCE_MAP[args.confidence_type] 
# args.task_label_to_name_dict = TASK_LABEL_TO_NAME[args.task]
# args.task_name_to_lebel_dict = TASK_NAME_TO_LABEL[args.task]
model_info = MODEL_INFO[args.model_type]



from src.utils.shared import load_data
data_to_evaluate, label_names = load_data(args)


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

from src.utils.shared import SimpleDataset
# Convert filtered datasets into TextAttack Dataset format 
args.dataset =  SimpleDataset(data_to_evaluate,label_names = label_names ) 



print("Dataset loaded successfully.",args.dataset)
# args.confidence_type = 'weighted_confidence'






args.model_name =  model_info['model_name']
args.start_prompt_header = model_info['start_prompt_header']
args.end_prompt_footer = model_info['end_prompt_footer']

# print ('model_info',model_info) 
# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=cache_dir,trust_remote_code=True)
 
 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

 
# args.tokenizer = tokenizer
# args.device = device
# args.model = model
from src.utils.shared.misc import initialize_model_and_tokenizer
args = initialize_model_and_tokenizer(args) 
from src.containers import AbstractPredictor, BasePredictorResults, ClassifierPredictorResults

class PredictionContainer(AbstractPredictor):
    def __init__(self):
        self.base_results = BasePredictorResults()
        self.classifier_results = ClassifierPredictorResults()

from src.inference import Step2KPredAvg

# if args.prompting_type == 'step2_k_pred_avg':
from src.inference.inference_config import DYNAMIC_INFERENCE
args.predictor = DYNAMIC_INFERENCE[args.prompting_type](**vars(args))
args.predictor.predictor_container = PredictionContainer()

if 'gpt-4o' in args.model_type: 
    # we load a llama3 model so that our code is compatible with huggingface, but every call is made directly to the api
    import requests
    import json
    import time
    def _call_model_API(self,generate_args,extra_args):
        api_key = ''
        print ('generate_args',generate_args)
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        payload = {
            'model': 'gpt-4o',
            'messages': [
                {'role': 'system', 'content': "You are an expert assistant"},
                {'role': 'user', 'content': extra_args['prompt']},
            ],
            'max_tokens': generate_args['max_new_tokens'],
            'n': 1,
            'stop': None,
            'temperature': generate_args['temperature'],
            # 'logprobs': True,
            # 'top_logprobs': 5,
            'logprobs': False, 
        }

        # response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print ('response',response)
        # if response.status_code == 200:
        #     response_data = response.json()
        #     print ('response_data',response_data)
        #     choice = response_data['choices'][0]
        #     message_content = choice['message']['content']
        #     print ('message_content',message_content)
        #     return message_content
        # else:
        #     print(f"Error: {response.status_code}")
        #     print(response.text)
        #     return None
        max_retries = 10
        wait_time = 60  # seconds

        for attempt in range(max_retries):
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            print('response', response)

            if response.status_code == 200:
                response_data = response.json()
                print('response_data', response_data)
                choice = response_data['choices'][0]
                message_content = choice['message']['content']
                print('message_content', message_content)
                return message_content
            else:
                print(f"Error: {response.status_code}")
                print(response.text)

                if attempt < max_retries - 1:  # Don't wait after the last attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return None


    from src.prompting.classification import BaseClassificationPrompt
    args.predictor.prompt_class._call_model = _call_model_API.__get__(args.predictor.prompt_class,BaseClassificationPrompt)



# if args.prompting_type == 'step2_k_pred_avg':
#     predictor = Step2KPredAvg(**vars(args))
#     predictor.predictor_container = PredictionContainer()
# else:
#     print ('invalid')
#     sys.exit()


# print ('texts',texts,'len texts', len(texts), 'true_labels',sum([i for i in true_labels if i == 1]),len(true_labels))
# print (true_labels)
# Predict and construct confusion matrix
predictions = []
probabilities = []
correctness=[]
confidences=[]
smaller_true_labels = []
counter_null = 0
# for  text, true_label in zip(texts, true_labels):
for datapoint in args.dataset:
    # print ('datapoint',datapoint) 
    text, true_label = datapoint

 
    print ('text true label',text,true_label)
     
    # expected_prediction = label_names[true_label]
    # if prompting_type == '2step':
    #     guess, probs, confidence  = predict_sentiment_and_verbal_confidence_2steps(text,task) 
    # elif prompting_type == 's2':
    #     guess, probs, confidence  = predict_sentiment_and_verbal_confidence(text,task) 
    # elif prompting_type == 'k_pred_avg':
    #     expected_prediction = label_names[true_label]
    #     guess, probs, confidence = predict_sentiment_and_verbal_confidence_k_pred_avg(text,task,expected_prediction=expected_prediction)
    # elif prompting_type == '2step_k_pred_avg':

    #     guess, probs, confidence = predict_sentiment_and_verbal_confidence_2step_k_pred_avg(datapoint=datapoint,label_names=label_names)
    # print ('guess, probs, confidence',guess, probs, confidence)

    guess, probs, confidence = args.predictor.predict_and_confidence(datapoint)
    if guess == 'null':
        counter_null+=1
    #     continue
    # prediction_label = 0 if guess == 'negative' elif guess = 'positive' 1 else 2
    
    prediction_label = args.predictor.prompt_class.task_name_to_label[guess] #TASK_NAME_TO_LABEL




    if args.task == 'popQA':
        true_label=1
    elif args.task =='strategyQA': 
        true_label = int(true_label) 
    elif args.task == 'triviaQA':
        true_label = 1 if guess == 'true' else 0
        # true_label = 1
    
    args.predictor.predictor_container.add_true_label(true_label)
    args.predictor.predictor_container.add_probability(probs)
    args.predictor.predictor_container.add_confidence(confidence)
    
    
    predictions.append(prediction_label)
    # Take the probability of the positive label for calibration curve
    # probabilities.append(probs[1])
    probabilities.append(probs)

    correctness.append(prediction_label)
    confidences.append(confidence)
    smaller_true_labels.append(true_label)



from src.utils.shared.evaluation.predictor_evaluation import predictor_evaluation

predictor_evaluation(args,args.predictor)




# true_labels = np.array(smaller_true_labels)




# conf_matrix = confusion_matrix(true_labels, predictions, labels=list(range(args.n_classes+1)))


# # Output the confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)
# acc = accuracy_score(true_labels, predictions)
# print("Accuracy:", acc, 'counter_null:',counter_null)

# null_class = args.n_classes
# # Filter out null class instances
# # filtered_true_labels = [label for label in true_labels if label != null_class]
# # filtered_predictions = [pred for true_label, pred in zip(true_labels, predictions) if true_label != null_class]

# filtered_predictions = [pred for pred in predictions if pred != null_class]
# filtered_true_labels = [true_label for true_label, pred in zip(true_labels, predictions) if pred != null_class]
# filtered_probabilities = [probability for probability, pred in zip(probabilities, predictions) if pred != null_class]
# filtered_probabilities = np.array(filtered_probabilities)
# filtered_true_labels =  np.array(filtered_true_labels)
# filtered_probabilities = np.array(filtered_probabilities)
# # Calculate the accuracy ignoring the null class
# filtered_accuracy = accuracy_score(filtered_true_labels, filtered_predictions)

# print(f'Filtered Accuracy: {filtered_accuracy:.4f}')

# probabilities = np.array(probabilities)
# # print ('probabilities',probabilities) 
# correctness = np.array([1 if true_labels[i] == np.argmax(probabilities[i]) else 0 for i in range(len(true_labels))])
# confidences = np.max(probabilities, axis=1)



# name_plot = 'calibration_baseline_probs'
# print ('Base Calibration Metrics')
# from src.utils.shared import calculate_roc_metrics, plot_roc_curve,  plot_calibration_curve, calculate_ece_for_all_classes
# calculate_roc_metrics(args, true_labels, probabilities)
# plot_roc_curve(args, true_labels, probabilities, name_plot = 'calibration_baseline_probs')
# calculate_ece_for_all_classes(args, true_labels, probabilities) 
# plot_calibration_curve(args, true_labels, probabilities, name_plot = 'calibration_baseline_probs') 

 

# name_plot = 'calibration_filtered_probs'
# print ('Filtered Calibration Metrics')
# calculate_roc_metrics(args, filtered_true_labels, filtered_probabilities)
# plot_roc_curve(args, filtered_true_labels, filtered_probabilities, name_plot = 'calibration_filtered_probs')
# calculate_ece_for_all_classes(args, filtered_true_labels, filtered_probabilities) 
# plot_calibration_curve(args, filtered_true_labels, filtered_probabilities, name_plot = 'calibration_filtered_probs') 
 
