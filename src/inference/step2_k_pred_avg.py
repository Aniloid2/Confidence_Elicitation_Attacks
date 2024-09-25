import json
import re
import numpy as np
import torch
from .base_predictor  import BasePredictor
from numpy.random import dirichlet
# from src.utils.shared.misc import identify_correct_incorrect_labels
from src.prompting.prompt_config import DYNAMIC_PROMPT
# DYNAMIC_PROMPT = {
    
#     'ag_news':AgNewsPrompts() [_predict_prompt(), _confidence_prompt(), _predict_and_confidence_prompt(), cot_promp()], AgNewsPrompts __init__ will setup vars like self.task_dictionary_counts_correct
# }
import os




class Step2KPredAvg(BasePredictor):
    def __init__(self, **kwargs):# model, tokenizer): 
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.technique_name = self.__class__.__name__
    
        if self.task not in DYNAMIC_PROMPT[self.task_structure]:
            print ('couldent find any prompt template, this can be implemented to a generalized class to handle generalized prompt structures')
            pass # prompt_class = BasePrompts() [initializes a classification task]
        else:
            self.prompt_class = DYNAMIC_PROMPT[self.task_structure][self.task](**kwargs)
        
        # classifiers
        self.predictor_container = None
        self.inference_step = None
        self.current_sample = None
        # self.top_k_max_prediction = None
        # self.top_k_max_prediction_and_confidence = None
        # self.top_k_dirichlet_mean = None
        # self.vanilla_prediction = None
        # self.vanilla_prediction_and_confidence = None
        # self.cot_prediction = None
        # self.cot_prediction_and_confidence = None
    # prompt_class._predict()
    # prompt_class._confidence()

    # this class should rally return a custom object in predict_and_confidence where we have an object called target which holds all information associated with prompt inference result

    # def __init__(self, tokenizer, model, device, task, label_names):
    #     self.tokenizer = tokenizer
    #     self.model = model
    #     self.device = device
    #     self.task = task
    #     self.label_names = label_names

    # def predict_sentiment_and_verbal_confidence_2step_k_pred_avg(self,datapoint):
    def predict_and_confidence(self, datapoint):
         
        # if task not in ['sst2', 'ag_news', 'popQA']:
        #     raise ValueError("Unsupported task. Please choose 'sst2', 'ag_news', or 'popQA'.")

        # Custom processing for PopQA

        # if self.task == 'popQA': 
        #     data_point = {'question':datapoint[0], 'possible_answers':datapoint[1]}
        #     text = data_point['question']

        #     possible_answers = json.loads(data_point['possible_answers'])
        #     label_names = possible_answers['correct'] + possible_answers['incorrect']
        #     print ('label_names', label_names)

        #     expected_prediction = possible_answers['correct']
        #     incorrect_answers = possible_answers['incorrect']
        #     guess_pattern = '|'.join(label_names)
        #     class_number = len(label_names)

        #     # Create task-specific dictionaries for PopQA
        #     task_dictionary_counts_correct = {self.task:{label: 0 for label in expected_prediction}}
        #     task_dictionary_counts_incorrect = {self.task:{label: 0 for label in incorrect_answers}}

        #     task_dictionary_confidences = {self.task:{label: 0 for label in label_names}}
        #     task_dictionary_confidences['null'] = 0

        #     task_dictionary_counts = {self.task:{label: 0 for label in label_names}}
        #     task_dictionary_counts['null'] = 0

        #     # Create a sample list for examples in the prompt
        #     sampled_answers = random.sample(label_names, min(5, len(label_names)))
        #     sampled_answers_str = ', '.join(label_names)
        #     sampled_confidences_str =', '.join(self.confidence_type_dict)
        #     print ('sampled_confidences_str',sampled_confidences_str)
        #     prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following question. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses; not a complete sentence, just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: ${data_point['question']} Guesses:{self.end_prompt_footer}"""
        # else:
        #     print ('datapoint',datapoint)
        #     if self.task == 'sst2':
        #         # guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
        #         class_number = 2
        #         label_list = self.dataset.label_names  # ['positive', 'negative']
        #         label_index = datapoint[1]
        #         guess_pattern = '|'.join([f'\\b{label}\\b' for label in label_list])
        #     elif self.task == 'ag_news':
        #         # guess_pattern = r'(world|business|tech|science|sports)'
        #         guess_pattern = r'(world|business|sci/tech|sports)'
        #         class_number = 4
        #         label_list = [lab.lower() for lab in self.dataset.label_names] 
        #         label_index = datapoint[1]
        #         guess_pattern = '|'.join([f'\\b{label}\\b' for label in label_list])
        #     elif self.task == 'strategyQA':
        #         guess_pattern = r'(true|TRUE|True|false|FALSE|False)'
        #         class_number = 2
        #         label_list = self.dataset.label_names # ['false', 'true']
        #         label_index = 1 if datapoint[1] else 0
        #         guess_pattern = '|'.join([f'\\b{label}\\b' for label in label_list])
            
        #     print ('label_index',label_index,label_list) 
        #     text = datapoint[0]
        #     # expected_prediction = [label_list[label_index]]
        #     # print ('sets',set(label_list) , set([label_list[label_index]])) 
        #     # incorrect_answers = list (set(label_list) -  set([label_list[label_index]])) 
        #     # print ('text', text)
        #     # print ('expec', expected_prediction) 
        #     # print ('expected_prediction',expected_prediction)
        #     # print ('incorrect_answers',incorrect_answers)

        #     expected_prediction, incorrect_answers  = identify_correct_incorrect_labels(label_list,label_index)

        #     task_dictionary_counts = {self.task: {label: 0 for label in label_list}}
        #     task_dictionary_counts[self.task]['null'] = 0

        #     task_dictionary_confidences = {self.task: {label: 0 for label in label_list}}
        #     task_dictionary_confidences[self.task]['null'] = 0
        #     # Create task-specific dictionaries for PopQA
        #     task_dictionary_counts_correct = {self.task :{label: 0 for label in expected_prediction}}
        #     task_dictionary_counts_incorrect = {self.task:{label: 0 for label in incorrect_answers}}

        #     sampled_confidences_str =', '.join(self.confidence_type_dict)
        #     print ('sampled_confidences_str',sampled_confidences_str)
            
        #     if self.task == 'sst2':
        #         prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a comma, for example [Negative, Positive, Positive, Negative ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""

        #     elif self.task == 'ag_news':
        #         # options = 'world, business, tech, science, sports'  # we separate tech/science into two different prediction categories, but treat them as one label
        #         options = 'world, business, sci/tech, sports'  # we separate tech/science into two different prediction categories, but treat them as one label
                 
        #         prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses! Separated by a comma, for example [Sports, Business, Sports, World, Sci/Tech, World ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
        #     elif self.task == 'strategyQA':
        #         # prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guess for the following text (false, true). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either true orfalse; not a complete sentence, just the guesses! Separated by a comma, for example [True, False, False, True ...]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
        #         prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses, for example [True, False, False, True ...]. Only output your answer nothing else!\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
        #         # prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. Only output your answer nothing else!\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
                
        #         # prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false?\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
        

        text, label_index = datapoint
        self.prompt_class._initialize_sample_counters() 
        print ('label_index_text',text,label_index)
        expected_prediction, incorrect_answers = self.prompt_class._identify_correct_incorrect_labels(label_index)
        self.prompt_class._initialize_correct_incorrect_predictions(label_index)
        prompt = self.prompt_class._predict_prompt(text )
 
        
        

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": self.temperature,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        # with torch.no_grad():
        #     outputs = self.model.generate(**generate_args)

        # prompt_length = len(inputs['input_ids'][0])
        # generated_tokens = outputs[0][prompt_length:]
        # generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print('prompt', prompt)
        # print("Generated Prediction Text:", generated_text) 

        generated_text = self.prompt_class._call_model(generate_args)

        # Regex to find 'true' or 'false', case-insensitive, ensuring full word match
        # pattern = re.compile(r'\btrue\b|\bfalse\b', re.IGNORECASE)
        pattern = re.compile(self.prompt_class.guess_pattern_prediction, re.IGNORECASE)
        # Find all matches in the text
        matches = pattern.findall(generated_text)
        # Convert all matches to lowercase (optional, for consistency)
        results = [match.lower() for match in matches]



        # # Extract guesses, assuming they're separated by commas and ignoring case
        # results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
        # results = [result for result in results if result in label_list]# else 'null' for result in results]
        # # If fewer results than k_pred, fill with 'null'

        self.prompt_class._extend_with_null(results)
        # results.extend(['null'] * (self.k_pred - len(results)))



        print('results', results, expected_prediction)
        correct_predictions = sum(1 for pred in results if pred in expected_prediction)
        confidence_empirical = (correct_predictions / len(results)) * 100
        print('correct_predictions',correct_predictions)

        self.prompt_class._calculate_result_counts(results)

        # for result in results:
        #     if result in self.prompt_class.task_dictionary_counts[self.task]:
        #         self.prompt_class.task_dictionary_counts[self.task][result] += 1
        #     else:
        #         self.prompt_class.task_dictionary_counts[self.task]['null'] +=1
        
        self.prompt_class._calculate_result_confidences(results)
        # for pred, number_of_results in self.prompt_class.task_dictionary_counts[self.task].items():
        #     self.prompt_class.task_dictionary_confidences[pred] = (number_of_results / len(results))

        
        
        self.prompt_class._calculate_result_count_correct(results, weight = 1) 
        # for result in results:
        #     if result in self.prompt_class.task_dictionary_counts_correct[self.task]:
        #         self.prompt_class.task_dictionary_counts_correct[self.task][result]+=1

        self.prompt_class._calculate_result_count_incorrect(results, weight = 1)
        # for result in results:
        #     if result in self.prompt_class.task_dictionary_counts_incorrect[self.task]:
        #         self.prompt_class.task_dictionary_counts_incorrect[self.task][result]+=1

        print(f"Results for '{text}':")
        print(f"Counter: {self.prompt_class.task_dictionary_counts}")
        print(f"Empirical confidence: {confidence_empirical}%")
        guess_result = max(self.prompt_class.task_dictionary_counts[self.task], key=self.prompt_class.task_dictionary_counts[self.task].get)
        print('max_class', guess_result, expected_prediction)
        print ('task_dictionary_counts_correct[task]',self.prompt_class.task_dictionary_counts_correct[self.task])
        print ('task_dictionary_counts_incorrect[task]', self.prompt_class.task_dictionary_counts_incorrect[self.task])


        guesses_output = results

        confidence_prompt = self.prompt_class._confidence_prompt(text, guesses_output)
        
        # if self.task == 'sst2':
        #     # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [{sampled_confidences_str} ...]> Confidences:{self.end_prompt_footer}"""
        #     confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        
        # elif self.task == 'ag_news':
        #     # confidence_prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guesses for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: \n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences:  Confidences:{self.end_prompt_footer}"""
        #     confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        
        # elif self.task == 'popQA':
        #     confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${data_point['question']}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [{sampled_confidences_str} ...]> Confidences:{self.end_prompt_footer}"""
        # elif self.task == 'strategyQA':
        #     # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (false, true). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either true or false; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        #     # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. Only output your answer nothing else!\n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        #     confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. \n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. \n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        #     # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guess.\n\nStatement: {text}$. The guess were: {guesses_output}, given the guess provide the verbal confidences that your guess us correct. \n\nFor example:\n\Confidence: <the confidence, from either {self.confidence_type_dict} that your guess is correct, for example [{sampled_confidences_str} ...]; just the confidence!> Confidence:{self.end_prompt_footer}"""
        
        
        
        inputs = self.tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.92,
            "temperature": self.temperature,
            "max_new_tokens": 300,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        generated_text = self.prompt_class._call_model(generate_args)
        # with torch.no_grad():
        #     outputs = self.model.generate(**generate_args)

        # prompt_length = len(inputs['input_ids'][0])
        # generated_tokens = outputs[0][prompt_length:]
        # generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print("Generated Confidence Text:", generated_text) 

        # confidence_options = '|'.join(self.confidence_list) 
        confidence_options = self.prompt_class.guess_pattern_confidence
        confidence_guesses = re.findall(confidence_options, generated_text, flags=re.IGNORECASE)
        confidence_guesses = self.prompt_class._lower_labels(confidence_guesses) # [match.lower() for match in confidence_guesses]
        print('confidence_guesses', confidence_guesses)
        confidence_list = self.prompt_class.confidence_list
        print('confidence_list', confidence_list)
        confidence_results = [result for result in confidence_guesses if result in confidence_list]
        confidence_results = self.prompt_class._extend_with_null(confidence_results)
        # confidence_results.extend(['null'] * (self.k_pred - len(confidence_results)))

        confidence_map = self.prompt_class.confidence_map

        confidence_numerical_results = [
            confidence_map[result] if result != 'null' else min(confidence_map.values()) 
            for result in confidence_results
        ] 
        print('confidence_numerical_results', confidence_numerical_results) 
        print('guesses_output', guesses_output, self.prompt_class.label_list)

        weighted_counts = {label: 0.0 for label in self.prompt_class.label_list}
        weighted_counts['null'] = 0.0


        for pred, confidence in zip(guesses_output, confidence_numerical_results):
            if confidence:
                
                weighted_counts[pred] += confidence
            else:
                weighted_counts[pred] += 1
        
        guess_result_with_confidence = max(weighted_counts, key=weighted_counts.get) 

        self.prompt_class._add_confidence_weight_correct(results, confidence_numerical_results)

        # for result, confidence in zip(guesses_output, confidence_numerical_results):
        #     if result in self.prompt_class.task_dictionary_counts_correct[self.task]:
        #         self.prompt_class.task_dictionary_counts_correct[self.task][result]+=confidence

        self.prompt_class._add_confidence_weight_incorrect(results, confidence_numerical_results)

        # for result, confidence in zip(guesses_output, confidence_numerical_results):
        #     if result in self.prompt_class.task_dictionary_counts_incorrect[self.task]:
        #         self.prompt_class.task_dictionary_counts_incorrect[self.task][result]+=confidence
        
        weighted_counts_binary = {'incorrect':sum(self.prompt_class.task_dictionary_counts_incorrect[self.task].values()),
                                'correct':sum(self.prompt_class.task_dictionary_counts_correct[self.task].values()),
                                'null':weighted_counts['null']}
        #for all datasets have a correct, incorrect and null bucket

        def compute_dirichlet_statistics(weighted_counts, label_list):
            print(' ', weighted_counts)

            # You mentioned 'weighted_counts_binary' in your request, but it seems missing.
            # Assuming it's another dictionary similar to weighted_counts. For now, we skip this.
            # print('weighted_counts_binary', weighted_counts_binary)  

            alpha_prior = 1.0
            alpha = {label: weighted_counts[label] + alpha_prior for label in label_list}
            alpha['null'] = weighted_counts['null'] + alpha_prior

            alpha_values = list(alpha.values())
            sample_size = 1000
            dirichlet_distribution = dirichlet(alpha_values, size=sample_size)
            # samples_ternary = [(p[0], p[1], p[2]) for p in dirichlet_distribution]  
            # samples_ternary = [tuple(p[i] for i in range(self.n_classes + 1)) for p in dirichlet_distribution]
            empirical_mean = np.mean(dirichlet_distribution, axis=0)
            # empirical_mean_ternary = (empirical_mean[0], empirical_mean[1], empirical_mean[2])
            print('empirical_mean', empirical_mean) 

            from src.utils.shared.plotting import ternary_plot, ternary_mean_plot
            if self.ternary_plot == True:
                if empirical_mean.shape[0] == 3:
                    if self.current_sample:
                        # self.inference_step +=1
                        samples_ternary = [tuple(p[i] for i in range(self.n_classes + 1)) for p in dirichlet_distribution]
                        empirical_means_ternary = (empirical_mean[0], empirical_mean[1], empirical_mean[2])
                        dirichlet_folder = os.path.join(self.test_folder, 'dirichlet')
                        
                        if not os.path.exists(dirichlet_folder):
                            os.makedirs(dirichlet_folder)
                        ternary_plot_file = os.path.join(dirichlet_folder, f'dirichlet_cs{self.current_sample}_is{self.inference_step}_a({alpha_values})_n{str(sample_size)}')
                        self.logging.info(f'Plotting ternary plot for sample {ternary_plot_file}')
                        ternary_mean_plot(samples_ternary,alpha_values,empirical_means_ternary,ternary_plot_file)
                else:
                    logging.warning('Not plotting ternary plot, as the number of classes is not 3')
            def dirichlet_variance(alpha):
                alpha_0 = sum(alpha)
                variances = [(alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1)) for alpha_i in alpha]
                return variances

            alpha_vector = list(alpha.values())
            second_order_uncertainty = dirichlet_variance(alpha_vector)
            probabilities = dirichlet_distribution[0] 
            
            print("Counts:", self.prompt_class.task_dictionary_counts)
            print("Numerical Confidences:", confidence_numerical_results)
            print("Weighted Counts:", weighted_counts)
            print("Alpha Vector:", alpha_vector)
            print("Probabilities:", probabilities)
            print("Second Order Uncertainty:", second_order_uncertainty) 
            return alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities 

        alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts,weighted_counts.keys())
        # alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts_binary,weighted_counts_binary.keys())
        self.second_order_uncertainty = second_order_uncertainty
        self.empirical_mean = empirical_mean  
        

        if self.predictor_container: 
            self.predictor_container.add_top_k_max_prediction(self.prompt_class.task_name_to_label[guess_result])

            self.predictor_container.add_top_k_max_prediction_and_confidence(self.prompt_class.task_name_to_label[guess_result_with_confidence])
            guess_result_empirical_mean = self.prompt_class.label_list_with_null[np.argmax(empirical_mean)]
            self.predictor_container.add_top_k_dirichlet_mean(self.prompt_class.task_name_to_label[guess_result_empirical_mean])
        
        confidence_result = max(probabilities)
        return guess_result, empirical_mean, confidence_result
        # return guess_result, probabilities, confidence_result




# class Step2KPredAvg(BasePredictor):
#     def __init__(self, **kwargs):# model, tokenizer): 
#         for key, value in kwargs.items():
#             setattr(self, key, value)
    
    
#         if self.task not in DYNAMIC_PROMPT[self.task_structure]:
#             print ('couldent find any prompt template')
#             pass # prompt_class = BasePrompts() [initializes a classification task]
#         else:
#             prompt_class = DYNAMIC_PROMPT[self.task_structure][self.task](**kwargs)
#     # prompt_class._predict()
#     # prompt_class._confidence()

#     # this class should rally return a custom object in predict_and_confidence where we have an object called target which holds all information associated with prompt inference result

#     # def __init__(self, tokenizer, model, device, task, label_names):
#     #     self.tokenizer = tokenizer
#     #     self.model = model
#     #     self.device = device
#     #     self.task = task
#     #     self.label_names = label_names

#     # def predict_sentiment_and_verbal_confidence_2step_k_pred_avg(self,datapoint):
#     def predict_and_confidence(self, datapoint):
         
#         # if task not in ['sst2', 'ag_news', 'popQA']:
#         #     raise ValueError("Unsupported task. Please choose 'sst2', 'ag_news', or 'popQA'.")

#         # Custom processing for PopQA
#         if self.task == 'popQA': 
#             data_point = {'question':datapoint[0], 'possible_answers':datapoint[1]}
#             text = data_point['question']

#             possible_answers = json.loads(data_point['possible_answers'])
#             label_names = possible_answers['correct'] + possible_answers['incorrect']
#             print ('label_names', label_names)

#             expected_prediction = possible_answers['correct']
#             incorrect_answers = possible_answers['incorrect']
#             guess_pattern = '|'.join(label_names)
#             class_number = len(label_names)

#             # Create task-specific dictionaries for PopQA
#             task_dictionary_counts_correct = {self.task:{label: 0 for label in expected_prediction}}
#             task_dictionary_counts_incorrect = {self.task:{label: 0 for label in incorrect_answers}}

#             task_dictionary_confidences = {self.task:{label: 0 for label in label_names}}
#             task_dictionary_confidences['null'] = 0

#             task_dictionary_counts = {self.task:{label: 0 for label in label_names}}
#             task_dictionary_counts['null'] = 0

#             # Create a sample list for examples in the prompt
#             sampled_answers = random.sample(label_names, min(5, len(label_names)))
#             sampled_answers_str = ', '.join(label_names)
#             sampled_confidences_str =', '.join(self.confidence_type_dict)
#             print ('sampled_confidences_str',sampled_confidences_str)
#             prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following question. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses; not a complete sentence, just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: ${data_point['question']} Guesses:{self.end_prompt_footer}"""
#         else:
#             print ('datapoint',datapoint)
#             if self.task == 'sst2':
#                 # guess_pattern = r'(positive|POSITIVE|Positive|negative|NEGATIVE|Negative)'
#                 class_number = 2
#                 label_list = self.dataset.label_names  # ['positive', 'negative']
#                 label_index = datapoint[1]
#                 guess_pattern = '|'.join([f'\\b{label}\\b' for label in label_list])
#             elif self.task == 'ag_news':
#                 # guess_pattern = r'(world|business|tech|science|sports)'
#                 guess_pattern = r'(world|business|sci/tech|sports)'
#                 class_number = 4
#                 label_list = [lab.lower() for lab in self.dataset.label_names] 
#                 label_index = datapoint[1]
#                 guess_pattern = '|'.join([f'\\b{label}\\b' for label in label_list])
#             elif self.task == 'strategyQA':
#                 guess_pattern = r'(true|TRUE|True|false|FALSE|False)'
#                 class_number = 2
#                 label_list = self.dataset.label_names # ['false', 'true']
#                 label_index = 1 if datapoint[1] else 0
#                 guess_pattern = '|'.join([f'\\b{label}\\b' for label in label_list])
            
#             print ('label_index',label_index,label_list) 
#             text = datapoint[0]
#             # expected_prediction = [label_list[label_index]]
#             # print ('sets',set(label_list) , set([label_list[label_index]])) 
#             # incorrect_answers = list (set(label_list) -  set([label_list[label_index]])) 
#             # print ('text', text)
#             # print ('expec', expected_prediction) 
#             # print ('expected_prediction',expected_prediction)
#             # print ('incorrect_answers',incorrect_answers)

#             expected_prediction, incorrect_answers  = identify_correct_incorrect_labels(label_list,label_index)

#             task_dictionary_counts = {self.task: {label: 0 for label in label_list}}
#             task_dictionary_counts[self.task]['null'] = 0

#             task_dictionary_confidences = {self.task: {label: 0 for label in label_list}}
#             task_dictionary_confidences[self.task]['null'] = 0
#             # Create task-specific dictionaries for PopQA
#             task_dictionary_counts_correct = {self.task :{label: 0 for label in expected_prediction}}
#             task_dictionary_counts_incorrect = {self.task:{label: 0 for label in incorrect_answers}}

#             sampled_confidences_str =', '.join(self.confidence_type_dict)
#             print ('sampled_confidences_str',sampled_confidences_str)
            
#             if self.task == 'sst2':
#                 prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a comma, for example [Negative, Positive, Positive, Negative ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""

#             elif self.task == 'ag_news':
#                 # options = 'world, business, tech, science, sports'  # we separate tech/science into two different prediction categories, but treat them as one label
#                 options = 'world, business, sci/tech, sports'  # we separate tech/science into two different prediction categories, but treat them as one label
                 
#                 prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} guess for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {options}; not a complete sentence, just the guesses! Separated by a comma, for example [Sports, Business, Sports, World, Sci/Tech, World ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
#             elif self.task == 'strategyQA':
#                 # prompt = f"""{self.start_prompt_header}Provide your {k_pred} best guess for the following text (false, true). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either true orfalse; not a complete sentence, just the guesses! Separated by a comma, for example [True, False, False, True ...]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
#                 prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses, for example [True, False, False, True ...]. Only output your answer nothing else!\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
#                 # prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. Only output your answer nothing else!\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
                
#                 # prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false?\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
        
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)

#         generate_args = {
#             "input_ids": inputs['input_ids'],
#             "attention_mask": inputs['attention_mask'],
#             "do_sample": True,  # enable sampling
#             "top_k": 40,  # top-k sampling
#             "top_p": 0.92,  # nucleus sampling probability
#             "temperature": 0.7,  # sampling temperature
#             "max_new_tokens": 300,
#             'pad_token_id': self.tokenizer.eos_token_id
#         }

#         with torch.no_grad():
#             outputs = self.model.generate(**generate_args)

#         prompt_length = len(inputs['input_ids'][0])
#         generated_tokens = outputs[0][prompt_length:]
#         generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
#         print('prompt', prompt)
#         print("Generated Prediction Text:", generated_text) 


#         # Regex to find 'true' or 'false', case-insensitive, ensuring full word match
#         # pattern = re.compile(r'\btrue\b|\bfalse\b', re.IGNORECASE)
#         pattern = re.compile(guess_pattern, re.IGNORECASE)
#         # Find all matches in the text
#         matches = pattern.findall(generated_text)
#         # Convert all matches to lowercase (optional, for consistency)
#         results = [match.lower() for match in matches]



#         # # Extract guesses, assuming they're separated by commas and ignoring case
#         # results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
#         # results = [result for result in results if result in label_list]# else 'null' for result in results]
#         # # If fewer results than k_pred, fill with 'null'
#         results.extend(['null'] * (self.k_pred - len(results)))



#         print('results', results, expected_prediction)
#         correct_predictions = sum(1 for sentiment in results if sentiment in expected_prediction)
#         confidence_empirical = (correct_predictions / len(results)) * 100
#         print('correct_predictions',correct_predictions)
#         for result in results:
#             if result in task_dictionary_counts[self.task]:
#                 task_dictionary_counts[self.task][result] += 1
#             else:
#                 task_dictionary_counts[self.task]['null'] +=1
            
#         for sentiment, number_of_results in task_dictionary_counts[self.task].items():
#             task_dictionary_confidences[sentiment] = (number_of_results / len(results))

        
        

#         for result in results:
#             if result in task_dictionary_counts_correct[self.task]:
#                 task_dictionary_counts_correct[self.task][result]+=1
#             elif result in task_dictionary_counts_incorrect[self.task]:
#                 task_dictionary_counts_incorrect[self.task][result]+=1

#         print(f"Results for '{text}':")
#         print(f"Counter: {task_dictionary_counts}")
#         print(f"Empirical confidence: {confidence_empirical}%")
#         max_class = max(task_dictionary_counts[self.task], key=task_dictionary_counts[self.task].get)
#         print('max_class', max_class, expected_prediction)
#         print ('task_dictionary_counts_correct[task]',task_dictionary_counts_correct[self.task])
#         print ('task_dictionary_counts_incorrect[task]', task_dictionary_counts_incorrect[self.task])

#         guess_result = max_class
#         guesses_output = results
        
#         if self.task == 'sst2':
#             # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [{sampled_confidences_str} ...]> Confidences:{self.end_prompt_footer}"""
#             confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        
#         elif self.task == 'ag_news':
#             # confidence_prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guesses for the following news article ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: \n\nThe text is:${text} Guesses: {guesses_output} Provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences:  Confidences:{self.end_prompt_footer}"""
#             confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text ({options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        
#         elif self.task == 'popQA':
#             confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text. Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${data_point['question']}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever; just the confidence! Separated by a coma, for example [{sampled_confidences_str} ...]> Confidences:{self.end_prompt_footer}"""
#         elif self.task == 'strategyQA':
#             # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {k_pred} best guesses for the following text (false, true). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either true or false; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
#             # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. Only output your answer nothing else!\n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, without any extra commentary whatsoever, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
#             confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. \n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. \n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_type_dict} that your guesses are correct, for example [{sampled_confidences_str} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
#             # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guess.\n\nStatement: {text}$. The guess were: {guesses_output}, given the guess provide the verbal confidences that your guess us correct. \n\nFor example:\n\Confidence: <the confidence, from either {self.confidence_type_dict} that your guess is correct, for example [{sampled_confidences_str} ...]; just the confidence!> Confidence:{self.end_prompt_footer}"""
        
#         print('confidence_prompt', confidence_prompt)
#         inputs = self.tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
#         generate_args = {
#             "input_ids": inputs['input_ids'],
#             "attention_mask": inputs['attention_mask'],
#             "do_sample": True,
#             "top_k": 40,
#             "top_p": 0.92,
#             "temperature": 0.7,
#             "max_new_tokens": 200,
#             'pad_token_id': self.tokenizer.eos_token_id
#         }

#         with torch.no_grad():
#             outputs = self.model.generate(**generate_args)

#         prompt_length = len(inputs['input_ids'][0])
#         generated_tokens = outputs[0][prompt_length:]
#         generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
#         print("Generated Confidence Text:", generated_text) 

#         confidence_options = '|'.join(self.confidence_type_dict) 
#         confidence_guesses = re.findall(confidence_options, generated_text, flags=re.IGNORECASE)
#         confidence_guesses = [match.lower() for match in confidence_guesses]
#         print('confidence_guesses', confidence_guesses)
#         confidence_list = self.confidence_type_dict
#         print('confidence_list', confidence_list)
#         confidence_results = [result for result in confidence_guesses if result in confidence_list]
#         confidence_results.extend(['null'] * (self.k_pred - len(confidence_results)))
#         confidence_map = self.confidence_map_dict

#         confidence_numerical_results = [
#             confidence_map[result] if result != 'null' else min(confidence_map.values()) 
#             for result in confidence_results
#         ] 
#         print('confidence_numerical_results', confidence_numerical_results) 
#         print('guesses_output', guesses_output)

#         weighted_counts = {label: 0.0 for label in label_list}
#         weighted_counts['null'] = 0.0


#         for sentiment, confidence in zip(guesses_output, confidence_numerical_results):
#             if confidence:
                
#                 weighted_counts[sentiment] += confidence
#             else:
#                 weighted_counts[sentiment] += 1
        

#         for result, confidence in zip(guesses_output, confidence_numerical_results):
#             if result in task_dictionary_counts_correct[self.task]:
#                 task_dictionary_counts_correct[self.task][result]+=confidence
#             elif result in task_dictionary_counts_incorrect[self.task]:
#                 task_dictionary_counts_incorrect[self.task][result]+=confidence
        
#         weighted_counts_binary = {'incorrect':sum(task_dictionary_counts_incorrect[self.task].values()),
#                                 'correct':sum(task_dictionary_counts_correct[self.task].values()),
#                                 'null':weighted_counts['null']}
#         #for all datasets have a correct, incorrect and null bucket

#         def compute_dirichlet_statistics(weighted_counts, label_list):
#             print('weighted_counts', weighted_counts)

#             # You mentioned 'weighted_counts_binary' in your request, but it seems missing.
#             # Assuming it's another dictionary similar to weighted_counts. For now, we skip this.
#             # print('weighted_counts_binary', weighted_counts_binary)  

#             alpha_prior = 1.0
#             alpha = {label: weighted_counts[label] + alpha_prior for label in label_list}
#             alpha['null'] = weighted_counts['null'] + alpha_prior

#             alpha_values = list(alpha.values())
#             sample_size = 1000
#             dirichlet_distribution = dirichlet(alpha_values, size=sample_size)
#             samples_ternary = [(p[0], p[1], p[2]) for p in dirichlet_distribution]  
#             empirical_mean = np.mean(dirichlet_distribution, axis=0)
#             empirical_mean_ternary = (empirical_mean[0], empirical_mean[1], empirical_mean[2])
#             print('empirical_mean', empirical_mean) 

#             def dirichlet_variance(alpha):
#                 alpha_0 = sum(alpha)
#                 variances = [(alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1)) for alpha_i in alpha]
#                 return variances

#             alpha_vector = list(alpha.values())
#             second_order_uncertainty = dirichlet_variance(alpha_vector)
#             probabilities = dirichlet_distribution[0] 
            
#             print("Counts:", task_dictionary_counts)
#             print("Numerical Confidences:", confidence_numerical_results)
#             print("Weighted Counts:", weighted_counts)
#             print("Alpha Vector:", alpha_vector)
#             print("Probabilities:", probabilities)
#             print("Second Order Uncertainty:", second_order_uncertainty) 
#             return alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities 

#         alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts,weighted_counts.keys())
#         # alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts_binary,weighted_counts_binary.keys())
        
#         confidence_result = max(probabilities)


#         # sys.exit()
#         return guess_result, empirical_mean, confidence_result
#         # return guess_result, probabilities, confidence_result



# prompts
# -baseprompt
# -ag_news (2step_vc, 1step_pred,)
# -sst2
# -strategyQA