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




class EmpiricalConfidence(BasePredictor):
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

        
    

    def _query_model_and_return_result_list(self,generate_args,extra_args):
        generated_text = self.prompt_class._call_model(generate_args,extra_args)

        # Regex to find 'true' or 'false', case-insensitive, ensuring full word match
        # pattern = re.compile(r'\btrue\b|\bfalse\b', re.IGNORECASE)
        pattern = re.compile(self.prompt_class.guess_pattern_prediction, re.IGNORECASE)
        # Find all matches in the text
        matches = pattern.findall(generated_text)
        # Convert all matches to lowercase (optional, for consistency)
        results = [match.lower() for match in matches]
        return results

    def _perform_multiple_predictions(self,generate_args,extra_args,  n=20):
        n = self.k_pred
        results = []
        for _ in range(n):
            result = self._query_model_and_return_result_list(generate_args,extra_args)
            result = self.prompt_class._extend_with_null(result,1)
            results.append(result)
        return results
    def predict_and_confidence(self, datapoint):
 

        text, label_index = datapoint
        self.prompt_class._initialize_sample_counters() 
        print ('label_index_text',text,label_index)
        expected_prediction, incorrect_answers = self.prompt_class._identify_correct_incorrect_labels(label_index)
        self.prompt_class._initialize_correct_incorrect_predictions(label_index)
        self.prompt_class._initialize_guess_pattern_prediction(datapoint,self.prompt_class.label_list)
        
        prompt = self.prompt_class._predict_prompt_singular(text )
 
        
        

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
        extra_args = {
            "prompt": prompt,
        }

        # with torch.no_grad():
        #     outputs = self.model.generate(**generate_args)

        # prompt_length = len(inputs['input_ids'][0])
        # generated_tokens = outputs[0][prompt_length:]
        # generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print('prompt', prompt)
        # print("Generated Prediction Text:", generated_text) 
        results = self._perform_multiple_predictions(generate_args, extra_args)
        # in this case results in a list of results
        print ('results',  results)
        results = [result for sublist in results for result in sublist]
        # results = [result[0] for result in results]
        print ('results', results)

        # generated_text = self.prompt_class._call_model(self.generate_args,inputs)

        # # Regex to find 'true' or 'false', case-insensitive, ensuring full word match
        # # pattern = re.compile(r'\btrue\b|\bfalse\b', re.IGNORECASE)
        # pattern = re.compile(self.prompt_class.guess_pattern_prediction, re.IGNORECASE)
        # # Find all matches in the text
        # matches = pattern.findall(generated_text)
        # # Convert all matches to lowercase (optional, for consistency)
        # results = [match.lower() for match in matches]



        # # Extract guesses, assuming they're separated by commas and ignoring case
        # results = [guess.lower() for guess in re.split(r'\s*,\s*', generated_text.strip())]
        # results = [result for result in results if result in label_list]# else 'null' for result in results]
        # # If fewer results than k_pred, fill with 'null' 
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

         

        weighted_counts = {label: 0.0 for label in self.prompt_class.label_list}
        weighted_counts['null'] = 0.0


        for pred in guesses_output:  
            weighted_counts[pred] += 1
        
        guess_result_with_confidence = max(weighted_counts, key=weighted_counts.get) 

         
        weighted_counts_binary = {'incorrect':sum(self.prompt_class.task_dictionary_counts_incorrect[self.task].values()),
                                'correct':sum(self.prompt_class.task_dictionary_counts_correct[self.task].values()),
                                'null':weighted_counts['null']}
        #for all datasets have a correct, incorrect and null bucket

        def compute_dirichlet_statistics(weighted_counts, label_list):
            print(' ', weighted_counts)

            # You mentioned 'weighted_counts_binary' in your request, but it seems missing.
            # Assuming it's another dictionary similar to weighted_counts. For now, we skip this.
            # print('weighted_counts_binary', weighted_counts_binary)  

            alpha_prior = 0.0001
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
            # print("Numerical Confidences:", confidence_numerical_results)
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

 