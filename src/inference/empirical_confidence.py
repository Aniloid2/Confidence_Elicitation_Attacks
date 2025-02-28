import json
import re
import numpy as np
import torch
from .base_predictor  import BasePredictor
from numpy.random import dirichlet 
from src.prompting.prompt_config import DYNAMIC_PROMPT 
import os




class EmpiricalConfidence(BasePredictor):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.technique_name = self.__class__.__name__
    
        if self.task not in DYNAMIC_PROMPT[self.task_structure]:
            
            print('Couldn\'t find any prompt template. This can be implemented in a generalized class to handle various prompt structures.')
            self.ceattack_logger.warning('Couldn\'t find any prompt template. This functionality could be implemented in a generalized class to handle various prompt structures.')
            pass 
        else:
            self.prompt_class = DYNAMIC_PROMPT[self.task_structure][self.task](**kwargs)
        
        
        self.predictor_container = None
        self.inference_step = None
        self.current_sample = None

        
    

    def _query_model_and_return_result_list(self,extra_args):
        generated_text = self.prompt_class._call_model(extra_args)
        generated_text = generated_text[0]
        print ('Generated text empirical:',generated_text)
        self.ceattack_logger.debug(f"Generated text empirical: \n {generated_text}")
        
        
        pattern = re.compile(self.prompt_class.guess_pattern_prediction, re.IGNORECASE) 
        matches = pattern.findall(generated_text) 
        results = [match.lower() for match in matches]
        return results

    def _perform_multiple_predictions(self,extra_args,  n=20):
        n = self.k_pred
        results = []
        for _ in range(n):
            result = self._query_model_and_return_result_list(extra_args)
            result = self.prompt_class._extend_with_null(result,1)
            results.append(result)
        return results
    
    
    def add_prompt_and_call_model(self, datapoint): 
        text = datapoint 
        self.prompt_class._initialize_sample_counters()  
        self.prompt_class._initialize_guess_pattern_prediction(datapoint,self.prompt_class.label_list)
        
        prompt = self.prompt_class._predict_prompt_singular(text )
 
         
        tokenizer_encoding_args = {
            'return_tensors':"pt",
            'truncation':True, 
            'max_length':2000,
        }

        generate_args = { 
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
 
        self.prompt_class.model.general_tokenizer_encoding_args = tokenizer_encoding_args
        self.prompt_class.model.general_generate_args = generate_args
        results = self._perform_multiple_predictions(extra_args) 
        self.ceattack_logger.debug(f"Results after multiple predictions: \n {results}")
        print ('Results after multiple predictions:',  results)
        results = [result for sublist in results for result in sublist]   
        generated_text = results
        generated_text_conf = []
        
        return {'raw_responses':f'{generated_text} {generated_text_conf}', 'predictions':str(generated_text),'confidences':str(generated_text_conf)}
    
    
    def standarize_output(self,output):
        generated_text = output['predictions']
        generated_text_conf = output['confidences']

        results = self.prompt_class._extract_answer_prompt(generated_text) 
        
        
        results_post_process = self.prompt_class._answer_post_processing(results) 
 

        self.prompt_class._extend_with_null(results_post_process) 
    
        self.prompt_class._calculate_result_counts(results_post_process)


        self.prompt_class._calculate_result_confidences(results_post_process)  
 

        self.ceattack_logger.debug(f"Counter: \n {self.prompt_class.task_dictionary_counts}")
        print(f"Counter: {self.prompt_class.task_dictionary_counts}")
        
        guess_result = self.prompt_class._predictor_decision()
        self.ceattack_logger.debug(f"Max class based on prediction: \n {guess_result}")
        print('Max class based on prediction:', guess_result)

        guesses_output = results

        confidence_numerical_results = [0]*len(results_post_process)

        self.ceattack_logger.debug(f"Numerical Confidence Result: \n {confidence_numerical_results}")
        print('Numerical Confidence Result:', confidence_numerical_results)  

        self.ceattack_logger.debug(f"Guessed output and label list: \n Guess:{confidence_numerical_results}, Label List: {self.prompt_class.label_list}")
        print(f"Guessed output and label list: Guess: {guesses_output}, Label List: {self.prompt_class.label_list}")


        return {'raw_responses': output['raw_responses'], 'predictions':results_post_process,'confidences':confidence_numerical_results}
    
    
    def aggregate_output(self, output):
        self.ceattack_logger.debug(f"Output that needs aggregation: \n Output: {output}")
        print ('Output that needs aggregation:' , output)

        results_post_process = output['predictions']
        confidence_numerical_results = output['confidences']
        # inference_step and current_sample_id only applicable if we are doing inference with a goal function class 
        if 'inference_step' in output:
            inference_step = output['inference_step']
        else:
            inference_step = None
        if 'current_sample_id' in output:
            current_sample_id = output['current_sample_id']
        else:
            current_sample_id = None
        weighted_counts = {label: 0.0 for label in self.prompt_class.label_list}
        weighted_counts['null'] = 0.0
        
        for pred, confidence in zip(results_post_process, confidence_numerical_results):
            if confidence:
                
                weighted_counts[pred] += confidence
            else:
                weighted_counts[pred] += 1
         
        self.ceattack_logger.debug(f"Weighted count of prediction weight plus confidence weight: \n Weights: {weighted_counts}")
        print ('Weighted count of prediction weight plus confidence weight:' , weighted_counts)
        guess_result_with_confidence = max(weighted_counts, key=weighted_counts.get) 

        def compute_dirichlet_statistics(weighted_counts, label_list,current_sample_id=None,inference_step=None):
            
            
            alpha_prior = 1.0
            alpha = {label: weighted_counts[label] + alpha_prior for label in label_list}
            alpha['null'] = weighted_counts['null'] + alpha_prior

            alpha_values = list(alpha.values())
            sample_size = 1000
            dirichlet_distribution = dirichlet(alpha_values, size=sample_size)
            
            empirical_mean = np.mean(dirichlet_distribution, axis=0)
             
            if self.ternary_plot == True:
                from src.utils.shared.plotting import ternary_plot, ternary_mean_plot
                if empirical_mean.shape[0] == 3:
                    if self.current_sample: 
                        samples_ternary = [tuple(p[i] for i in range(self.n_classes + 1)) for p in dirichlet_distribution]
                        empirical_means_ternary = (empirical_mean[0], empirical_mean[1], empirical_mean[2])
                        dirichlet_folder = os.path.join(self.test_folder, 'dirichlet')
                        
                        if not os.path.exists(dirichlet_folder):
                            os.makedirs(dirichlet_folder)
                        ternary_plot_file = os.path.join(dirichlet_folder, f'dirichlet_cs{current_sample_id}_is{inference_step}_a({alpha_values})_n{str(sample_size)}')
                        self.ceattack_logger.info(f'Plotting ternary plot for sample {ternary_plot_file}')
                        ternary_mean_plot(samples_ternary,alpha_values,empirical_means_ternary,ternary_plot_file)
                else:
                    self.ceattack_logger.warning('Not plotting ternary plot, as the number of classes is not 3')
            def dirichlet_variance(alpha):
                alpha_0 = sum(alpha)
                variances = [(alpha_i * (alpha_0 - alpha_i)) / (alpha_0 ** 2 * (alpha_0 + 1)) for alpha_i in alpha]
                return variances

            alpha_vector = list(alpha.values())
            second_order_uncertainty = dirichlet_variance(alpha_vector)
            probabilities = dirichlet_distribution[0] 
              
            print (f'Sample ID: {current_sample_id} and Inference step {inference_step}')
            print("Counts:", self.prompt_class.task_dictionary_counts)
            print("Numerical Confidences:", confidence_numerical_results)
            print("Weighted Counts:", weighted_counts)
            print("Alpha Vector:", alpha_vector)
            print('Empirical Mean', empirical_mean) 
            print("Probabilities:", probabilities)
            print("Second Order Uncertainty:", second_order_uncertainty) 

            self.ceattack_logger.info(f'Sample ID: {current_sample_id} and Inference step {inference_step}')
            self.ceattack_logger.info("Counts: %s", self.prompt_class.task_dictionary_counts)
            self.ceattack_logger.info("Numerical Confidences: %s", confidence_numerical_results)
            self.ceattack_logger.info("Weighted Counts: %s", weighted_counts)
            self.ceattack_logger.info("Alpha Vector: %s", alpha_vector)
            self.ceattack_logger.info('Empirical Mean: %s', empirical_mean) 
            self.ceattack_logger.info("Probabilities: %s", probabilities)
            self.ceattack_logger.info("Second Order Uncertainty: %s", second_order_uncertainty)



            return alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities 

        alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts,weighted_counts.keys(),current_sample_id,inference_step)
        return guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities
    
    
    def predict_and_confidence(self, datapoint):
        
        output = self.add_prompt_and_call_model(datapoint)
        output = self.standarize_output(output)
        guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities = self.aggregate_output(output)
        guess_result = self.prompt_class._predictor_decision()

        if self.predictor_container: 
            self.predictor_container.add_top_k_max_prediction(self.prompt_class.task_name_to_label[guess_result])

            self.predictor_container.add_top_k_max_prediction_and_confidence(self.prompt_class.task_name_to_label[guess_result_with_confidence])
            guess_result_empirical_mean = self.prompt_class.label_list_with_null[np.argmax(empirical_mean)]
            self.predictor_container.add_top_k_dirichlet_mean(self.prompt_class.task_name_to_label[guess_result_empirical_mean])
        


        confidence_result = max(probabilities)
        return guess_result, empirical_mean, confidence_result
    
    
    