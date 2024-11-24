from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP
import torch
import re
from numpy.random import dirichlet
import numpy as np
import os
class BaseGenerationPrompt:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            
            setattr(self, key, value)
            
        self.class_number = self.n_classes 
        if not hasattr(self, 'dataset') or not hasattr(self.dataset, 'label_names'):
            raise ValueError('Dataset or dataset label names are not provided')
        
        
        self.label_list = self._lower_labels(self.dataset.label_names) #[label.lower() for label in self.dataset.label_names]  # ['positive', 'negative']
        self.label_list_with_null = self._labels_add_null(self.label_list)
        self.confidence_list = CONFIDENCE_LEVELS[self.confidence_type]
        self.confidence_map = CONFIDENCE_MAP[self.confidence_type]
        # options for prompts to choose from
        self.prediction_options = ', '.join(self.label_list) 
        self.confidence_options = ', '.join(self.confidence_list) 

        # self._initialize_guess_pattern_prediction() # guess pattern is dynamic
        self._initialize_guess_pattern_confidence()

        self.task_name_to_label = self._initialize_task_name_to_label()
        self.task_label_to_name = self._initialize_task_label_to_name()
    
    def _initialize_guess_pattern_prediction(self,datapoint,label_list):
        text, answers = datapoint
        self.guess_pattern_prediction = '|'.join([f'\\b{label}\\b' for label in answers])
    
    def _initialize_guess_pattern_confidence(self):
        self.guess_pattern_confidence = '|'.join([f'\\b{confidence}\\b' for confidence in self.confidence_list])

    def _initialize_task_name_to_label(self):
        task_name_to_label = {label: i for i,label in enumerate(self.label_list)}
        task_name_to_label['null'] = self.n_classes
        return task_name_to_label
    
 
    def _initialize_task_label_to_name(self):
        task_label_to_name = {i: label for i,label in enumerate(self.label_list)}
        task_label_to_name[self.n_classes] = 'null'
        return task_label_to_name

    def _lower_labels(self,label_list):
        label_list = [label.lower() for label in label_list]
        return label_list

    def _labels_add_null(self,label_list):
        label_list = label_list + ['null']
        return label_list

    def _extend_with_null(self,result_list):
        result_list.extend(['null'] * (self.k_pred - len(result_list)))
        return result_list
    
    def _aggregate(self, weighted_counts,results_post_process,confidence_numerical_results):
        for pred, confidence in zip(results_post_process, confidence_numerical_results):
            if confidence:
                
                weighted_counts[pred] += confidence
            else:
                weighted_counts[pred] += 1
        
        print ('weighted_counts+conf',weighted_counts)
        guess_result_with_confidence = max(weighted_counts, key=weighted_counts.get) 

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
 
            if self.ternary_plot == True:
                from src.utils.shared.plotting import ternary_plot, ternary_mean_plot
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
            
            print("Counts:", self.task_dictionary_counts)
            print("Numerical Confidences:", confidence_numerical_results)
            print("Weighted Counts:", weighted_counts)
            print("Alpha Vector:", alpha_vector)
            print("Probabilities:", probabilities)
            print("Second Order Uncertainty:", second_order_uncertainty) 
            return alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities 

        alpha, dirichlet_distribution, empirical_mean, second_order_uncertainty, probabilities = compute_dirichlet_statistics(weighted_counts,weighted_counts.keys())
        return guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities



    def _initialize_sample_counters(self):
        self.task_dictionary_counts = {self.task: {label: 0 for label in self.label_list}}
        self.task_dictionary_counts[self.task]['null'] = 0
        self.task_dictionary_confidences = {self.task: {label: 0 for label in self.label_list}}
        self.task_dictionary_confidences[self.task]['null'] = 0 
    
    # def _identify_correct_incorrect_labels(self, label_index):
    #     # Convert boolean to an index if necessary
    #     if isinstance(label_index, bool):
    #         label_index = int(label_index)  # True becomes 1, False becomes 0

    #     expected_prediction = [self.label_list[label_index]]
    #     incorrect_answers = list(set(self.label_list) - set([self.label_list[label_index]]))

    #     return expected_prediction, incorrect_answers

    def _extract_answer_prompt(self,generated_text): 
        results = [sentences.strip().lower() for sentences in generated_text.split(',') if sentences.strip()]
        return results
    
    def _extract_predictions(self, results, expected_prediction):
        return None, None
    
    def _calculate_result_counts(self, results):

        for result in results:
            if result in self.task_dictionary_counts[self.task]:
                self.task_dictionary_counts[self.task][result] += 1
            else:
                self.task_dictionary_counts[self.task]['null'] +=1
        
    def _calculate_result_confidences(self, results):
        pass
        # for pred, number_of_results in self.task_dictionary_counts[self.task].items():
        #     self.task_dictionary_confidences[pred] = (number_of_results / len(results))

    def _calculate_result_count_correct(self, results, weight): 
        pass
        # for result in results:
        #     if result in self.task_dictionary_counts_correct[self.task]:
        #         self.task_dictionary_counts_correct[self.task][result]+= weight

    def _calculate_result_count_incorrect(self, results,weight):
        pass
    
    def _add_confidence_weight_correct(self,results, confidence_numerical_results):
        pass

    def _add_confidence_weight_incorrect(self,results, confidence_numerical_results):
        pass
    
    def _call_model(self,generate_args,extra_args):
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)

        # prompt_length = len(inputs['input_ids'][0])
        prompt_length = len(generate_args['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("Generated Confidence Text:", generated_text)
        return generated_text 