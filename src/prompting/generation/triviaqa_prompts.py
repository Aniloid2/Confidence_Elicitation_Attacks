 
from .base_generation_prompts import BaseGenerationPrompt
from rouge_score import rouge_scorer
from numpy.random import dirichlet
import numpy as np
import os
class TriviaQAPrompts(BaseGenerationPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



        self.label_index = None  # Placeholder, it should be set when calling prompt methods
        self.task_dictionary_counts_correct = None
        self.task_dictionary_counts_incorrect = None
         

    def _predict_prompt(self, text):
        question = text[0]
        context = text[1]
        # prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a comma, for example [Negative, Positive, Positive, Negative ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
        if self.k_pred == 1:
            prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following question. Give ONLY the guess, no other words or explanation. \n\nFor example:\n\nGuess: <most likely guess; just the guess! for example [Answer1]>\n\nThe question is: {question}, \n\n The context is: {context} \n\n Guesses:{self.end_prompt_footer}"""
        else:
            # prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ... x{self.k_pred}]>\n\nThe question is: {question}, \n\n The context is: {context} \n\n Guesses:{self.end_prompt_footer}"""
            prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ... x{self.k_pred}]>\n\nThe question is: {question}, \n\n Guesses:{self.end_prompt_footer}"""

        print ('predict_prompt:', prompt)

        return prompt

    def _confidence_prompt(self, text, guesses_output):
        question = text[0]
        context = text[0]
        # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        if self.k_pred == 1:
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of an answer being correct. The previeous prompt was $Provide your {self.k_pred} best guess for the following question. Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess; just the guess! Separated by a comma, for example [Answer1]>\n\nThe question is: {question}, \n\n The context is: {context} \n\n the guess was: {guesses_output} , given this guess provide the verbal confidences that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, from either {self.confidence_options} that your guess is correct, without any extra commentary whatsoever, for example [{self.confidence_options[0]}]; just the confidence! Separated by a coma> Confidence: {self.end_prompt_footer}"""
        
        else:
            # confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {question}, \n\n The context is: {context} \n\n the guesses were: {guesses_output} , given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences: {self.end_prompt_footer}"""
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best independent guesses for the following question. Give ONLY the guesses, no other words or explanation. Feel free to repeat the same answer.\n\nFor example:\n\nGuesses: <most likely guesses; just the guesses! Separated by a comma, for example [Answer1, Answer2, Answer3 ...]>\n\nThe question is: {question}, \n\n the guesses were: {guesses_output} , given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences: {self.end_prompt_footer}"""
        
        print ('confidence_prompt:',confidence_prompt)

        return confidence_prompt

    def _predict_and_confidence_prompt(self, text):
        # Placeholder for the predict_and_confidence prompt
        prompt = f"[Placeholder for predict_and_confidence_prompt with text: {text}]"
        return prompt

    
    

    def cot_prompt(self, text):
        # Placeholder for chain-of-thought prompt
        prompt = f"[Placeholder for cot_prompt with text: {text}]"
        return prompt

    def _identify_correct_incorrect_labels(self, label_index):
        # correct incorrect answers are based on a threshold
        return ['true'], ['false']
    
   
    def _calculate_rouge_score_max(self,prediction, references):
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        max_score = 0
        for ref in references:
            scores = scorer.score(prediction, ref)
            max_score = max(max_score, scores['rouge1'].fmeasure)
        return max_score

    def _answer_post_processing(self,results, answers):
        true_labels = []
        for guess in results:
            print ('tgt:',guess, answers)
            rouge1_score = self._calculate_rouge_score_max(guess, answers)
            print ('rouge1_score',rouge1_score)
            is_correct = 1 if rouge1_score >= 0.3 else 0
            true_labels.append(is_correct)
        
        true_labels = ['true' if i == 1 else 'false' for i in true_labels]
        return true_labels
    def _predictor_decision(self):
        # if there is at least one correct entry in the top k then we say it's successful
        #max(self.task_dictionary_counts[self.task], key=self.task_dictionary_counts[self.task].get)
        if self.task_dictionary_counts[self.task]['true'] != 0:
            return 'true' 
        else:
            return 'false'
        # return guess_result
    # def _aggregate(self, weighted_counts, results_post_process, confidence_numerical_results):
    #     print ('results_post_process, confidence_numerical_results',results_post_process, confidence_numerical_results)
    #     conf_to_num = {5:0.75, 4:0.6,3:0.5,2:0.4,1:0.25}
    #     second_order_uncertainty = None
    #     if results_post_process[0] == 'true':
    #         empirical_mean = [1-conf_to_num[confidence_numerical_results[0]], conf_to_num[confidence_numerical_results[0]]]
    #     else:
    #         empirical_mean = [conf_to_num[confidence_numerical_results[0]], 1-conf_to_num[confidence_numerical_results[0]]]
    #     probabilities = empirical_mean

    #     guess_result_with_confidence = results_post_process[0]
    #     return guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities



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



    def _initialize_correct_incorrect_predictions(self, label_index):
        self.label_index = label_index
        expected_prediction, incorrect_answers = self._identify_correct_incorrect_labels(label_index)

        self.task_dictionary_counts_correct = {self.task: {label: 0 for label in expected_prediction}}
        self.task_dictionary_counts_incorrect = {self.task: {label: 0 for label in incorrect_answers}}

         
    
