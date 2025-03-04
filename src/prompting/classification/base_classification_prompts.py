
from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP
import torch
import re
 
class BaseClassificationPrompt:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            
            setattr(self, key, value)
            
        self.class_number = self.n_classes 
        if not hasattr(self, 'dataset') or not hasattr(self.dataset, 'label_names'):
            raise ValueError('Dataset or dataset label names are not provided')
        
        
        self.label_list = self._lower_labels(self.dataset.label_names)  
        self.label_list_with_null = self._labels_add_null(self.label_list)
 
        self.confidence_list = CONFIDENCE_LEVELS[self.confidence_type]
        self.confidence_map = CONFIDENCE_MAP[self.confidence_type]

        
        
        self.prediction_options = ', '.join(self.label_list) 
        self.confidence_options = ', '.join(self.confidence_list)  
        
        self._initialize_guess_pattern_confidence()

        self.task_name_to_label = self._initialize_task_name_to_label()
        self.task_label_to_name = self._initialize_task_label_to_name()


    def _predict_prompt_singular(self, text):
        
        prompt = f"""{self.start_prompt_header}Provide your best guess for the following text ({self.prediction_options}). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either {self.prediction_options}; not a complete sentence, just the guess!>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""

        return prompt
       
    def _predict_prompt(self, text):
        
        prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following text ({self.prediction_options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either ({self.prediction_options}); not a complete sentence, just the guesses! Separated by a comma, for example [{self.prediction_options} ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""

        return prompt

    def _confidence_prompt(self, text, guesses_output):
        
        confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text ({self.prediction_options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either ({self.prediction_options}); not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""

        return confidence_prompt

    def _initialize_task_name_to_label(self):
        task_name_to_label = {label: i for i,label in enumerate(self.label_list)}
        task_name_to_label['null'] = self.n_classes
        return task_name_to_label
    
 
    def _initialize_task_label_to_name(self):
        task_label_to_name = {i: label for i,label in enumerate(self.label_list)}
        task_label_to_name[self.n_classes] = 'null'
        return task_label_to_name


    def _initialize_sample_counters(self):
        self.task_dictionary_counts = {self.task: {label: 0 for label in self.label_list}}
        self.task_dictionary_counts[self.task]['null'] = 0
        self.task_dictionary_confidences = {self.task: {label: 0 for label in self.label_list}}
        self.task_dictionary_confidences[self.task]['null'] = 0 
        

    def _identify_correct_incorrect_labels(self, label_index):
        if isinstance(label_index, bool):
            label_index = int(label_index)  

        expected_prediction = [self.label_list[label_index]]
        incorrect_answers = list(set(self.label_list) - set([self.label_list[label_index]]))

        return expected_prediction, incorrect_answers

    def _initialize_guess_pattern_prediction(self,datapoint,label_list):
        self.guess_pattern_prediction = '|'.join([f'\\b{label}\\b' for label in label_list])
    
    def _initialize_guess_pattern_confidence(self):
        self.guess_pattern_confidence = '|'.join([f'\\b{confidence}\\b' for confidence in self.confidence_list])

    def _calculate_result_counts(self, results):

        for result in results:
            if result in self.task_dictionary_counts[self.task]:
                self.task_dictionary_counts[self.task][result] += 1
            else:
                self.task_dictionary_counts[self.task]['null'] +=1
        
    def _calculate_result_confidences(self, results):
        for pred, number_of_results in self.task_dictionary_counts[self.task].items():
            self.task_dictionary_confidences[pred] = (number_of_results / len(results))

    def _lower_labels(self,label_list):
        label_list = [label.lower() for label in label_list]
        return label_list
    
    def _labels_add_null(self,label_list):
        label_list = label_list + ['null']
        return label_list
    
    def _extend_with_null(self,result_list, local_k_pred=None):
        if local_k_pred:
            result_list.extend(['null'] * (local_k_pred - len(result_list)))
        else:
            result_list.extend(['null'] * (self.k_pred - len(result_list)))
        return result_list
    
    def _extract_answer_prompt(self,generated_text): 
        pattern = re.compile(self.guess_pattern_prediction, re.IGNORECASE)
        
        
        matches = pattern.findall(generated_text)
        
        results = [match.lower() for match in matches]
        return results
    
    def _extract_predictions(self, results, expected_prediction): 
        correct_predictions = sum(1 for pred in results if pred in expected_prediction)
        confidence_empirical = (correct_predictions / len(results)) * 100 
        return correct_predictions, confidence_empirical
    
    def _answer_post_processing(self,results):
        return results
    
    def _predictor_decision(self):
        guess_result = max(self.task_dictionary_counts[self.task], key=self.task_dictionary_counts[self.task].get)
        return guess_result
    
    def _label_to_index(self,label):
        position = self.label_list_with_null.index(label)
        return position


    def _call_model(self,extra_args):
        with torch.no_grad(): 
            outputs = self.model(extra_args['prompt'])
         
        return outputs
 