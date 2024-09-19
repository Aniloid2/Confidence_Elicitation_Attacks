# from src.utils.shared.misc import identify_correct_incorrect_labels
from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP
import torch
class BaseClassificationPrompt:
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

        self._initialize_guess_pattern_prediction()
        self._initialize_guess_pattern_confidence()

        self.task_name_to_label = self._initialize_task_name_to_label()
        self.task_label_to_name = self._initialize_task_label_to_name()

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
        

    # def _identify_correct_incorrect_labels(self, label_index): 
    #     # This function would implement logic to find expected predictions and incorrect answers
    #     expected_prediction = [self.label_list[label_index]]
    #     incorrect_answers = [label for label in self.label_list if label != self.label_list[label_index]]
    #     return expected_prediction, incorrect_answers

    def _identify_correct_incorrect_labels(self, label_index):
        # Convert boolean to an index if necessary
        if isinstance(label_index, bool):
            label_index = int(label_index)  # True becomes 1, False becomes 0

        expected_prediction = [self.label_list[label_index]]
        incorrect_answers = list(set(self.label_list) - set([self.label_list[label_index]]))

        return expected_prediction, incorrect_answers

    def _initialize_guess_pattern_prediction(self):
        self.guess_pattern_prediction = '|'.join([f'\\b{label}\\b' for label in self.label_list])
    
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

    def _calculate_result_count_correct(self, results, weight): 
        for result in results:
            if result in self.task_dictionary_counts_correct[self.task]:
                self.task_dictionary_counts_correct[self.task][result]+= weight

    def _calculate_result_count_incorrect(self, results,weight):
        for result in results:
            if result in self.task_dictionary_counts_incorrect[self.task]:
                self.task_dictionary_counts_incorrect[self.task][result]+=weight

    def _add_confidence_weight_correct(self, results, confidence_numerical_results):

        for result, confidence in zip(results, confidence_numerical_results):
            if result in self.task_dictionary_counts_correct[self.task]:
                self.task_dictionary_counts_correct[self.task][result]+=confidence

    def _add_confidence_weight_incorrect(self, results, confidence_numerical_results):
        
        for result, confidence in zip(results, confidence_numerical_results):
            if result in self.task_dictionary_counts_incorrect[self.task]:
                self.task_dictionary_counts_incorrect[self.task][result]+=confidence

    def _lower_labels(self,label_list):
        label_list = [label.lower() for label in label_list]
        return label_list
    
    def _labels_add_null(self,label_list):
        label_list = label_list + ['null']
        return label_list
    
    def _extend_with_null(self,result_list):
        result_list.extend(['null'] * (self.k_pred - len(result_list)))
        return result_list

    def _call_model(self,generate_args ,inputs):
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)

        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("Generated Confidence Text:", generated_text)
        return generated_text 
