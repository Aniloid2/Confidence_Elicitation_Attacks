 
from .base_classification_prompts import BaseClassificationPrompt

class RTEPrompts(BaseClassificationPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



        self.label_index = None   
        self.task_dictionary_counts_correct = None
        self.task_dictionary_counts_incorrect = None
         

    def _predict_prompt(self, text):
        
        prompt = f"""{self.start_prompt_header}You are an AI language model trained to perform natural language inference tasks. Given two sentences, your task is to determine whether the first sentence entails the second. Provide your {self.k_pred} best guess for the following text ({self.prediction_options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {self.prediction_options}; not a complete sentence, just the guesses! Separated by a comma, for example [{self.prediction_options} ... x{self.k_pred}]>\n\nSentence1 is: {text[0]}\n\nSentence2 is: {text[1]}\n\n Guesses:{self.end_prompt_footer}"""
        
        
        
        return prompt

    def _confidence_prompt(self, text, guesses_output):
        
        confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are an AI language model trained to perform natural language inference tasks. Given two sentences, your task is to determine whether the first sentence entails the second. Provide your {self.k_pred} best guess for the following text ({self.prediction_options}). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either {self.prediction_options}; not a complete sentence, just the guesses! Separated by a comma, for example [{self.prediction_options} ... x{self.k_pred}]>\n\nSentence1 is: {text[0]}\n\nSentence2 is: {text[1]}. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
        
        
        return confidence_prompt

    def _predict_and_confidence_prompt(self, text):
        
        prompt = f"[Placeholder for predict_and_confidence_prompt with text: {text}]"
        return prompt

    
    

    def cot_prompt(self, text):
        # Placeholder for chain-of-thought prompt
        prompt = f"[Placeholder for cot_prompt with text: {text}]"
        return prompt

    def _initialize_correct_incorrect_predictions(self, label_index):
        self.label_index = label_index
        expected_prediction, incorrect_answers = self._identify_correct_incorrect_labels(label_index)

        self.task_dictionary_counts_correct = {self.task: {label: 0 for label in expected_prediction}}
        self.task_dictionary_counts_incorrect = {self.task: {label: 0 for label in incorrect_answers}}

         
    
