 
from .base_classification_prompts import BaseClassificationPrompt

class SST2Prompts(BaseClassificationPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



        self.label_index = None  # Placeholder, it should be set when calling prompt methods
        self.task_dictionary_counts_correct = None
        self.task_dictionary_counts_incorrect = None
         




    def _predict_prompt(self, text):
        if self.k_pred == 1:
            prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guess! Separated by a comma, for example [Negative]>\n\nThe text is:${text} Guess:{self.end_prompt_footer}"""
        else:
            prompt = f"""{self.start_prompt_header}Provide your {self.k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guesses, either positive or negative; not a complete sentence, just the guesses! Separated by a comma, for example [Negative, Positive, Positive, Negative ... x{self.k_pred}]>\n\nThe text is:${text} Guesses:{self.end_prompt_footer}"""
        print ('predict_prompt:', prompt)

        return prompt

    def _confidence_prompt(self, text, guesses_output):
        if self.k_pred == 1:
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of an answer being correct. The previeous prompt was $Provide your {self.k_pred} best guess for the following text (positive, negative). Give ONLY the guess, no other words or explanation.\n\nFor example:\n\nGuess: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guess was: {guesses_output}, given this guess provide the verbal confidence that your guess is correct. Give ONLY the verbal confidence, no other words or explanation.\n\nFor example:\n\Confidence: <the confidence, from either {self.confidence_options} that the guess is correct, without any extra commentary whatsoever, for example [{self.confidence_options[0]}]; just the confidence! Separated by a coma> Confidence:{self.end_prompt_footer}"""
        
        else:
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $Provide your {self.k_pred} best guesses for the following text (positive, negative). Give ONLY the guesses, no other words or explanation.\n\nFor example:\n\nGuesses: <most likely guess, either positive or negative; not a complete sentence, just the guesses!>\n\nThe text is:${text}$ the guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. Give ONLY the verbal confidences, no other words or explanation.\n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, without any extra commentary whatsoever, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
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

    def _initialize_correct_incorrect_predictions(self, label_index):
        self.label_index = label_index
        expected_prediction, incorrect_answers = self._identify_correct_incorrect_labels(label_index)

        self.task_dictionary_counts_correct = {self.task: {label: 0 for label in expected_prediction}}
        self.task_dictionary_counts_incorrect = {self.task: {label: 0 for label in incorrect_answers}}

         
    
