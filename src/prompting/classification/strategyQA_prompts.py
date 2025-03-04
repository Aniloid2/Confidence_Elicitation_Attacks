# from src.utils.shared.misc import identify_correct_incorrect_labels

from .base_classification_prompts import BaseClassificationPrompt

class StrategyQAPrompts(BaseClassificationPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



        self.label_index = None  
        self.task_dictionary_counts_correct = None
        self.task_dictionary_counts_incorrect = None
         
    def _predict_prompt(self, text):
        if self.k_pred == 1:
            prompt = f"""{self.start_prompt_header}You are a factual question answering model. Is the following statement true or false? Output one guess, for example [True] or [False]. Only output your answer, nothing else!\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
 
        else:
            prompt = f"""{self.start_prompt_header}You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses, for example [True, False, False, True ...]. Only output your answer nothing else!\n\nStatement: {text} Answer:{self.end_prompt_footer}"""
 
        return prompt

    def _confidence_prompt(self, text, guesses_output):
        if self.k_pred == 1:
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of an answer being correct. The previous prompt was: "You are a factual question answering model. Is the following statement true or false? Output one guess." \n\nStatement: {text}. The guess was: {guesses_output}. Given this guess, provide the verbal confidence that your guess is correct. \n\nFor example:\n\nConfidence: <the confidence, chosen from {self.confidence_options}, that your guess is correct, for example [{self.confidence_options[0]}]; just the confidence!> Confidence:{self.end_prompt_footer}"""
        else:
            confidence_prompt = f"""{self.start_prompt_header}You're a model that needs to give the confidence of answers being correct. The previeous prompt was $You are a factual question answering model, Is the following statement true or false? output {self.k_pred} guesses. \n\nStatement: {text}$. The guesses were: {guesses_output}, given these guesses provide the verbal confidences that your guesses are correct. \n\nFor example:\n\Confidences: <the confidences, from either {self.confidence_options} that your guesses are correct, for example [{self.confidence_options} ...]; just the confidence! Separated by a coma> Confidences:{self.end_prompt_footer}"""
 
        return confidence_prompt

    def _predict_and_confidence_prompt(self, text):
         
        prompt = f"[Placeholder for predict_and_confidence_prompt with text: {text}]"
        return prompt

    def cot_prompt(self, text):
         
        prompt = f"[Placeholder for cot_prompt with text: {text}]"
        return prompt

    def _initialize_correct_incorrect_predictions(self, label_index):
        self.label_index = label_index
        expected_prediction, incorrect_answers = self._identify_correct_incorrect_labels(label_index)

        self.task_dictionary_counts_correct = {self.task: {label: 0 for label in expected_prediction}}
        self.task_dictionary_counts_incorrect = {self.task: {label: 0 for label in incorrect_answers}}

         
    
