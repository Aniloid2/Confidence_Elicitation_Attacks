
from textattack.transformations import WordSwap
from textattack.shared import AttackedText

import torch
import re


class SelfWordSubstitutionW1(WordSwap):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    """
    This implementation is from the W1 attack in AN LLM CAN FOOL ITSELF: A PROMPT-BASED ADVERSARIAL ATTACK https://arxiv.org/pdf/2310.13345

    In the original paper, the authors report an approximate 5% attack success rate, which is roughly consistent with our own findings. However, 
    we also observe that models such as llama3 and Mistral struggle to swap words while ensuring that the class switch maintains the original sentiment. 
    As a result, they often substitute a word with its antonym. In the original paper, the attack is evaluated by checking the classification label in 
    the generated output. For example, given a prompt like 'change one word in the sentence “this is a good day” so that the sentiment class shifts from 
    positive to negative while preserving the semantics,' the typical output might be: 'I changed the word “good” to “dull” so the new sentence is “this 
    is a dull day,” which preserves the semantics while altering the sentiment to negative.' The authors then determine the final prediction by taking the 
    majority label from the generated output. In our work, we follow previous methodologies by extracting the new adversarial sentence, passing it through 
    a semantic similarity encoder to check the semantics remain intact, and finally performing inference again.
    """



        
    def _query_model(self, prompt):
        
        tokenizer_args = {'return_tensors':"pt", 
                          'truncation':True, 
                          'max_length':2000}
        
        generate_args = {
            
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            
        }

        # Generate the output with the model
        self.model.general_tokenizer_encoding_args = tokenizer_args
        self.general_generate_args = generate_args
        with torch.no_grad():
            generated_text = self.model(prompt)[0]
            
            
        return generated_text.strip()

    
    

    def _generate_prompt(self, context_sentence, label_index):
        
        label_list =  self.dataset.label_names 
        
        expected_prediction, filtered_label_list = self.predictor.prompt_class._identify_correct_incorrect_labels(label_index)
            
        other_classes = ' or '.join(filtered_label_list)
        attack_type = 'Replace at most two words in the sentence with synonyms.'
        original_input = f"The original sentence '{context_sentence}' is classified as {expected_prediction}. "
        attack_objective = (
            f"Your task is to generate a new sentence which must satisfy the following conditions:\n"
            f"1) Keeping the semantic meaning of the new sentence unchanged.\n"
            f"2) The new sentence and the original sentence should be classified as {other_classes}.\n "
        )
        if self.prompt_shot_type == 'zs':
            attack_guidance = (
                f"You can finish the task by modifying sentence using the following guidance: {attack_type} "
                f"Only output the new sentence without anything else."
                f"The new sentence is: [INSERT YOUR SENTENCE TRANSFORMATION HERE]"
            )
        elif self.prompt_shot_type == 'fs':
            original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work', 'The sun is shining brightly', 'He cooked dinner for his family']
            perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']

            list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])

            attack_guidance = (
                f"You can finish the task by modifying sentence using the following guidance: {attack_type} "
                f"Here are five examples that fit the guidance: {list_examples}"
                f"Only output the new sentence without anything else."
                f"The new sentence is: [INSERT YOUR SENTENCE TRANSFORMATION HERE]"
            )
        prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
        
        return prompt

    def _generate_extractor_prompt(self, context_sentence, label_index):
        
        prompt = f'{self.start_prompt_header}' + f'The text in the brackets is what a langugage model has returned from a query [{context_sentence}] can you extract only the generated text and dont return anything else, if multiple answers are given return only 1! The text is:' + f'{self.end_prompt_footer}'
        
        return prompt
     

    def _remove_labels(self, text): 
        pattern = r'\b(' + '|'.join(map(re.escape, self.dataset.label_names)) + r')\b'
 
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
 
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text

    def _get_transformations(self, current_text, indices_to_modify):
        print (f'Original Sample: \n {current_text}' ) 
        self.ceattack_logger.debug(f'Original Sample: \n {current_text}')
                    
        
        
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = current_text.text
        transformations = []
        
        
        prompt = self._generate_prompt(context_sentence,expected_sentiment)
        generated_sentence = self._query_model(prompt)
        print (f'W1 Generated raw output: \n {generated_sentence}')
        self.ceattack_logger.debug(f'W1 Generated raw output: \n {generated_sentence}')

        extract_ans_prompt = self._generate_extractor_prompt(generated_sentence,expected_sentiment)
        extract_ans_prompt = self._remove_labels(extract_ans_prompt) 

        new_sentence = self._query_model(extract_ans_prompt)
        print (f'W1 attack New sentence: \n {new_sentence}' )
        self.ceattack_logger.debug(f'W1 attack New sentence: \n {new_sentence}')

        if (new_sentence) and (new_sentence != context_sentence):
            Att_sen_new_sentence = AttackedText(new_sentence)
            
            Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
            
            Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            
            # if the model returns a string with no words and only punctuation and nothering else e.g -> we return an empty list
            if len(Att_sen_new_sentence.attack_attrs['original_index_map']) == 0:
                return []
            
            transformations.append(Att_sen_new_sentence)
                
        return transformations 

