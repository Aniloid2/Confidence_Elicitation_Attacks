
from textattack.transformations import WordSwap
from textattack.shared import AttackedText

import torch
import re


class SelfWordSubstitutionW1(WordSwap):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # def __init__(self, tokenizer, model ,task,prompt_shot_type,num_transformations,goal_function):
    #     self.tokenizer = tokenizer
    #     self.model = model
    #     self.task = task
    #     self.goal_function = goal_function
    #     self.num_transformations = num_transformations
    #     self.prompt_shot_type = prompt_shot_type
    #     self.device = next(self.model.parameters()).device
        
    def _query_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        # Generate the output with the model
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)


        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print ('generated_text:',generated_text) 
        return generated_text.strip()

    # def _query_model(self, prompt):
    #     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
    #     generate_args = {
    #         "input_ids": inputs['input_ids'],
    #         "attention_mask": inputs['attention_mask'],
    #         "do_sample": True,  # enable sampling
    #         "top_k": 40,  # top-k sampling
    #         "top_p": 0.92,  # nucleus sampling probability
    #         "temperature": 0.7,  # sampling temperature
    #         "max_new_tokens": 200,
    #         'pad_token_id': self.tokenizer.eos_token_id
    #     }

    #     # Generate the output with the model
    #     with torch.no_grad():
    #         outputs = self.model.generate(**generate_args)


    #     prompt_length = len(inputs['input_ids'][0])
    #     generated_tokens = outputs[0][prompt_length:]
    #     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    #     # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print ('generated_text:',generated_text)
    #     # generated_text = ' this is a random sentence BUSINESS .Business (business) and ;sport; and business and businesss and lbusinessk and kbusiness'
    #     # Use regex to extract the modified text
    #     # match_pattern = "(?:Here's a possible new sentence:|Based on the given conditions, here's the new sentence:|Here's my attempt:|Here's a suggestion:|Here's a new sentence that fits your requirements:|Here is the new sentence:|One possible solution:|A possible aswer:|One possible new sentence:|New sentence:|Here is the solution:|A possible solution could be:|A possible solution for the given task could be:|One possible solution:|Here's a possible solution:)(?:[.,;?!])?"
    #     # match_generated_text = re.search(r"New sentence: (.+)", generated_text)
    #     # match_generated_text = re.search(rf"{match_pattern} (.+)", generated_text)
    #     match_generated_text = None
    #     if self.task == 'sst2' or self.task == 'strategyQA' :
    #         pattern = r"(?<=:)(.+)"  
    #         match_generated_text = re.search(pattern, generated_text)
    #     elif self.task == 'ag_news': # label leaking filtering
    #         substrings_to_remove = ['business', 'world', 'tech/sci','sci/tech', 'tech', 'science', 'sport', 'sports']
            

    #         pattern = r'\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b'

    #         # Remove the matched keywords and their surrounding punctuation, ensuring spacing is maintained.
    #         def replace(match):
    #             preceding = generated_text[max(0, match.start()-1)]
    #             following = generated_text[min(len(generated_text), match.end()):min(len(generated_text), match.end() + 1)]

    #             # Check for spaces to avoid having multiple spaces
    #             need_space = (preceding not in ' \t\n\r') and (following not in ' \t\n\r')

    #             if need_space:
    #                 return ' '
    #             else:
    #                 return ''

    #         clean_text = re.sub(r'[\[\]{}(),]*\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b[\[\]{}(),]*', replace, generated_text,flags=re.IGNORECASE)

    #         # Clean up remaining extra spaces left by the removals
    #         clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    #         generated_text = clean_text 

    #         pattern = r"(?<=:)(.+)"  
    #         match_generated_text = re.search(pattern, generated_text)
    #     print ('generated_text_after_cleanup:',generated_text)
    #     print ('match_generated_text',match_generated_text)
    #     if match_generated_text:
    #         print ('match_generated_text.group(1).strip()',match_generated_text.group(1).strip())
    #         return match_generated_text.group(1).strip()
    #     return generated_text.strip()

    # def _generate_prompt(self, context_sentence, expected_sentiment):
    #     if self.task not in ['sst2', 'ag_news']:
    #         raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

    #     if self.task == 'sst2':
    #         text_type = 'sentence' 
    #         expected_sentiment = 'positive' if expected_sentiment == 1 else 0
    #         expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
    #         label_types = ['negative', 'positive']
    #         filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
    #         other_classes = ' or '.join(filtered_label_types)
    #         attack_type = 'Add at most two semantically neutral words to the sentence..'
    #         original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
    #         attack_objective = (
    #             f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
    #             f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
    #             f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}. "
    #         )
    #         if self.prompt_shot_type == 'zs':
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
    #                 f"Only output the new {text_type} without anything else."
    #                 f"The new sentece is:"
    #             )
    #         elif  self.prompt_shot_type == 'fs':
    #             original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
    #             perturbed_example = ['The feline cat is on the table desk', 'The the boy is is playing soccer', 'She drove her car to work i think', 'The round sun is shining brightly mmmh', 'He cooked a large dinner for his family']
                
    #             list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
    #                 f"Here are five examples that fit the guidance: {list_examples}"
    #                 f"Only output the new {text_type} without anything else."
    #                 f"The new sentece is:"
    #             )
    #         # prompt = original_input + attack_objective + attack_guidance
    #         prompt =f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
    #         # print ('prompt',prompt)
            
    #     elif self.task == 'ag_news':
    #         text_type = 'sentence' 
    #         # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
    #         # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
    #         expected_sentiment_label = self.goal_function.ground_truth_output
            
    #         print ('expected sentiment',expected_sentiment) 
    #         label_types= self.dataset.label_names
    #         expected_sentiment = label_types[expected_sentiment_label]
    #         print ('label_types',label_types)
    #         filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
    #         other_classes = ' or '.join(filtered_label_types)
    #         attack_type = 'Add at most two semantically neutral words to the sentence.'
    #         original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
    #         attack_objective = (
    #             f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
    #             f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
    #             f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
    #             # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
    #         )
    #         if self.prompt_shot_type == 'zs':
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
    #                 f"Only output the new {text_type} without anything else."
    #                 f"The new sentece is:"
    #             )
    #         elif  self.prompt_shot_type == 'fs':
    #             original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
    #             perturbed_example = ['The cat cat is on the table table', 'The the boy is is playing soccer', 'She drove her car to work i think', 'The round sun is shining brightly mmmh', 'He cooked a large dinner for his family']
                
    #             list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
    #             attack_guidance = (
    #                 f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} \n"
    #                 f"Here are five examples that fit the guidance: {list_examples} \n"
    #                 f"Only output the new {text_type} without anything else. Don't provide reasoning "
    #                 f"The new sentece is: "
    #             )
    #         # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
    #         #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
    #         # system_prompt = ''
    #         # <<SYS>>{system_prompt}<</SYS>>
    #         prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
    #         # prompt =  original_input + attack_objective + attack_guidance 

    #         # print ('attack_prompt:', prompt)

    #         # prompt = (
    #         #     f"[INST]Provide your best guess and the verbal confidence that it is correct ('lowest', 'low', 'medium',"
    #         #     f"'high', 'highest') for the following news article. Give ONLY the guess and verbal confidence, no other "
    #         #     f"words or explanation. For example:\nGuess: <most likely guess, either world, sports, business or tech/"
    #         #     f"science; not a complete sentence, just the guess!>\n Confidence: <the confidence, either 'lowest', 'low', "
    #         #     f"'medium','high', 'highest' that your guess is correct, without any extra commentary whatsoever; just the "
    #         #     f"confidence!>\n\nThe text is: {context_sentence} News type & confidence:[/INST]"
    #         # )
    #     print ('prompt',prompt)
    #     sys.exit()
    #     return prompt

    def _generate_prompt(self, context_sentence, label_index):
        # if self.task not in ['sst2', 'ag_news', 'strategyQA']:
        #     raise ValueError("Unsupported task. Please choose either 'sst2', 'ag_news', or 'strategyQA'.")

        label_list =  self.dataset.label_names 
        print ('label_index',label_index)
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
        # if self.task not in ['sst2', 'ag_news', 'strategyQA']:
        #     raise ValueError("Unsupported task. Please choose either 'sst2', 'ag_news', or 'strategyQA'.")

        
        prompt = f'{self.start_prompt_header}' + f'The text in the brackets is what a langugage model has returned from a query [{context_sentence}] can you extract only the generated text and dont return anything else, if multiple answers are given return only 1! The text is:' + f'{self.end_prompt_footer}'
        
        return prompt
    # def _get_transformations(self, current_text, indices_to_modify):
    #     print ('Current_Text',current_text.attack_attrs )
    #     # original_index_map = current_text.attack_attrs['original_index_map']
    #     # print ('current_text.attack_attrs',self.ground_truth_output)
    #     print ('self.goal_function',self.goal_function )
    #     print ('self gto', self.goal_function.ground_truth_output)
    #     expected_sentiment =  self.goal_function.ground_truth_output
    #     context_sentence = current_text.text
    #     transformations = []
    #     for i in range(self.num_transformations):
    #         prompt = self._generate_prompt(context_sentence,expected_sentiment)
    #         new_sentence = self._query_model(prompt)

    #         if (new_sentence) and (new_sentence != context_sentence):
    #             Att_sen_new_sentence = AttackedText(new_sentence)
    #             # print ('words',Att_sen_new_sentence)
    #             # current_text.generate_new_attacked_text(Att_sen_new_sentence.words)
    #             print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
    #             print ('att sen new words',Att_sen_new_sentence.words, len(Att_sen_new_sentence.words) ) 
    #             Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
    #             Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
    #             # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(range(len(Att_sen_new_sentence.words)))
    #             # Att_sen_new_sentence.attack_attrs['original_index_map'] = original_index_map
    #             Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
    #             print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)

    #             if len(Att_sen_new_sentence.attack_attrs['original_index_map']) == 0:
    #                 continue
                
                
                
    #             transformations.append(Att_sen_new_sentence)
                
    #     return transformations 

    def _remove_labels(self, text):
        # Create a regex pattern with ignoring case and match as whole word \b
        pattern = r'\b(' + '|'.join(map(re.escape, self.dataset.label_names)) + r')\b'

        # Substitute the matched word with an empty string
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Removing extra spaces if any
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text

    def _get_transformations(self, current_text, indices_to_modify):
        print ('Current_Text',current_text.attack_attrs )
        # original_index_map = current_text.attack_attrs['original_index_map']
        # print ('current_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = current_text.text
        transformations = []
        
        
        prompt = self._generate_prompt(context_sentence,expected_sentiment)
        generated_sentence = self._query_model(prompt)
        print ('Generated_sentence:',generated_sentence)
        extract_ans_prompt = self._generate_extractor_prompt(generated_sentence,expected_sentiment)
        extract_ans_prompt = self._remove_labels(extract_ans_prompt) 

        new_sentence = self._query_model(extract_ans_prompt)
        print ('New_sentence:', new_sentence)
        if (new_sentence) and (new_sentence != context_sentence):
            Att_sen_new_sentence = AttackedText(new_sentence)
            # print ('words',Att_sen_new_sentence)
            # current_text.generate_new_attacked_text(Att_sen_new_sentence.words)
            print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
            print ('att sen new words',Att_sen_new_sentence.words, len(Att_sen_new_sentence.words) ) 
            Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
            # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(range(len(Att_sen_new_sentence.words)))
            # Att_sen_new_sentence.attack_attrs['original_index_map'] = original_index_map
            Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)

            # if the model returns a string with no words and only punctuation and nothering else e.g -> we return an empty list
            if len(Att_sen_new_sentence.attack_attrs['original_index_map']) == 0:
                return []
            
            
            
            transformations.append(Att_sen_new_sentence)
                
        return transformations 

