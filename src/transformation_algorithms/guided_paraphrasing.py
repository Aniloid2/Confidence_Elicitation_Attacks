from textattack.transformations import WordSwap
from textattack.shared import AttackedText

import torch
import re



class GuidedParaphrasing(WordSwap):
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

 

    def _query_model(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 1,  # nucleus sampling probability
            "temperature": 1,  # sampling temperature
            "max_new_tokens": 2000,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        # Generate the output with the model
        with torch.no_grad():
            outputs = self.model.generate(**generate_args)


        prompt_length = len(inputs['input_ids'][0])
        generated_tokens = outputs[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

         
        print ('generated_text:',generated_text)
 
        match_generated_text = None
        if self.task == 'sst2':
            pattern = r"(?<=:)(.+)"  
            match_generated_text = re.search(pattern, generated_text)
        elif self.task == 'ag_news': 
            substrings_to_remove = ['business', 'world', 'tech/sci','sci/tech', 'tech', 'science', 'sport']
            

            pattern = r'\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b'

            # Remove the matched keywords and their surrounding punctuation, ensuring spacing is maintained.
            def replace(match):
                preceding = generated_text[max(0, match.start()-1)]
                following = generated_text[min(len(generated_text), match.end()):min(len(generated_text), match.end() + 1)]

                # Check for spaces to avoid having multiple spaces
                need_space = (preceding not in ' \t\n\r') and (following not in ' \t\n\r')

                if need_space:
                    return ' '
                else:
                    return ''

            clean_text = re.sub(r'[\[\]{}(),]*\b(?:' + '|'.join(map(re.escape, substrings_to_remove)) + r')\b[\[\]{}(),]*', replace, generated_text,flags=re.IGNORECASE)

            # Clean up remaining extra spaces left by the removals
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            generated_text = clean_text 

            pattern = r"(?<=:)(.+)"  
            match_generated_text = re.search(pattern, generated_text)
        print ('generated_text_after_cleanup:',generated_text)
        print ('match_generated_text',match_generated_text)
        if match_generated_text:
            print ('match_generated_text.group(1).strip()',match_generated_text.group(1).strip())
            return match_generated_text.group(1).strip()
        return generated_text.strip()

 

    def _generate_prompt(self, context_sentence, expected_sentiment):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2':
            # text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            # label_types = ['negative', 'positive']
            # filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            # other_classes = ' or '.join(filtered_label_types)
            # attack_type = 'Rewrite the sentance.'
            # original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment} \n. "
            # attack_objective = (
            #     f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
            #     f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
            #     f"2) The new {text_type} should be classified as {other_classes}.\n"
            #     )
            # if self.prompt_shot_type == 'zs':
            #     attack_guidance = (
            #         f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
            #         f"Only output the new {text_type} without anything else."
            #         f"The new sentece is:"
            #     )
            # elif  self.prompt_shot_type == 'fs':
            #     # original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            #     # perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
            #     original_example = ['This is a great day', 'never again will I play this game', 'She drove a wonderfull car to work']
            #     perturbed_example = ['The day was wonderful', 'last time that this game will be played is today', 'She guided her good vehicle to work']
                
            #     list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
            #     attack_guidance = (
            #         f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
            #         f"Here are five examples that fit the guidance: {list_examples}.\n"
            #         f"Only output the new {text_type} without anything else.\n"
            #         f"The new sentece is:"
            #     )
                

            text_type = 'sentence'
            expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            label_types = ['negative', 'positive']
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Rewrite the sentence.'

            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}.\n"
            examples = [
                ("It's a wonderful and uplifting journey.", "It's a decent and somewhat inspiring journey."),
                ("The movie is a delightful experience.", "The movie is a fairly enjoyable experience.")
            ]

            example_prompts = ""
            for pos, neg in examples:
                example_prompts += f"- Original: '{pos}'\n  New: '{neg}'\n  Explanation: The semantic meaning remains the same, but the emotional tone has been subtly shifted to be more neutral.\n\n"

            attack_objective = (f"Your task is to generate a new {text_type}")
            
            #  (
            #     f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
            #     # f"1) create a large semantic meaning change of the new {text_type}, however, the meaning the information and message should be the same.\n"
            #     # f"2) The new {text_type} should be classified as {other_classes}.\n"
            #     # f"3) Make subtle changes to adjust the sentiment while keeping the overall intent and meaning unaltered.\n" 
            #     # f"Here are some examples:\n{example_prompts}"
            #     # f"Think through the following steps to complete your task:\n"
            #     # f"Step 1: Rephrase or paraphrase the entire original sentance.\n"
            #     # f"Step 2: Ensure this change has a large semantic change.\n"
            #     # f"Step 3: Determine words or phrases that convey the positive sentiment in the new paraphrased sentance and list them.\n"
            #     # f"Step 4: Reword this new pharaprased sentance with counterfitted word substitutions\n"
            #     # f"Step 5: Ensure the new {text_type} maintains the original meaning.\n"
            # )

            # attack_guidance = (
            #     f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
            #     f"Only output the new {text_type} without anything else.\n"
            #     f"The new sentence is:"
            # )

            attack_guidance = (
                f"You can finish the task by modifying {text_type} by rewriting the sentance, be creative with your response. "
                f"Only output the new {text_type} without anything else.\n"
                f"The new sentence is:"
            )
            prompt =f'{self.start_prompt_header}' + original_input + attack_objective  + attack_guidance + f'{self.end_prompt_footer}'
             
            
            
            print ('prompt',prompt)
            
        elif self.task == 'ag_news':
            text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
            expected_sentiment_label = self.goal_function.ground_truth_output
            
            print ('expected sentiment',expected_sentiment) 
            label_types= self.dataset.label_names
            expected_sentiment = label_types[expected_sentiment_label]
            print ('label_types',label_types)
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Replace at most two words in the sentence with synonyms.'
            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
            attack_objective = (
                f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
                f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
                f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
                # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
            )
            if self.prompt_shot_type == 'zs':
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            elif  self.prompt_shot_type == 'fs':
                original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
                perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
                
                list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Here are five examples that fit the guidance: {list_examples} "
                    f"Only output the new {text_type} without anything else. Don't provide reasoning "
                    f"The new sentece is: "
                )
            # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
            #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
            # system_prompt = ''
            # <<SYS>>{system_prompt}<</SYS>>
            prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
  
        return prompt

    def _get_transformations(self, current_text, indices_to_modify):
        print ('current_text',current_text )
        # print ('current_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = current_text.text
        transformations = []
        for i in range(self.num_transformations):
            # prompt = self._generate_prompt(context_sentence,expected_sentiment)
            
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences',new_sentences) 

            # if new_sentence and new_sentence != context_sentence:
            #     Att_sen_new_sentence = AttackedText(new_sentence)
            #     print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
            #     Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            #     Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            #     print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
            #     transformations.append(Att_sen_new_sentence)

            prompt = self._generate_prompt(context_sentence,expected_sentiment)
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences_explore',new_sentences)
            # sys.exit()
            new_sentence = self._query_model(prompt) 
            print ('new_sentences_explore',new_sentence) 

            if new_sentence and new_sentence != context_sentence:
                Att_sen_new_sentence = AttackedText(new_sentence)
                print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
                Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
                Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
                print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
                transformations.append(Att_sen_new_sentence)

        print ('transformations',transformations)

        return transformations  
    
    def _generate_prompt_maximise_semantic_sim(self, context_sentence, best_sentence):
        if self.task not in ['sst2', 'ag_news']:
            raise ValueError("Unsupported task. Please choose either 'sst2' or 'ag_news'.")

        if self.task == 'sst2': 

            text_type = 'sentence'
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            # label_types = ['negative', 'positive']
            # filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            # other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Rewrite the sentence.'

            # original_input = f"The original {text_type} '{best_sentence}' is very different to the original {context_sentence}.\n"
            original_input = f"The original {text_type} '{best_sentence}'"
            # attack_objective = (f"Your task is to increase the semantic similarity between '{best_sentence}' and '{context_sentence} \n'")
            attack_objective = (f"Your task is to rephrase the following sentence: '{context_sentence}' to make it slightly similar to '{best_sentence}', don't make many changes!")
             

            attack_guidance = (
                # f"You can finish the task by modifying {text_type} by rewriting the sentance, be creative with your response. "
                f"Only output the new {text_type} without anything else.\n"
                f"The new sentence is:"
            )
            prompt =f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
             
            
            
            print ('prompt',prompt)
            
        elif self.task == 'ag_news':
            text_type = 'sentence' 
            # expected_sentiment = 'positive' if expected_sentiment == 1 else 0
            # expected_sentiment_label = 1 if expected_sentiment == 'positive' else 0 
            
            expected_sentiment_label = self.goal_function.ground_truth_output
            
            print ('expected sentiment',expected_sentiment) 
            label_types= self.dataset.label_names
            expected_sentiment = label_types[expected_sentiment_label]
            print ('label_types',label_types)
            filtered_label_types = [label for i, label in enumerate(label_types) if i != expected_sentiment_label]
            other_classes = ' or '.join(filtered_label_types)
            attack_type = 'Replace at most two words in the sentence with synonyms.'
            original_input = f"The original {text_type} '{context_sentence}' is classified as {expected_sentiment}. "
            attack_objective = (
                f"Your task is to generate a new {text_type} which must satisfy the following conditions:\n"
                f"1) Keeping the semantic meaning of the new {text_type} unchanged.\n"
                f"2) The new {text_type} and the original {text_type} should be classified as {other_classes}.\n"
                # f"3) In your answer, don't generate any of the following tokens: {label_types}\n "
            )
            if self.prompt_shot_type == 'zs':
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Only output the new {text_type} without anything else."
                    f"The new sentece is:"
                )
            elif  self.prompt_shot_type == 'fs':
                original_example = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
                perturbed_example = ['The feline is on the desk', 'The lad is playing football', 'She guided her vehicle to work', 'The sol is shining vividly', 'He prepared supper for his family']
                
                list_examples = ' , '.join([original_example[i] + '->' + perturbed_example[i] for i in range(len(original_example)) ])
                 
                attack_guidance = (
                    f"You can finish the task by modifying {text_type} using the following guidance: {attack_type} "
                    f"Here are five examples that fit the guidance: {list_examples} "
                    f"Only output the new {text_type} without anything else. Don't provide reasoning "
                    f"The new sentece is: "
                )
            # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                
            #                     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """
            # system_prompt = ''
            # <<SYS>>{system_prompt}<</SYS>>
            prompt = f'{self.start_prompt_header}' + original_input + attack_objective + attack_guidance + f'{self.end_prompt_footer}'
  
        return prompt

    def _maximise_semantic_sim(self, original_attacked_text, best_attacked_text ):
        print ('current_text',original_attacked_text )
        # print ('original_attacked_text.attack_attrs',self.ground_truth_output)
        print ('self.goal_function',self.goal_function )
        print ('self gto', self.goal_function.ground_truth_output)
        expected_sentiment =  self.goal_function.ground_truth_output
        context_sentence = original_attacked_text.text
        best_sentence = best_attacked_text.text
        transformations = []
        for i in range(self.num_transformations):
            # prompt = self._generate_prompt(context_sentence,expected_sentiment)
            
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences',new_sentences) 

            # if new_sentence and new_sentence != context_sentence:
            #     Att_sen_new_sentence = AttackedText(new_sentence)
            #     print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
            #     Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
            #     Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
            #     print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
            #     transformations.append(Att_sen_new_sentence)

            prompt = self._generate_prompt_maximise_semantic_sim(context_sentence,best_sentence)

            print ('prompt increase semantic sim',prompt)
            # new_sentences = [self._query_model(prompt) for i in range(20)]
            # print ('new_sentences_explore',new_sentences)
            # sys.exit()
            new_sentence = self._query_model(prompt) 
            print ('new_sentences_explore',new_sentence) 

            if new_sentence and new_sentence != context_sentence:
                Att_sen_new_sentence = AttackedText(new_sentence)
                print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs) 
                Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
                Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
                print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
                transformations.append(Att_sen_new_sentence)

        print ('transformations',transformations)
        return transformations

