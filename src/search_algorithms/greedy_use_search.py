from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
 
 

import numpy as np
import torch
import math
 
class GreedyUSESearch(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """
    def __init__(self,beam_width=1,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    # def __init__(self,index_order_technique,goal_function, beam_width=1):
    #     self.index_order_technique = index_order_technique
    #     self.goal_function = goal_function
        self.beam_width = beam_width
        self.previeous_beam = [] 
        self.current_sample_id = 0
    

    def _get_index_order(self, initial_text, max_len=-1):
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        elif self.index_order_technique  == 'prompt_top_k': 
            # ground_truth = 'positive'
            K = '' # number of important words to return 
            print ('self.ground_truth_output',self.goal_function.ground_truth_output) # should test with and without ground truth
            
            label_list = self.dataset.label_names
            label_index = self.goal_function.ground_truth_output
            expected_prediction, other_classes = self.predictor.prompt_class._identify_correct_incorrect_labels( label_index)
            
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('initial_text',initial_text)
            print ('len text indeces to order',len_text,indices_to_order)
            examples = ['The cat is on the table', 'The boy is playing soccer', 'She drove her car to work','The sun is shining brightly', 'He cooked dinner for his family']
            
            # prompt = f"""{self.start_prompt_header}return the most important words for the task of {task} where the text is '{initial_text.text}' and is classified as {ground_truth}.
            # Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            # Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            # The top {K} words are: {self.end_prompt_footer}"""

            prompt = f"""{self.start_prompt_header}return the most important words (in descending order of importance) for the following text '{initial_text.text}' which is classified as {expected_prediction}.
            Do not output anything else just the top words! separated as a comma, for example generated text: playing, soccer, boy
            Here are five examples that fit the task: 'The cat is on the table' -> cat, table | 'The boy is playing soccer' -> playing, soccer, boy | 'She drove her car to work'-> work, drove, car | 'The sun is shining brightly' -> brightly, shining, sun | 'He cooked dinner for his family' -> family, cooked, dinner
            The top {K} words in descending order are: {self.end_prompt_footer}"""
            # print ('prompt',prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        

            generate_args = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": 0.7,  # sampling temperature
            "max_new_tokens": 200,
            'pad_token_id': tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = model.generate(**generate_args)
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = outputs[0][prompt_length:]
            generated_text = tokenizer.decode(generated_tokens,skip_special_tokens=True)
            print("Generated Text word order:", generated_text)
            print ('words of original text',initial_text.words)
            words_list = [word.strip().lower() for word in generated_text.split(',')]
            initial_words_list = [word.lower() for word in initial_text.words]
            print ('word_list',words_list, set(initial_words_list), set(words_list)  )
            # find set of words that are in both, then for each word in words list that is in set find index in initial_text.words
            interesection_words = set(initial_words_list) & set(words_list) 
            print ('interesection_words',interesection_words)
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            print ('len_text, indices_to_order',len_text, indices_to_order)
            
            
            indices_to_order = []

            # for i,w in enumerate(initial_words_list):
            #     if w in interesection_words:
            #         indices_to_order.append(i)

            initial_word_index_pair = {j:i for i,j in enumerate(initial_words_list) }
            print ('initial_word_index_pair',initial_word_index_pair)
            for i,w in enumerate(words_list):
                if w in interesection_words:
                    indices_to_order.append(initial_word_index_pair[w])# initial_words_list.index(w)) # potntially use miaos ranking explenation for theory behond this?


            print('indices_to_order',indices_to_order) 

            # len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            search_over = False 

            if len(index_order) == 0:
                len_text, indices_to_order = self.get_indices_to_order(initial_text)
                index_order = indices_to_order
                np.random.shuffle(index_order)
                search_over = False

            search_over = False
            return index_order, search_over
        elif self.index_order_technique == 'delete': 
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            print ('leave_one_texts',leave_one_texts)
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            print ('leave_one_results, search_over',leave_one_results, search_over)

            index_scores = np.array([result.score for result in leave_one_results])
            print ('index_scores',index_scores, 'search_over',search_over)
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            return  index_order, search_over
    def perform_search(self, initial_result): 
        # self.current_sample_id+=1
        # print ('initial_result',initial_result)
        # print ('initial_result.attacked_text',initial_result.attacked_text)
        # initial_result.attacked_text.current_sample_id = self.current_sample_id
        # print ('initial_result.attacked_text.current_sample_id',initial_result.attacked_text.current_sample_id)
        
        
        
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None  
        best_result = None
        max_similarity = -float("inf")
        # pick two indexes and modify them
        while i < len(index_order) and not search_over:
            if i > self.max_iter_i:
                print ('reached max i',i)
                break
            # print ('cur_result.attacked_text',cur_result.attacked_text.current_sample_id)
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            ) 
            print ('transformed_text_candidates',i,transformed_text_candidates,len(transformed_text_candidates))
            i += 1
            # apply filtering operation
            # for i,tranform in enumerate(transformed_text_candidates):
            #     sim_final_original, sim_final_pert = self.use_constraint.encode([initial_result.attacked_text.text, tranform.text])

            #     if not isinstance(sim_final_original, torch.Tensor):
            #         sim_final_original = torch.tensor(sim_final_original)

            #     if not isinstance(sim_final_pert, torch.Tensor):
            #         sim_final_pert = torch.tensor(sim_final_pert)

            #     sim_score = self.use_constraint.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
            #     print ('sim_score transform',sim_score, (1 - (args.similarity_threshold) / math.pi))
            #     if sim_score <  (1 - (args.similarity_threshold) / math.pi):
            #         continue

            valid_candidates = []
            for candidate in transformed_text_candidates:
                # similarity = self.use_constraint.similarity_function(
                #     initial_result.attacked_text.text, 
                #     candidate.text
                # )
                # print ('similarity',similarity)
                print ('waiting here')
                
                sim_score = self.use_constraint.get_sim_score(initial_result.attacked_text.text, candidate.text)
                print ('simsocre',sim_score)
                # import time
                # time.sleep(1000)
                
                if sim_score >= self.use_constraint.threshold:# (1 - (args.similarity_threshold) / math.pi):
                    valid_candidates.append(candidate)

            # Now, valid_candidates only contains those candidates that meet the USE constraint
            print('valid_candidates', valid_candidates, len(valid_candidates))
            transformed_text_candidates = valid_candidates

            if len(transformed_text_candidates) == 0:
                continue
            
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter')
            print ('null_label',null_label)
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            # print ('results_after_null_filter',results)

            # a self.track_result_score
            # can put all scores here so that we can access them later
            # for each i we can save a list of scores, then do the max in each, we expect for each i to increase
            # we can then show how this increases by number of perturbations by definition a low and high score
            # are high confidences, while a mid score are medium confidences

            results = sorted(results, key=lambda x: -x.score)
            print ('results_after_sorted',results)
            if len(results) == 0:
                continue
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score: 
                cur_result = results[0] 
                 
            else:
                continue 
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                candidate = cur_result.attacked_text
                # try:
                #     similarity_score = candidate.attack_attrs["similarity_score"]
                # except KeyError:
                #     break

                sim_final_original, sim_final_pert = self.use_constraint.encode([initial_result.attacked_text.text, cur_result.attacked_text.text])

                if not isinstance(sim_final_original, torch.Tensor):
                    sim_final_original = torch.tensor(sim_final_original)

                if not isinstance(sim_final_pert, torch.Tensor):
                    sim_final_pert = torch.tensor(sim_final_pert)

                sim_score = self.use_constraint.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                print ('sim_score success',sim_score, (1 - (self.similarity_threshold) / math.pi))
                
                
                # if sim_score <  (1 - (args.similarity_threshold) / math.pi):
                #     continue
                sim_score = round(sim_score, 4)
                similarity_score = sim_score 


                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_result = cur_result

        
        # self.goal_function.model.reset_inference_steps()
        if best_result: 
            return best_result
        else:
            return cur_result
        # else:
        #     sim_final_original, sim_final_pert = self.use_constraint.encode([initial_result.attacked_text.text, cur_result.attacked_text.text])

        #     if not isinstance(sim_final_original, torch.Tensor):
        #         sim_final_original = torch.tensor(sim_final_original)

        #     if not isinstance(sim_final_pert, torch.Tensor):
        #         sim_final_pert = torch.tensor(sim_final_pert)

        #     sim_score = self.use_constraint.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
        #     print ('sim_score end',sim_score, (1 - (args.similarity_threshold) / math.pi))
        #     # print ('prev similarity_score',similarity_score)
        #     if sim_score <  (1 - (args.similarity_threshold) / math.pi):
        #         return []
        #     else:
        #         return cur_result 

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]
