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
            
        self.beam_width = beam_width
        self.previeous_beam = [] 
        self.current_sample_id = 0
    

    def _get_index_order(self, initial_text, max_len=-1):
        
        """ How do we decide the order in which to substitute words?

            -Random: We select some words at random.
            -Delete: Popularized by TextFooler, we perform one parse of the entire sample and 
            check how the confidence changes if each word is removed from the input. 
            We then rank the words by the change in confidence."
        """
        if self.index_order_technique  == 'random':
            len_text, indices_to_order = self.get_indices_to_order(initial_text)
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
            return index_order, search_over
        
        elif self.index_order_technique == 'delete': 
            len_text, indices_to_order = self.get_indices_to_order(initial_text)

            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ] 
            leave_one_results, search_over = self.get_goal_results(leave_one_texts) 
            index_scores = np.array([result.score for result in leave_one_results]) 
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            return  index_order, search_over
    def perform_search(self, initial_result): 
        
        
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        i = 0
        cur_result = initial_result
        results = None  
        best_result = None
        max_similarity = -float("inf")
        

        while i < len(index_order) and not search_over:
            if i > self.max_iter_i:
                
                break
            
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            ) 
            print (f"All {len(transformed_text_candidates)} transformations {transformed_text_candidates}")
            self.ceattack_logger.debug(f"All {len(transformed_text_candidates)} transformations: \n {transformed_text_candidates}")
            
            i += 1
            valid_candidates = []
            for candidate in transformed_text_candidates:
                
                sim_score = self.use_constraint.get_sim_score(initial_result.attacked_text.text, candidate.text)
                
                if sim_score >= self.use_constraint.threshold: 
                    valid_candidates.append(candidate)

            # Now, valid_candidates only contains those candidates that meet the USE constraint
            print(f"All {len(valid_candidates)} Valid candidates (quality surpasses set similarity threshold) \n", valid_candidates)
            self.ceattack_logger.debug(f"All {len(valid_candidates)} Valid candidates (quality surpasses set similarity threshold): \n {valid_candidates}")
            
            transformed_text_candidates = valid_candidates

            if len(transformed_text_candidates) == 0:
                continue
            
            results, search_over = self.get_goal_results(transformed_text_candidates)
            null_label = len(self.dataset.label_names)
            
            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            
            
            results = sorted(results, key=lambda x: -x.score)
            print ('Sorted Results: \n',results)
            self.ceattack_logger.debug(f"Sorted Results: \n {results}")
            if len(results) == 0:
                continue

            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score: 
                cur_result = results[0] 
                 
            else:
                continue 
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                candidate = cur_result.attacked_text
                

                sim_final_original, sim_final_pert = self.use_constraint.encode([initial_result.attacked_text.text, cur_result.attacked_text.text])

                if not isinstance(sim_final_original, torch.Tensor):
                    sim_final_original = torch.tensor(sim_final_original)

                if not isinstance(sim_final_pert, torch.Tensor):
                    sim_final_pert = torch.tensor(sim_final_pert)

                sim_score = self.use_constraint.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                
                
                sim_score = round(sim_score, 4)
                similarity_score = sim_score 


                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_result = cur_result

        
        
        if best_result: 
            return best_result
        else:
            return cur_result
        
    
    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]
