
import numpy as np
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod

class BlackBoxSearch(SearchMethod):
    """A black-box search that queries the model only to get transformations
    and evaluates each transformation to determine if it meets the goal.

    Args:
        num_transformations (int): The number of transformations to generate for each query.
    """
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        

    def perform_search(self, initial_result):
        print ('starting queries',self.goal_function.num_queries)
        self.number_of_queries = 0
        for i in range(self.num_transformations): 
            


            transformed_text_candidates = self.get_transformations(
            initial_result.attacked_text,
            original_text=initial_result.attacked_text,
            indices_to_modify=None, 
            )  
            self.number_of_queries +=1 # to get trasnformations we have to query 1 time the model
            
            print (f"All {len(transformed_text_candidates)} transformations {transformed_text_candidates}")
            self.ceattack_logger.debug(f"All {len(transformed_text_candidates)} transformations: \n {transformed_text_candidates}")
            valid_candidates = []
            for candidate in transformed_text_candidates:
                
                sim_score = self.use_constraint.get_sim_score(initial_result.attacked_text.text, candidate.text)
                
                
                if sim_score >= self.use_constraint.threshold:
                    valid_candidates.append(candidate)

            # Now, valid_candidates only contains those candidates that meet the USE constraint
            print(f"All {len(valid_candidates)} Valid candidates (quality surpasses set similarity threshold) \n", valid_candidates)
            self.ceattack_logger.debug(f"All {len(valid_candidates)} Valid candidates (quality surpasses set similarity threshold): \n {valid_candidates}")
            transformed_text_candidates = valid_candidates

            if not transformed_text_candidates:
                continue # try to get another transformation 

            
            results, search_over = self.get_goal_results(transformed_text_candidates)
            
            null_label = len(self.dataset.label_names)
            
            

            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            print ('Sorted Results: \n',results)
            self.ceattack_logger.debug(f"Sorted Results: \n {results}")

            # Return the first successful perturbation
            for result in results:
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    self.goal_function.num_queries += self.number_of_queries # if adv sample found we query model N times to find a suitable transformation then N times to check it's actually adv
                    return result

        
        self.goal_function.num_queries += self.number_of_queries # no succesful adv samples found, we still query model N times to generate transformations
        
        return initial_result

   
   
    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["num_transformations"]

    def __repr__(self):
        return default_class_repr(self)
