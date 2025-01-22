
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
        # def __init__(self, num_transformations=20):
        #     self.num_transformations = num_transformations
        

    def perform_search(self, initial_result):
        print ('starting queries',self.goal_function.num_queries)
        self.number_of_queries = 0
        for i in range(self.num_transformations): # we ask the model N types of perturbations then


            # Get transformations using the custom transformation method 
            transformed_text_candidates = self.get_transformations(
            initial_result.attacked_text,
            original_text=initial_result.attacked_text,
            indices_to_modify=None,  # Modify the entire text
            )  
            self.number_of_queries +=1 # to get trasnformations we have to query 1 time the model
            # random.shuffle(transformed_text_candidates)
            # transformed_text_candidates = transformed_text_candidates[:min(self.num_transformations, len(transformed_text_candidates))]
            print ('transformed_text_candidates',transformed_text_candidates,len(transformed_text_candidates))
            valid_candidates = []
            for candidate in transformed_text_candidates:
                # similarity = self.use_constraint.similarity_function(
                #     initial_result.attacked_text.text, 
                #     candidate.text
                # )
                # print ('similarity',similarity)
                sim_score = self.use_constraint.get_sim_score(initial_result.attacked_text.text, candidate.text)
                # sim_final_original, sim_final_pert = self.use_constraint.encode([initial_result.attacked_text.text, candidate.text])

                # if not isinstance(sim_final_original, torch.Tensor):
                #     sim_final_original = torch.tensor(sim_final_original)

                # if not isinstance(sim_final_pert, torch.Tensor):
                #     sim_final_pert = torch.tensor(sim_final_pert)

                # sim_score2 = self.use_constraint.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                # print ('sim_score transform1',sim_score,round(sim_score, 4), (1 - (args.similarity_threshold) / math.pi),self.use_constraint.threshold)
                
                # print ('sim_score transform2',sim_score2,round(sim_score2, 4), (1 - (args.similarity_threshold) / math.pi),self.use_constraint.threshold)
                # sim_score = round(sim_score, 4)
                if sim_score >= self.use_constraint.threshold:# (1 - (args.similarity_threshold) / math.pi):
                    valid_candidates.append(candidate)

            # Now, valid_candidates only contains those candidates that meet the USE constraint
            print('valid_candidates', valid_candidates, len(valid_candidates))
            transformed_text_candidates = valid_candidates

            if not transformed_text_candidates:
                continue # try to get another transformation 

            # Evaluate each transformation
            results, search_over = self.get_goal_results(transformed_text_candidates)
            #get the number of classes, then do #classes+1 in results.predicted pop any that meets this value
            null_label = len(self.dataset.label_names)
            print ('results_before_null_filter',results)
            print ('null_label',null_label)
            

            results = [i for i in results if i.output != null_label] # filter out all attacks that lead to null
            print ('results_after_null_filter',results)
            # Return the first successful perturbation
            for result in results:
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    print ('successful adv sample ', self.goal_function.num_queries, self.number_of_queries )
                    self.goal_function.num_queries += self.number_of_queries # if adv sample found we query model N times to find a suitable transformation then N times to check it's actually adv
                    return result

        print ('ending queries',self.goal_function.num_queries,self.number_of_queries )
        self.goal_function.num_queries += self.number_of_queries # no succesful adv samples found, we still query model N times to generate transformations
        
        return initial_result

    # def get_transformations(self, current_text, original_text=None, indices_to_modify=None):
    #     """Generate N transformations using the transformation method."""
    #     print ('current_text2',current_text)
    #     if hasattr(self.transformation, "transform"):
    #         print ('list transforms',list(self.transformation.transform(current_text, self.num_transformations)))
    #         sys.exit()
    #         return list(self.transformation.transform(current_text, self.num_transformations))
    #     return []

    # def get_goal_results(self, transformed_text_candidates):
    #     print ('transformed_text_candidates',transformed_text_candidates)
    #     sys.exit()
    #     """Evaluate the goal on the transformed text candidates."""
    #     results = [self.goal_function.get_result(text) for text in transformed_text_candidates]
    #     search_over = any(result.goal_status == GoalFunctionResultStatus.SUCCEEDED for result in results)
    #     return results, search_over

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["num_transformations"]

    def __repr__(self):
        return default_class_repr(self)
