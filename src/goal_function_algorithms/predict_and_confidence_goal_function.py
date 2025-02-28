

from textattack.goal_functions import UntargetedClassification

import torch
import numpy as np

class Prediction_And_Confidence_GoalFunction(UntargetedClassification):
    def __init__(self,*args, target_max_score=None ,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.target_max_score = target_max_score
        self.current_sample_id = 0
        super().__init__(*args)
        

    def _is_goal_complete(self, model_output, _):
        # """
        # Check if the confidence of the true label is within the target range.
        # """
        
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):  
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            same_or_not =  model_output.argmax() != self.ground_truth_output
            self.ceattack_logger.debug(f'Is the model output prediction the same as the ground truth? model pred ({model_output.argmax()}), ground truth ({self.ground_truth_output}), answer ({same_or_not})')
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, _):
        
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            
            return 1 - model_output[self.ground_truth_output]

    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []
        


        inputs = [at.tokenizer_input for at in attacked_text_list]


        i = 0
        
        predictions = []
        outputs = []
        while i < len(inputs):
            logit_list = []
            datapoint = inputs[i]
            
            
            raw_output = self.predictor.add_prompt_and_call_model(datapoint) # do one function call because this function might require 1 or multiple calls to the model to determine confidence
            self.ceattack_logger.debug(f"Raw output: \n {raw_output}")
            print ('Raw output:',raw_output)
            standarized_output = self.predictor.standarize_output(raw_output)
            self.ceattack_logger.debug(f"Standarized output: \n {standarized_output}")
            print ('Standarized output:',standarized_output) 
            
            inference_step = i 
            
            standarized_output['inference_step'] = inference_step
            
            standarized_output['current_sample_id'] = self.current_sample_id
            guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities = self.predictor.aggregate_output(standarized_output)
            guess_result = self.predictor.prompt_class._predictor_decision()
            
            # here
            
            guess = guess_result
            probs = empirical_mean
            if guess == 'null':
                probs = torch.zeros(self.n_classes+1, dtype=torch.float16)
                probs[-1] = 1.0


            predictions.append(guess)
            logit_list.append(torch.tensor(probs, device=self.device)) 
            
            
            i += 1
            
            logit_tensor = torch.stack(logit_list)
            

            batch_preds = logit_tensor
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]


            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
                
        
        if isinstance(outputs[0], torch.Tensor):
            
            outputs = torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], np.ndarray):
            
            outputs = np.concatenate(outputs).ravel()
            
        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)
            
        
        
        
    def get_results(self, attacked_text_list, check_skip=False): 
        # this is what get_goal_results in search function calls indirectly so get_goal_results=get_results
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        self.ceattack_logger.debug(f"List of samples to attack: \n {attacked_text_list}")
        print ('List of samples to attack:',attacked_text_list) 

        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries 
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        self.ceattack_logger.debug(f"Final number of queries: {self.num_queries}")
        print ('Final number of queries:',self.num_queries)
        
        model_outputs = self._call_model(attacked_text_list)
        
        
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            
            goal_function_score = self._get_score(raw_output, attacked_text)
            
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget

    def init_attack_example(self, attacked_text, ground_truth_output):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = attacked_text
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        
        print ('initializing a new example')
        self.ceattack_logger.debug(f"Initializing a new example, Current sample under attack ID {self.current_sample_id}")
        self.current_sample_id +=1
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _
