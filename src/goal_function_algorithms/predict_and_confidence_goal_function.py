

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
        
    # def __init__(self, *args, target_max_score=None, **kwargs):
    #     self.target_max_score = target_max_score
    #     super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        # """
        # Check if the confidence of the true label is within the target range.
        # """
        
        if self.target_max_score:
            print ('1')
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):  
            print ('2')
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            print ('3', model_output.argmax(),self.ground_truth_output, model_output.argmax() != self.ground_truth_output)
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        # print ('true_label_confidence is get score',model_output[self.ground_truth_output],1 - model_output[self.ground_truth_output])
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            # print ('model output get score',model_output,self.ground_truth_output, 1 - model_output[self.ground_truth_output])

            # issue here, need to change so that it's not 1-, but instead is current prediction score
            # we are doing now [negative 0.7000, positive 0.2500, null 0.0500], 1-0.25 because ground truth point 1, but we want
            # confidence 0.7
            return 1 - model_output[self.ground_truth_output]

    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []
        

        # current_sample = attacked_text_list[0].current_sample (any sample from this will give you current sample)
        inputs = [at.tokenizer_input for at in attacked_text_list]

        # inputs = [inputs[0],inputs[0],inputs[0]]
        # print('inputs',inputs)

        # i = 0
        # while i < len(inputs):
        #     batch = inputs[i : i + self.batch_size]
        #     print ('here1')
        #     raw_output = self.predictor.add_prompt_and_call_model(batch) # do one function call because this function might require 1 or multiple calls to the model to determine confidence
        #     print ('raw_output',raw_output)
        #     standarized_output = self.predictor.standarize_output(raw_output)
        #     print ('standarized_output',standarized_output)
        #     guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities = self.predictor.aggregate_output(standarized_output)
        #     guess_result = self.predictor.prompt_class._predictor_decision()
            

        i = 0
        # logit_list = []
        predictions = []
        outputs = []
        while i < len(inputs):
            logit_list = []
            datapoint = inputs[i]
            # guess, probs, confidence = self.predictor.predict_and_confidence(datapoint)
            print ('here1') 
            
            raw_output = self.predictor.add_prompt_and_call_model(datapoint) # do one function call because this function might require 1 or multiple calls to the model to determine confidence
            print ('raw_output',raw_output)
            standarized_output = self.predictor.standarize_output(raw_output)
            print ('standarized_output',standarized_output) 
            # pass i to aggregate and current_sample_id
            inference_step = i 
            # print ('attacked_text_list[0]',attacked_text_list[0])
            # try:
            #     current_sample_id = attacked_text_list[0].current_sample_id
            #     print ('current_sample_id new',current_sample_id)
            # except Exception as e:
            #     current_sample_id = 0
            #     print ('exception',e)
            standarized_output['inference_step'] = inference_step
            print ('self.current_sample_id',self.current_sample_id)
            standarized_output['current_sample_id'] = self.current_sample_id
            guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities = self.predictor.aggregate_output(standarized_output)
            guess_result = self.predictor.prompt_class._predictor_decision()
            
            # here
            
            guess = guess_result
            probs = empirical_mean
            if guess == 'null': # return the logits with original label at 1
                probs = torch.zeros(self.n_classes+1, dtype=torch.float16)
                probs[-1] = 1.0

            print ('probs',probs)
            predictions.append(guess)
            logit_list.append(torch.tensor(probs, device=self.device)) 
            # logit_list.append(torch.tensor(probs, device=self.device)) 
            
            i += 1
            print ('logit_list',logit_list)
            logit_tensor = torch.stack(logit_list)
            print ('logit_tensor',logit_tensor)
         
        # probs: tensor([[0.1484, 0.8385, 0.0130],
                # [0.3207, 0.6645, 0.0148],
                # [0.1501, 0.8357, 0.0141],
                # [0.1493, 0.8371, 0.0136]], device='cuda:0', dtype=torch.float64)

            batch_preds = logit_tensor
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                # outputs.append(batch_preds)
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            print ('outputs',outputs)
            print ('outputs[0]',outputs[0],len(inputs),len(outputs))
        
        if isinstance(outputs[0], torch.Tensor):
            print ('what1')
            outputs = torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], np.ndarray):
            print ('what2')
            outputs = np.concatenate(outputs).ravel()
        print ('outputs',outputs, len(outputs))
        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)
            
        print ('logit list:',outputs)
        logit_tensor = torch.stack(outputs)
        print('logit_tensor:', logit_tensor)
        return logit_tensor

        # this is the raw input that needs to be processes and turned into a prompt
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + self.batch_size]
            # batch just sends a text to model (the __call__ from huggingface model wrapper)
            
            batch_preds = self.model(batch,self.ground_truth_output)
            # return a raw string
            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                # outputs.append(batch_preds)
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            i += self.batch_size

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], np.ndarray):
            outputs = np.concatenate(outputs).ravel()

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)
            
        
    # def _call_model_uncached(self, attacked_text_list):
    #     """Queries model and returns outputs for a list of AttackedText
    #     objects."""
    #     if not len(attacked_text_list):
    #         return []
        

 
    #     inputs = [at.tokenizer_input for at in attacked_text_list]

    #     i = 0
    #     # logit_list = []
    #     predictions = []
    #     outputs = []
    #     while i < len(inputs):
    #         logit_list = []
    #         datapoint = inputs[i]
    #         # guess, probs, confidence = self.predictor.predict_and_confidence(datapoint)
    #         print ('here1')
    #         raw_output = self.predictor.add_prompt_and_call_model(datapoint) # do one function call because this function might require 1 or multiple calls to the model to determine confidence
    #         print ('raw_output',raw_output)
    #         standarized_output = self.predictor.standarize_output(raw_output)
    #         print ('standarized_output',standarized_output) 
    #         guess_result_with_confidence, empirical_mean, second_order_uncertainty, probabilities = self.predictor.aggregate_output(standarized_output)
    #         guess_result = self.predictor.prompt_class._predictor_decision()
            
    #         # here
            
    #         guess = guess_result
    #         probs = empirical_mean
    #         if guess == 'null': # return the logits with original label at 1
    #             probs = torch.zeros(self.n_classes+1, dtype=torch.float16)
    #             probs[-1] = 1.0

    #         print ('probs',probs)
    #         predictions.append(guess)
    #         logit_list.append(torch.tensor(probs, device=self.device)) 
    #         # logit_list.append(torch.tensor(probs, device=self.device)) 
            
    #         i += 1
    #         print ('logit_list',logit_list)
    #         logit_tensor = torch.stack(logit_list)
    #         print ('logit_tensor',logit_tensor)
         
    #     # probs: tensor([[0.1484, 0.8385, 0.0130],
    #             # [0.3207, 0.6645, 0.0148],
    #             # [0.1501, 0.8357, 0.0141],
    #             # [0.1493, 0.8371, 0.0136]], device='cuda:0', dtype=torch.float64)

    #         batch_preds = logit_tensor
    #         if isinstance(batch_preds, str):
    #             batch_preds = [batch_preds]

    #         # Get PyTorch tensors off of other devices.
    #         if isinstance(batch_preds, torch.Tensor):
    #             batch_preds = batch_preds.cpu()

    #         if isinstance(batch_preds, list):
    #             outputs.extend(batch_preds)
    #         elif isinstance(batch_preds, np.ndarray):
    #             # outputs.append(batch_preds)
    #             outputs.append(torch.tensor(batch_preds))
    #         else:
    #             outputs.append(batch_preds)
    #         print ('outputs',outputs)
    #         print ('outputs[0]',outputs[0],len(inputs),len(outputs))
        
    #     if isinstance(outputs[0], torch.Tensor):
    #         print ('what1')
    #         outputs = torch.cat(outputs, dim=0)
    #     elif isinstance(outputs[0], np.ndarray):
    #         print ('what2')
    #         outputs = np.concatenate(outputs).ravel()
    #     print ('outputs',outputs, len(outputs))
    #     assert len(inputs) == len(
    #         outputs
    #     ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

    #     return self._process_model_outputs(attacked_text_list, outputs)
            
    #     print ('logit list:',outputs)
    #     logit_tensor = torch.stack(outputs)
    #     print('logit_tensor:', logit_tensor)
    #     return logit_tensor

    #     # this is the raw input that needs to be processes and turned into a prompt
    #     outputs = []
    #     i = 0
    #     while i < len(inputs):
    #         batch = inputs[i : i + self.batch_size]
    #         # batch just sends a text to model (the __call__ from huggingface model wrapper)
            
    #         batch_preds = self.model(batch,self.ground_truth_output)
    #         # return a raw string
    #         # Some seq-to-seq models will return a single string as a prediction
    #         # for a single-string list. Wrap these in a list.
    #         if isinstance(batch_preds, str):
    #             batch_preds = [batch_preds]

    #         # Get PyTorch tensors off of other devices.
    #         if isinstance(batch_preds, torch.Tensor):
    #             batch_preds = batch_preds.cpu()

    #         if isinstance(batch_preds, list):
    #             outputs.extend(batch_preds)
    #         elif isinstance(batch_preds, np.ndarray):
    #             # outputs.append(batch_preds)
    #             outputs.append(torch.tensor(batch_preds))
    #         else:
    #             outputs.append(batch_preds)
    #         i += self.batch_size

    #     if isinstance(outputs[0], torch.Tensor):
    #         outputs = torch.cat(outputs, dim=0)
    #     elif isinstance(outputs[0], np.ndarray):
    #         outputs = np.concatenate(outputs).ravel()

    #     assert len(inputs) == len(
    #         outputs
    #     ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

    #     return self._process_model_outputs(attacked_text_list, outputs)
        
    def get_results(self, attacked_text_list, check_skip=False): 
        # this is what get_goal_results in search function calls indirectly so get_goal_results=get_results
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        print ('attacked_text_list',attacked_text_list,check_skip)
        print ('self.num_queries',self.num_queries)
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries 
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        print ('final_num_queries',self.num_queries)
        # prepare the attacked_text_list so that the sample is put into template
        # 
        model_outputs = self._call_model(attacked_text_list)
        # model outputs will be raw string, this is where we can perform our cleaning
        # and aggregation

        # counter = 1
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            print ('get score')
            goal_function_score = self._get_score(raw_output, attacked_text)
            # attacked_text.inference_step+=counter
            # counter+=1
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
        self.current_sample_id +=1
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _
