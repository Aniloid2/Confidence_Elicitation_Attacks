
from textattack.models.wrappers import ModelWrapper

class HuggingFaceLLMWrapper(ModelWrapper):
    """A wrapper around HuggingFace for LLMs.

    Args:
        model: A HuggingFace pretrained LLM
        tokenizer: A HuggingFace pretrained tokenizer
    """

    def __init__(self, **kwargs):
 
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.generated_max_length = 2
        # self.inference_step = 0
        # self.current_sample = 1
        self.general_generate_args = {
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": self.temperature,  # sampling temperature
            "max_new_tokens": 200,
            # 'pad_token_id': self.tokenizer.eos_token_id
        }

        self.general_tokenizer_encoding_args = {
            'return_tensors':"pt",
            }
        



    # def reset_inference_steps(self):
    #     """Resets the inference step counter to 0 and current sample to +1"""
    #     self.inference_step = 0
    #     self.current_sample +=1


    def __call__(self, text_input_list):#,ground_truth_output):
        
        self.device = next(self.model.parameters()).device
        
        print ('text_input_list',text_input_list)
        print ('self.device',self.device)
        # text_input_list = [text_input_list,text_input_list]
        inference_tokenizer_args = self.general_tokenizer_encoding_args
        inference_tokenizer_args['text'] = text_input_list

        inputs = self.tokenizer(**inference_tokenizer_args)
        # inputs = self.tokenizer(text_input_list, return_tensors="pt")
        
        input_ids = inputs['input_ids'].to(self.device) 
        att_ids = inputs['attention_mask'].to(self.device) 
        # print ('input_ids',input_ids)

        inference_generate_args = {
            "input_ids": input_ids,
            "attention_mask": att_ids, 
            "do_sample": self.general_generate_args['do_sample'],  # enable sampling
            "top_k": self.general_generate_args['top_k'],# 40,  # top-k sampling
            "top_p": self.general_generate_args['top_p'],# 0.92,  # nucleus sampling probability
            "temperature": self.temperature,  # sampling temperature
            "max_new_tokens": self.general_generate_args['max_new_tokens'],#200,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        extra_args = {
            "prompt": text_input_list,
        }

        

        # outputs = self.model.generate(
        #     input_ids, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id
        # )

        outputs = self.model.generate(
            **inference_generate_args
            )
        
        # print ('outputs',outputs)
        generated_texts = []
        for i, output in enumerate(outputs):
            prompt_length = len(input_ids[i])  # Calculate the length for each individual input
            generated_tokens = output[prompt_length:]  # Slice using the correct prompt length
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print("Generated Confidence Text:", generated_text)
            generated_texts.append(generated_text)

        # print ('outputs',outputs)
        # prompt_length = len(inference_generate_args['input_ids'][0])
        # generated_tokens = outputs[0][prompt_length:]
        # generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print("Generated Confidence Text:", generated_text) 
        # print ('generated_text',generated_texts)
        return generated_texts
        print ('outputs',outputs)
        # trimmed_outputs = []
        # for i in range(len(outputs)):
        #     prompt_length = len(input_ids['input_ids'][i])
        #     generated_tokens = outputs[i][prompt_length:]
        #     trimmed_outputs.append(generated_tokens)

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print ('response',responses)
        if len(text_input_list) == 1:
            return responses[0]
        return responses




        logit_list = []
        predictions = []

        # go back at doing model.generate

 


        for t in text_input_list:
            datapoint = (t,ground_truth_output)
            
            self.inference_step +=1
            self.predictor.prompt_class.inference_step = self.inference_step 
            self.predictor.prompt_class.current_sample = self.current_sample
            # have enumerate, pass i to 
            guess, probs, confidence = self.predictor.predict_and_confidence(datapoint)
            

            if guess == 'null': # return the logits with original label at 1
                probs = torch.zeros(self.n_classes+1, dtype=torch.float16)
                probs[-1] = 1.0
             
            predictions.append(guess)
            logit_list.append(torch.tensor(probs, device=self.device)) 


        print ('logit list:',logit_list)
        logit_tensor = torch.stack(logit_list)
        print('logit_tensor:', logit_tensor)
        return logit_tensor 
    
