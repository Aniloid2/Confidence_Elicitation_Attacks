
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
        # standard setting, depending on the inference template in src/inference we change these parameters
        self.general_generate_args = {
            "do_sample": True,  # enable sampling
            "top_k": 40,  # top-k sampling
            "top_p": 0.92,  # nucleus sampling probability
            "temperature": self.temperature,  # sampling temperature
            "max_new_tokens": 200,
            
        }

        self.general_tokenizer_encoding_args = {
            'return_tensors':"pt",
            }
        

    def __call__(self, text_input_list):
        
        self.device = next(self.model.parameters()).device
         
        self.ceattack_logger.debug(f"'Input Text List in Wrapper': \n {text_input_list}")
        
        inference_tokenizer_args = self.general_tokenizer_encoding_args
        inference_tokenizer_args['text'] = text_input_list

        inputs = self.tokenizer(**inference_tokenizer_args)
        
        
        input_ids = inputs['input_ids'].to(self.device) 
        att_ids = inputs['attention_mask'].to(self.device)  

        inference_generate_args = {
            "input_ids": input_ids,
            "attention_mask": att_ids, 
            "do_sample": self.general_generate_args['do_sample'],  # enable sampling
            "top_k": self.general_generate_args['top_k'],  # top-k sampling
            "top_p": self.general_generate_args['top_p'], # nucleus sampling probability
            "temperature": self.temperature,  # sampling temperature
            "max_new_tokens": self.general_generate_args['max_new_tokens'], 
            'pad_token_id': self.tokenizer.eos_token_id
        }

         

        outputs = self.model.generate(
            **inference_generate_args
            )
        
        
        generated_texts = []
        for i, output in enumerate(outputs):
            # extract only the generated text
            prompt_length = len(input_ids[i])
            generated_tokens = output[prompt_length:]  
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) 
            generated_texts.append(generated_text)

        return generated_texts
        