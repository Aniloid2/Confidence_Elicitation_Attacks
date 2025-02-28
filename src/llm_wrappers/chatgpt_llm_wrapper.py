from textattack.models.wrappers import ModelWrapper

import requests
import json
import time

class ChatGPTLLMWrapper(ModelWrapper):
    """A wrapper around HuggingFace for LLMs.

    Args:
        model: A HuggingFace pretrained LLM
        tokenizer: A HuggingFace pretrained tokenizer
    """

    def __init__(self, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.generated_max_length = 2 
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
    
        self.model = self
        
        
 


    def __call__(self, text_input_list):
        
        if isinstance(text_input_list, list):
            pass
        else:
            text_input_list = [text_input_list]

        generated_texts = []
        for text_input in text_input_list:

            api_key = self.api_key
            
            inference_generate_args = {
                
                "do_sample": self.general_generate_args['do_sample'],  # enable sampling
                "top_k": self.general_generate_args['top_k'], # top-k sampling
                "top_p": self.general_generate_args['top_p'], # nucleus sampling probability
                "temperature": self.temperature,  # sampling temperature
                "max_new_tokens": self.general_generate_args['max_new_tokens'], 
                 
            }
            extra_args = {
                "prompt": text_input,
            }
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            } 
            self.ceattack_logger.info(f"Payload: {extra_args['prompt']}")
            payload = {
                'model': 'gpt-4o',
                'messages': [
                    {'role': 'system', 'content': "You are an expert assistant"},
                    {'role': 'user', 'content': extra_args['prompt']},
                ],
                'max_tokens': inference_generate_args['max_new_tokens'],
                'n': 1,
                'stop': None,
                'temperature': inference_generate_args['temperature'],
                'logprobs': False, 
            }

            max_retries = 10
            wait_time = 60  # seconds

            for attempt in range(max_retries):
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                self.ceattack_logger.info(f"Raw response: \n {response}")
                print ('response',response )
                if response.status_code == 200:
                    response_data = response.json() 
                    choice = response_data['choices'][0]
                    message_content = choice['message']['content']
                    print('Response messege content: \n', message_content)
                    self.ceattack_logger.info(f"Response messege content: \n {message_content}")
                    generated_texts.append(message_content)
                    
                    break
                else:
                    print(f"Error: {response.status_code}")
                    self.ceattack_logger.info(f"Error: {response.status_code}")
                    print(response.text)

                    if attempt < max_retries - 1:  # Don't wait after the last attempt
                        print(f"Retrying in {wait_time} seconds...")
                        self.ceattack_logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
        if generated_texts:
            return generated_texts
        else:
            return None
        