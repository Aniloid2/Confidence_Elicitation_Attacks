


CONFIDENCE_LEVELS = {
    'verbal_confidence': ['lowest', 'low', 'medium', 'high', 'highest'],
    'weighted_confidence': ['lowest', 'low', 'medium', 'high', 'highest'],
    'verbal_numerical_confidence': ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
    'sherman_kent': ['almost certainly', 'highly likely', 'very good chance', 'probable', 'likely', 'probably', 'we believe', 'better than even', 'about even', 'we doubt', 'improbable', 'unlikely', 'probably not', 'little chance', 'almost no chance', 'highly unlikely', 'chances are slight'],
    'yougov': ['Very unlikely', 'Unlikely', 'Not very likely', "Probably won't", 'Quite unlikely', 'Outside chance', 'Maybe', 'Might happen', 'Possibly', 'More likely than not', 'Quite likely', 'Likely', 'Probably will', 'Very likely', 'Almost certain'],
    'single_token_mix': ['Certain', 'Probable', 'Probably', 'Likely', 'Possibly', 'Maybe', 'Improbable', 'Unlikely', 'Impossible'],
    'gpt_thresholding': ['none', 'very low', 'low', 'below average', 'average', 'moderate','above average', 'high','very high', 'almost certain','certain']
}


CONFIDENCE_MAP = {'verbal_confidence':{
                        'lowest': 0,
                        'low': 25,
                        'medium': 50,
                        'high': 75,
                        'highest': 100
                    },
                    'weighted_confidence':{
                        'lowest': 1,
                        'low': 2,
                        'medium': 3,
                        'high': 4,
                        'highest': 5
                    },
                    'verbal_numerical_confidence':{
                        '0.0': 0.0,
                        '0.1': 0.1,
                        '0.2': 0.2,
                        '0.3': 0.3,
                        '0.4': 0.4,
                        '0.5': 0.5,
                        '0.6': 0.6,
                        '0.7': 0.7,
                        '0.8': 0.8,
                        '0.9': 0.9,
                        '1.0': 1.0,
                    },
                    'sherman_kent': {
                        'almost certainly':95,
                        'highly likely':90, 
                        'very good chance':85,
                        'probable':80, 
                        'likely':75, 
                        'probably':70, 
                        'we belive':65, 
                        'better than even':65, 
                        'about even':50, 
                        'we doubt':40, 
                        'improbable':35, 
                        'unlikely':30, 
                        'probably not':25, 
                        'little chance':20, 
                        'almost no chance':2, 
                        'highly unlikely':5, 
                        'chances are slight':10},
                    'yougov': {"very unlikely": 23.8,
                                "unlikely": 29.1,
                                "not very likely": 30.1,
                                "probably won't": 31.0,
                                "quite unlikely": 31.4,
                                "outside chance": 34.5,
                                "maybe": 49.5,
                                "50/50": 50.4,
                                "might happen": 53.1,
                                "possibly": 53.9,
                                "more likely than not": 62.0,
                                "quite likely": 66.3,
                                "likely": 66.5,
                                "probably will": 69.7,
                                "very likely": 75.9,
                                "almost certain": 79.8
                                },
                    'single_token_mix': { 
                        'certain':100,
                        'probable':80,
                        'probably':70, 
                        "likely": 66.5,
                        "maybe": 49.5,
                        "unlikely": 29.1,
                        "possibly": 53.9,
                        'improbable':35, 
                        'impossible':0,

                                },
                    'gpt_thresholding': {
                        'none': 0,
                        'very low': 10,
                        'low': 20,
                        'below average': 30,
                        'average': 40,
                        'moderate': 50,
                        'above average': 60,
                        'high': 70,
                        'very high': 80,
                        'almost certain': 90,
                        'certain': 100
                    }
    
    
    }



TASK_N_CLASSES = {
    'sst2': 2,
    'ag_news': 4,
    'popQA': 2,
    'strategyQA': 2,
    'triviaQA': 2,
    'mnli': 3,
    'qqp': 2,
    'rte': 2,
    'qnli': 2,
}

MODEL_INFO = {
    'llama3': {
        'model_name': "meta-llama/Meta-Llama-3-8B-Instruct",
        'start_prompt_header': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        'end_prompt_footer': "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    }, # llama3
    'Llama-3.2-11B-Vision-Instruct': {
        'model_name': "meta-llama/Llama-3.2-11B-Vision-Instruct",
        'start_prompt_header': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
        'end_prompt_footer': "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    },
    'gemma-2-9b-it':{
        'model_name': 'google/gemma-2-9b-it',
        'start_prompt_header':'<bos><start_of_turn>user',
        'end_prompt_footer':'<end_of_turn><start_of_turn>model'
    },
    'mistralv03': {
        'model_name': "mistralai/Mistral-7B-Instruct-v0.3",
        'start_prompt_header': "<s>[INST]",
        'end_prompt_footer': "[/INST]" #mistralv03
    },
    'Mistral-Nemo-Instruct-2407': {
        'model_name': "mistralai/Mistral-Nemo-Instruct-2407",
        'start_prompt_header': "<s>[INST]",
        'end_prompt_footer': "[/INST]",
        'precision': 'float16',
    },
    'Qwen2.5-7B-Instruct': {
        'model_name': "Qwen/Qwen2.5-7B-Instruct",
        'start_prompt_header': "<|im_start|>",
        'end_prompt_footer': "<|im_end|><|endoftext|>"
    },  
    'gpt-4o': {
        'model_name': None,
        'start_prompt_header': None,
        'end_prompt_footer': None
    },
} 
   