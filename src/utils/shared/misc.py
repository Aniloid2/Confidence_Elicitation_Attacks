from .globals import MODEL_INFO

def identify_correct_incorrect_labels(label_list, label_index):
    # Convert boolean to an index if necessary
    if isinstance(label_index, bool):
        label_index = int(label_index)  # True becomes 1, False becomes 0

    expected_prediction = [label_list[label_index]]
    incorrect_answers = list(set(label_list) - set([label_list[label_index]]))

    return expected_prediction, incorrect_answers


def set_stopwords():
    #Constraints
    # Define constraints (optional but recommended to refine the search space)
    stopwords = set(
                ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
            )
    extended_stopwords = set()
    for word in stopwords:
        extended_stopwords.add(word)             # Original lower-cased version
        extended_stopwords.add(word.upper())     # Fully upper-cased version
        extended_stopwords.add(word.capitalize())
    return extended_stopwords

import random
import numpy as np
import torch
from src.utils.shared.arg_config import get_args
import os
 

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def print_args_human_readable(namespace):
        print("Arguments Namespace:")
        for key, value in vars(namespace).items():
            print(f"  - {key.replace('_', ' ').capitalize()}: {value}")
 
import logging 
from datetime import datetime
def set_logging(test_folder, logging_level = 'debug'): 
    logger = logging.getLogger('Calibra')
    # Prevent propagation
    # logger.propagate = False 

    if logging_level == 'debug':
        log_level = logging.DEBUG
    elif logging_level == 'info':
        log_level = logging.INFO
    else:
        log_level = logging.INFO  # Default level if args is not as expected

    logs_folder = os.path.join(test_folder, 'logs')
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(logs_folder, f'{timestamp}_logfile.log')
    file_handler = logging.FileHandler(log_file_path)
    logger.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # logging.basicConfig(
    #     filename=logging_path,
    #     level=log_level,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
     
    return logger

def environment_setup():
    
    
     

    args = get_args()

    set_seed(args.seed)
    
    print_args_human_readable(args)
     
    
    # os.environ["HF_HOME"] = "/mnt/hdd/brian/"# args.cache_transformers# "/mnt/hdd/brian/"
    # os.environ["TRANSFORMERS_CACHE"] = "/mnt/hdd/brian/"
    cache_dir = args.cache_transformers # "/mnt/hdd/brian/hub"
    # os.environ['TFHUB_CACHE_DIR'] = cache_dir
    # os.environ['TRANSFORMERS_CACHE'] = cache_dir
    # os.environ['HF_DATASETS_CACHE'] = cache_dir

    high_level_folder = args.experiment_name_folder
    test_folder = os.path.join(high_level_folder, f'{args.model_type}_{args.task}_log_folder') 
    args.test_folder = test_folder 
    args.high_level_folder = high_level_folder 
 
    # Print the folder paths
    print('high_level_folder:', args.high_level_folder )
    print('test_folder:', args.test_folder)
    os.makedirs(args.test_folder, exist_ok=True)

    args.logging = set_logging(args.test_folder)
 
    return  args

from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
def initialize_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name ,cache_dir=args.cache_transformers,trust_remote_code=True  )
    model = AutoModelForCausalLM.from_pretrained(args.model_name , cache_dir=args.cache_transformers,trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def print_model_layer_dtype(model):
        print('\nModel dtypes:')
        for name, param in model.named_parameters():
            print(f"Parameter: {name}, Data type: {param.dtype}")
    # print_model_layer_dtype(model)
    if 'precision' in MODEL_INFO[args.model_type]:
        if MODEL_INFO[args.model_type]['precision'] == 'float32':
            pass
        elif MODEL_INFO[args.model_type]['precision'] == 'float16':
            model.half()
        else:
            ValueError(f'Other precision types are not yet supported, current model precision is {args.model_precision}' )
    else:
        pass
    # print_model_layer_dtype(model) 
    
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer's pad token is set
    model.config.pad_token_id = tokenizer.pad_token_id
    print ('tokenizer.pad_token',tokenizer.pad_token )
    print ( 'tokenizer.pad_token_id', tokenizer.pad_token_id)
    # args.update({'model': model,'device':device, 'tokenizer':tokenizer})
    # Check model dtype
    
    args.model = model
    args.device = device
    args.tokenizer = tokenizer 
    return args
