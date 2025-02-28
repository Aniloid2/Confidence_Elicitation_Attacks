import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack import Attacker, AttackArgs

import torch.nn.functional as F
import csv 
import torch


from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.search_methods import AlzantotGeneticAlgorithm

from textattack.shared import AttackedText
from textattack.attack import Attack

from textattack.search_methods import GreedyWordSwapWIR,  BeamSearch

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)

from textattack.constraints.semantics import WordEmbeddingDistance

from src.custom_constraints.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics.bert_score import BERTScore
from textattack.constraints.grammaticality import PartOfSpeech
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
 

import os 


from numpy.random import dirichlet

from src.arg_parser.arg_config import get_args
args = get_args()

from src.arg_parser.set_cache import set_huggingface_cache
set_huggingface_cache(args)

from src.utils.shared.misc import environment_setup
args = environment_setup(args) 

# from src.utils.shared.misc import set_logging
# args.ceattack_logger = set_logging(args)




# Ensure both the high-level and specific directories are created
os.makedirs(args.test_folder, exist_ok=True)

# name_of_test = f'EN{str(args.num_examples)}_MT{args.model_type}_TA{args.task}_PT{args.prompting_type}_PST{args.prompt_shot_type}_ST{args.similarity_technique}_NT{args.num_transformations}'



from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP,TASK_N_CLASSES,MODEL_INFO 


args.n_classes =  TASK_N_CLASSES[args.task] 
args.confidence_type_dict = CONFIDENCE_LEVELS[args.confidence_type] 
args.confidence_map_dict = CONFIDENCE_MAP[args.confidence_type] 
model_info = MODEL_INFO[args.model_type]
 
args.model_name =  model_info['model_name']
args.start_prompt_header = model_info['start_prompt_header']
args.end_prompt_footer = model_info['end_prompt_footer']


from src.utils.shared.misc import set_stopwords
stopwords = set_stopwords(args) 
constraints = [ RepeatModification(),StopwordModification(stopwords=stopwords)]

import math
if args.similarity_technique == 'USE':
    angular_use_threshold = args.similarity_threshold
    
    threshold = 1 - (angular_use_threshold) / math.pi 
    compare_against_original = True
    window_size = None 
    skip_text_shorter_than_window=False 

    use_constraint = UniversalSentenceEncoder(
                threshold=threshold,
                metric="angular",
                compare_against_original=compare_against_original,
                window_size=window_size,
                skip_text_shorter_than_window=skip_text_shorter_than_window,
            )
elif args.similarity_technique == 'BERTScore':
    
    threshold = args.similarity_threshold
    use_constraint = BERTScore(min_bert_score =threshold)


args.use_constraint = use_constraint # passing epsilon bound as a argument allows to use it more flexibly, currently textattack applies it after a get_transformations call if we add it as a constraint
args.ceattack_logger.info(f'Using Epsilon object: \n {args.use_constraint} with threshold: {threshold}')

constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))

input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
)
constraints.append(input_column_modification)


if args.task_type == 'question_answering':
    constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
    if args.transformation_type == 'word_level':
        from src.custom_constraints.swap_constraints import NoNounConstraint
        constraints.append(NoNounConstraint())
else:
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

from src.utils.shared.misc import initialize_device

args.device = initialize_device(args)




from src.utils.shared import load_data
dataset_class, label_names = load_data(args)

from src.utils.shared import SimpleDataset
dataset_class =  SimpleDataset(dataset_class,label_names = label_names ) 
args.dataset = dataset_class


from src.inference.inference_config import DYNAMIC_INFERENCE

 
import re


import random



from src.llm_wrappers.huggingface_llm_wrapper import HuggingFaceLLMWrapper
from src.llm_wrappers.chatgpt_llm_wrapper import ChatGPTLLMWrapper


args.dataset = dataset_class
if 'gpt-4o' in args.model_type: 
    model_wrapper = ChatGPTLLMWrapper(**vars(args))
else:
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name ,cache_dir=args.cache_transformers,trust_remote_code=True  )
    args.model = AutoModelForCausalLM.from_pretrained(args.model_name , cache_dir=args.cache_transformers,trust_remote_code=True)
    model_wrapper = HuggingFaceLLMWrapper(**vars(args))
args.model = model_wrapper



args.predictor = DYNAMIC_INFERENCE[args.prompting_type](**vars(args))






# define goal function, this is for now classification, but could be, in theory, other objectives
from src.goal_function_algorithms.predict_and_confidence_goal_function import Prediction_And_Confidence_GoalFunction
goal_function = Prediction_And_Confidence_GoalFunction(model_wrapper,**vars(args))
args.goal_function = goal_function
 
from src.transformation_algorithms.transformation_config import DYNAMIC_TRANSFORMATION
transformation = DYNAMIC_TRANSFORMATION[args.transformation_method](**vars(args))
args.transformation = transformation



from textattack.search_methods import SearchMethod




import copy




from collections import defaultdict







 
from src.search_algorithms.search_config import DYNAMIC_SEARCH
search_method = DYNAMIC_SEARCH[args.search_method](**vars(args))



greedy_attack = Attack(goal_function, constraints, transformation, search_method)






print ('Saving to:',args.test_folder,args.name_of_test )
attack_args = AttackArgs(
    num_examples=args.num_examples,
    log_to_csv=os.path.join(args.test_folder, f'normal_{args.name_of_test}.csv'),  # Adjusted to save in test_folder
    checkpoint_interval=1000,
    checkpoint_dir="checkpoints",
    disable_stdout=True,
    parallel=False,
    num_workers_per_device=8, 
    csv_coloring_style=None,
    enable_advance_metrics=True,
)



attacker_greedy_class_0 = Attacker(greedy_attack, dataset_class, attack_args)


import time

 
start_time = time.time()
 
results = attacker_greedy_class_0.attack_dataset()




end_time = time.time()


 
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
hours, minutes = divmod(minutes, 60)

# Print the elapsed time
print(f"Execution Time: {int(hours):03}:{int(minutes):02}:{int(seconds):02}")
args.logging.info(f"Execution Time: {int(hours):03}:{int(minutes):02}:{int(seconds):02}")

from src.logging import log_results, log_results_extension, evaluate_results

log_results(results, test_folder=args.test_folder, name_of_test = args.name_of_test, name_log = 'normal', args = args)
 
log_results_extension(results, test_folder=args.test_folder, name_of_test = args.name_of_test, name_log = 'normal', args = args )





evaluate_results(results, args)

import pickle
name_log = 'results'
file_path = os.path.join(args.test_folder, f'{name_log}_{args.name_of_test}.pkl')

# Saving the `results` list to a file
with open(file_path, 'wb') as file:
    pickle.dump(results, file)

print(f'Results saved to {file_path}')

# To load the `results` list from the file
with open(file_path, 'rb') as file:
    loaded_results = pickle.load(file)

for result in loaded_results:
    original_text = result.original_result.attacked_text
    perturbed_text = result.perturbed_result.attacked_text
    print ('original_text',original_text)
    print ('perturbed_text',perturbed_text)




  
filtered_results = []
for i, result in enumerate(results):   
    if result.original_result.output != args.n_classes: 
        filtered_results.append(result) 

log_results(filtered_results, test_folder=args.test_folder, name_of_test = args.name_of_test, name_log = 'filtered', args = args)

