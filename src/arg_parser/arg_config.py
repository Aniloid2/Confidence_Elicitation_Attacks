

import argparse
import os
from src.utils.shared.misc import set_logging
def get_args():
    parser = argparse.ArgumentParser(description='Argument parser for model configuration')

    # Define the arguments
    parser.add_argument('--model_type', type=str, default='llama3', choices=['llama3', 'Llama-3.2-11B-Vision-Instruct', 'gemma-2-9b-it', 'mistralv03', 'Mistral-Nemo-Instruct-2407', 'Qwen2.5-7B-Instruct', 'gpt-4o', 'other_model_types_here'],
                        help='Type of the model to use')
    parser.add_argument('--model_precision', type=str, default='float32', choices=['float32','float16'],
                        help='Type of the model to use')
    parser.add_argument('--task', type=str, default='strategyQA', choices=['sst2','ag_news','strategyQA','rte','qqp', 'other_tasks_here'],
                        help='Task to perform')
    parser.add_argument('--task_structure', type=str, default='classification', choices=['classification','generation'],
                        help='What is the task strtructure, for classification we expect a static label list across all samples. In future we may extend to sequence to sequence modelling, dynamic label classification etc') 
    parser.add_argument('--confidence_type', type=str, default='weighted_confidence', choices=['verbal_confidence','verbal_numerical_confidence','weighted_confidence','single_token_mix'],
                        help='type of confidence levels')
    parser.add_argument('--prompting_type', type=str, default='step2_k_pred_avg', choices=[ 'empirical_confidence','step2_k_pred_avg' ,'other_prompting_types_here'],
                        help='Type of prompting to use')
    parser.add_argument('--search_method', type=str, default='black_box_search', choices=['black_box_search','greedy_use_search','sspattack_search','texthoaxer_search'],
                        help='Type of search technique')
    parser.add_argument('--transformation_method', type=str, default='ceattack', choices=['ceattack','sspattack','texthoaxer','self_word_sub'],
                        help='Type of transformations to use')
    parser.add_argument('--n_embeddings', type=int, default=10, help='Type of prompting to use')
    parser.add_argument('--prompt_shot_type', type=str, default='zs', choices=['zs', 'other_shot_types_here'],
                        help='Type of prompt shot to use')
    parser.add_argument('--k_pred', type=int, default=20,
                        help='Number of predictions to perform')
    parser.add_argument('--max_iter_i', type=int, default=5,
                        help='Number of iterations to perform during search')
    parser.add_argument('--query_budget', type=int, default=500,
                        help='Attack query budget')
    parser.add_argument('--num_examples', type=int, default=500,
                        help='Number of examples to evaluate on')                    
    parser.add_argument('--similarity_technique', type=str, default='USE',
                        help='similarity technique (USE or BERTScore)'),
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='similarity threshold')
    parser.add_argument('--num_transformations', type=int, default=20,
                        help='Number of transformations to perform')
    parser.add_argument('--index_order_technique', type=str, default='random', choices=['random','delete' ,'other_techniques_here'],
                        help='Index order technique to use')
    parser.add_argument('--cache_transformers', type=str, default='/mnt/hdd1/brian/hub',
                        help='Directory for transformers cache')
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser('~/.cache/CEAttacks'),
                        help='Directory for caching files')
    parser.add_argument('--experiment_name_folder', type=str, default='attack_calibrated_model',
                        help='Folder name for the experiment')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature at inference')        
    parser.add_argument('--ternary_plot', type=bool, default=False,
                        help='Plot the dirichlet distribution if True of the entire dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Set seed')
    parser.add_argument('--api_key', type=str, default='',
                        help='set open ai api key')
    args = parser.parse_args()

    high_level_folder = args.experiment_name_folder
    test_folder = os.path.join(high_level_folder, f'{args.model_type}_{args.task}_log_folder') 
    args.test_folder = test_folder 
    args.high_level_folder = high_level_folder 
    args.name_of_test = f'EN{str(args.num_examples)}_MT{args.model_type}_TA{args.task}_PT{args.prompting_type}_PST{args.prompt_shot_type}_ST{args.similarity_technique}_NT{args.num_transformations}'

    args.ceattack_logger = set_logging(args)
    
    method_to_type = {
        'ceattack': 'word_level',
        'sspattack': 'word_level',
        'texthoaxer': 'word_level',
        'self_word_sub': 'sentence_level', # This is technically a word level attack, however, since the model can generate anything we have to treat it more like a sentence level attack since my current constraints can't keep track of which word changes unless it's an 1 to 1 mapping.
    }
    selected_method = args.transformation_method
    if selected_method in method_to_type:
        args.transformation_type = method_to_type[selected_method]
    else:
        args.transformation_type = 'sentence_level'
    args.ceattack_logger.info(f"Selected transformation method '{selected_method}' is of type '{method_to_type[selected_method]}'.")
    print(f"Selected transformation method '{selected_method}' is of type '{method_to_type[selected_method]}'.")

    # we use this mostly to define which constraints to use on the tasks. for example strategyQA will have a NoNounConstraint since otherwise we may change name of places and people
    task_to_type = {
        'sst2': 'classification',
        'ag_news': 'classification',
        'mnli': 'classification',
        'rte': 'classification',
        'qqp': 'classification',
        'qnli': 'classification', # it's convenient to classify it as classification althou it's technically question answering
        'strategyQA': 'question_answering',
        'triviaQA': 'question_answering', 
    }

    selected_task = args.task
    if selected_task in task_to_type:
        args.task_type = task_to_type[selected_task]
    else:
        args.task_type = 'classification'
    args.ceattack_logger.info(f"Selected transformation method '{selected_task}' is of type '{task_to_type[selected_task]}'.")
    print(f"Selected transformation method '{selected_task}' is of type '{task_to_type[selected_task]}'.")

    return args
 

 
 