# config.py
import argparse
import os
def get_args():
    parser = argparse.ArgumentParser(description='Argument parser for model configuration')

    # Define the arguments
    parser.add_argument('--model_type', type=str, default='llama3', choices=['llama2','mistral','mistralv03','mistral-8b-instruct-2410','mistral-small-instruct-2409','qwen2.5-14b-instruct','qwen2.5-14b-instruct-int8','qwen1.5-14b-chat','qwen2-7b-instruct','qwen2.5-7b-instruct','mistral-nemo-instruct-2407','llama3','gpt-4o-llama3','llama3_1','llama3_2_1b','llama3_2_3b','llama3_2_11b','phi3','llama2_13b','gemma2', 'other_model_types_here'],
                        help='Type of the model to use')
    parser.add_argument('--model_precision', type=str, default='float32', choices=['float32','float16'],
                        help='Type of the model to use')
    parser.add_argument('--task', type=str, default='strategyQA', choices=['sst2','ag_news','popQA','strategyQA','triviaQA', 'mnli','rte','qqp','qnli', 'other_tasks_here'],
                        help='Task to perform')
    parser.add_argument('--task_structure', type=str, default='classification', choices=['classification','generation'],
                        help='What is the task strtructure, for classification we expect a static label list across all samples. In future we may extend to sequence to sequence modelling, dynamic label classification etc') 
    parser.add_argument('--confidence_type', type=str, default='weighted_confidence', choices=['verbal_confidence','verbal_numerical_confidence','weighted_confidence','sherman_kent', 'yougov','single_token_mix'],
                        help='type of confidence levels')
    parser.add_argument('--prompting_type', type=str, default='step2_k_pred_avg', choices=['s1','w1','s1_black_box', '2step', 'empirical_confidence','k_pred_avg','step2_k_pred_avg' ,'other_prompting_types_here','e_guided_paraphrasing'],
                        help='Type of prompting to use')
    parser.add_argument('--search_method', type=str, default='black_box', choices=['black_box','greedy_search','greedy_search_use','greedy_search_use_hardlabel','sspattack','texthoaxer','greedy_search_withmin_use'],
                        help='Type of search technique')
    parser.add_argument('--transformation_method', type=str, default='word_swap_embedding', choices=['word_swap_embedding','sspattack','texthoaxer','self_word_sub','e_guided_paraphrasing'],
                        help='Type of transformations to use')
    parser.add_argument('--n_embeddings', type=int, default=10, help='Type of prompting to use')
    parser.add_argument('--prompt_shot_type', type=str, default='fs', choices=['fs','zs', 'other_shot_types_here'],
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
    parser.add_argument('--index_order_technique', type=str, default='prompt_top_k', choices=['prompt_top_k','random','delete' ,'other_techniques_here'],
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

    # Parse the arguments
    return parser.parse_args()
 

 
 