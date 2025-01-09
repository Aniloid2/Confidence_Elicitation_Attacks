# #!/bin/bash

# run_test() {
#   local model_type=$1
#   local task=$2
#   local gpu=$3
#   local experiment_name=$4
#   local num_transformations=$5
#   local prompting_type=$6
#   local similarity_threshold=$7
#   local confidence_type=$8
#   local k_pred=$9
#   local similarity_technique=${10}
#   local prompt_shot_type=${11}
#   local index_order_technique=${12}

#   # Create the experiment folder if it doesn't exist
#   mkdir -p $experiment_name

#   # Run the test and save the output to a log file within the experiment folder
#   CUDA_VISIBLE_DEVICES=$gpu python ../../robustness_eval_huggingface.py \
#     --model_type $model_type \
#     --task $task \
#     --prompting_type $prompting_type \
#     --prompt_shot_type $prompt_shot_type \
#     --k_pred $k_pred \
#     --similarity_technique $similarity_technique \
#     --num_transformations $num_transformations \
#     --index_order_technique $index_order_technique \
#     --cache_transformers /mnt/hdd/brian/hub \
#     --confidence_type $confidence_type \
#     --experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &
# }

# gpus=(0 1 2 3)
# # Test configurations: each row corresponds to a set of hyperparameters for a specific GPU
# # model type, task, number transformations, prompting_type, similarity threshold, confidence elicitation type, k_pred, semantic sim type, prompting type, top words type
# declare -a config_gpu_0=("llama3" "strategyQA" 1 "step2_k_pred_avg" 0.8 "weighted_confidence" 3 "USE" "zs" "random")
# declare -a config_gpu_1=("llama3" "ag_news" 1 "step2_k_pred_avg" 0.8 "weighted_confidence" 20 "USE" "zs" "random")
# declare -a config_gpu_2=("llama3" "sst2" 1 "step2_k_pred_avg" 0.8 "weighted_confidence" 20 "USE" "zs" "random")
# declare -a config_gpu_3=("llama3" "strategyQA" 1 "step2_k_pred_avg" 0.85 "weighted_confidence" 20 "USE" "fs" "random")

# # List of GPU configurations
# configs=(config_gpu_0 config_gpu_1 config_gpu_2)
# # config_gpu_3)

# # Iterate over GPU IDs and specify the configuration for each GPU
# for idx in ${!gpus[@]}; do
#   gpu=${gpus[$idx]}
#   eval config=(\"\${${configs[gpu]}[@]}\")
#   model=${config[0]}
#   task=${config[1]}
#   num_transformation=${config[2]}
#   prompting_type=${config[3]}
#   similarity_threshold=${config[4]}
#   confidence_type=${config[5]}
#   k_pred=${config[6]}
#   similarity_technique=${config[7]}
#   prompt_shot_type=${config[8]}
#   index_order_technique=${config[9]}

#   # Define a unique experiment name
#   experiment_name="experiment_${model}_${task}_NT${num_transformation}_PT${prompting_type}_BS${similarity_threshold}_CT${confidence_type}_KP${k_pred}_ST${similarity_technique}_PST${prompt_shot_type}_IOT${index_order_technique}"
#   echo "Experiment name: $experiment_name on GPU $gpu"

#   # Run the test in the background and log output to a file within the experiment folder
#   run_test $model $task $gpu $experiment_name $num_transformation $prompting_type $similarity_threshold $confidence_type $k_pred $similarity_technique $prompt_shot_type $index_order_technique
# done

# # Wait for all background jobs to complete
# wait

# echo "All tests have been completed."



#!/bin/bash

# Function to run the test
# run_test() {
#   local -n params=$1
#   local gpu=$2
#   local experiment_name=$3

#   # Create the experiment folder if it doesn't exist
#   mkdir -p $experiment_name

#   # Construct the command with the parameter names and values
#   cmd="CUDA_VISIBLE_DEVICES=$gpu python ../../robustness_eval_huggingface.py "
#   for param in "${!params[@]}"; do
#     cmd+="--$param ${params[$param]} "
#   done
#   cmd+="--experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &"

#   # Execute the constructed command
#   eval $cmd
# }

# # List of hyperparameter names
# declare -a param_names=("model_type" "task" "num_transformations" "prompting_type" "search_method" "transformation_method" "n_embeddings" "similarity_threshold" "confidence_type" "k_pred" "similarity_technique" "prompt_shot_type" "index_order_technique")

# # Test configurations: each row corresponds to a set of hyperparameters for a specific GPU
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random")
# declare -a config_gpu_1=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 3 "USE" "zs" "random")
# declare -a config_gpu_2=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random")
# declare -a config_gpu_3=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random")

# # List of GPU configurations
# configs=(config_gpu_0 config_gpu_1 config_gpu_2 config_gpu_3)

# # List of GPU IDs to use
# gpus=(0 1 2 3)

# # Iterate over each GPU ID
# for idx in ${!gpus[@]}; do
#   gpu=${gpus[$idx]}
#   eval config=(\"\${${configs[$idx]}[@]}\")

#   # Define an associative array for parameters
#   declare -A params
#   for i in ${!param_names[@]}; do
#     params[${param_names[$i]}]=${config[$i]}
#   done

#   # Define a unique experiment name
#   experiment_name="experiment_baseline_${params[prompting_type]}_${params[model_type]}_${params[task]}_NT${params[num_transformations]}_Bs${params[similarity_threshold]}_CT${params[confidence_type]}_KP${params[k_pred]}_ST${params[similarity_technique]}_PST${params[prompt_shot_type]}_IOT${params[index_order_technique]}_SM${params[search_method]}_TM${params[transformation_method]}_NE${params[n_embeddings]}"
#   echo "Experiment name: $experiment_name on GPU $gpu"

#   # Run the test in the background and log output to a file within the experiment folder
#   run_test params $gpu $experiment_name
# done

# # Wait for all background jobs to complete
# wait

# echo "All tests have been completed."



function run_test() {
  local gpu=$1
  local experiment_name=$2
  shift 2
  local params=("$@")

  # Create the experiment folder if it doesn't exist
  mkdir -p "$experiment_name"

  # Construct the command with the parameter names and values
  local cmd="CUDA_VISIBLE_DEVICES=$gpu python ../../robustness_eval_huggingface.py "
  cmd+="${params[@]} "
  cmd+="--experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &"

  # Execute the constructed command
  eval "$cmd"
}

# List of GPU IDs to use

# List of hyperparameter names
declare -a param_names=("model_type" "task" "num_transformations" "prompting_type" "search_method" "transformation_method" "n_embeddings" "similarity_threshold" "confidence_type" "k_pred" "similarity_technique" "prompt_shot_type" "index_order_technique" "temperature" "num_examples" "max_iter_i" "query_budget")

# Test configurations: each row corresponds to a set of hyperparameters for a specific GPU
# declare -a config_gpu_0=("mistralv03" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500)
# declare -a config_gpu_1=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500)
# declare -a config_gpu_2=("mistralv03" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500)

# declare -a config_gpu_3=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500)
# declare -a config_gpu_4=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500)
# declare -a config_gpu_5=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500)

# declare -a config_gpu_0=("llama3_2_11b" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_1=("llama3_2_11b" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("llama3_2_11b" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500)

# declare -a config_gpu_0=("qwen2-7b-instruct" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_0=("qwen1.5-14b-chat-int8" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_1=("qwen1.5-14b-chat-int8" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500)

# declare -a config_gpu_0=("qwen2.5-14b-instruct" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("qwen2.5-14b-instruct" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_3=("qwen2.5-14b-instruct" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500)

# declare -a config_gpu_0=("mistral-nemo-instruct-2407" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("mistral-nemo-instruct-2407" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_3=("mistral-nemo-instruct-2407" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500)

#gpt-4o-llama3
# declare -a config_gpu_0=("gpt-4o-llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("gpt-4o-llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_3=("gpt-4o-llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500)

# declare -a config_gpu_0=("llama3" "rte" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("llama3" "qqp" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_3=("llama3" "qnli" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_1=("llama3" "mnli" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)

# declare -a config_gpu_0=("mistralv03" "rte" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("mistralv03" "qqp" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_3=("mistralv03" "qnli" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_1=("mistralv03" "mnli" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)

 
# declare -a config_gpu_0=("llama3" "sst2" 1 "empirical_confidence" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 1 500 5 500)
# declare -a config_gpu_1=("llama3" "ag_news" 1 "empirical_confidence" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 1 500 5 500)
# declare -a config_gpu_3=("llama3" "strategyQA" 1 "empirical_confidence" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 1 500 5 500)

# declare -a config_gpu_0=("mistralv03" "sst2" 1 "empirical_confidence" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 1 500 5 500)
# declare -a config_gpu_1=("mistralv03" "ag_news" 1 "empirical_confidence" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 1 500 5 500)
# declare -a config_gpu_3=("mistralv03" "strategyQA" 1 "empirical_confidence" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 1 500 5 500)

declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 1 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_2=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "verbal_numerical_confidence" 20 "USE" "zs" "random" 0.001 500 5 500)
# declare -a config_gpu_3=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "verbal_numerical_confidence" 1 "USE" "zs" "random" 0.001 500 5 500)

 
# List of GPU IDs to use and their corresponding configurations
declare -A gpu_config_map=(
  [0]="config_gpu_0"
  [1]="config_gpu_1"
  [2]="config_gpu_2"
  [3]="config_gpu_3" 
)

gpus=( 0 1 2 3    )

# If you want to test on specific GPUs, you can directly assign the `gpus` array like below
# gpus=(3)

# Iterate over each GPU ID
for gpu in "${gpus[@]}"; do
  config_name="${gpu_config_map[$gpu]}"
  eval config=( "\"\${${config_name}[@]}\"" )

  # Define an associative array for parameters
  declare -A params
  for i in "${!param_names[@]}"; do
    params["${param_names[$i]}"]="${config[$i]}"
  done

  # Define a unique experiment name
  experiment_name="Check_calibration_EN${params[num_examples]}_${params[prompting_type]}_${params[model_type]}_${params[task]}_NT${params[num_transformations]}_Bs${params[similarity_threshold]}_CT${params[confidence_type]}_KP${params[k_pred]}_ST${params[similarity_technique]}_PST${params[prompt_shot_type]}_IOT${params[index_order_technique]}_SM${params[search_method]}_TM${params[transformation_method]}_NE${params[n_embeddings]}_TMP${params[temperature]}_MIT${params[max_iter_i]}_QB${params[query_budget]}"
  echo "Experiment name: $experiment_name on GPU $gpu"

  # Convert the associative array to a list of command-line options
  params_list=()
  for k in "${!params[@]}"; do
    params_list+=("--$k" "${params[$k]}")
  done

  # Run the test in the background and log output to a file within the experiment folder
  run_test "$gpu" "$experiment_name" "${params_list[@]}"
done

# Wait for all background jobs to complete
wait

echo "All tests have been completed."