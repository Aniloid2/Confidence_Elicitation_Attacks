# #!/bin/bash

# run_test() {
#   local model_type=$1
#   local task=$2
#   local gpu=$3
#   local experiment_name=$4
#   local num_transformations=$5
#   local prompting_type=$6
#   local similarity_threshold=$7

#   # Create the experiment folder if it doesn't exist
#   mkdir -p $experiment_name

#   # Run the test and save the output to a log file within the experiment folder
#   CUDA_VISIBLE_DEVICES=$gpu python ../../attack_llm_self_prompt.py \
#     --model_type $model_type \
#     --task $task \
#     --prompting_type $prompting_type \
#     --prompt_shot_type fs \
#     --similarity_threshold $similarity_threshold \
#     --num_transformations $num_transformations \
#     --index_order_technique random \
#     --cache_transformers /mnt/hdd/brian/ \
#     --experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &
# }

# # Test configurations
# declare -a models=("llama3" "mistral" "llama3" "mistral")
# declare -a tasks=("ag_news" "ag_news" "sst2" "sst2")
# declare -a num_transformations=("1" "1" "1" "1")
# declare -a prompting_types=("empirical" "empirical" "empirical" "empirical")
# declare -a similarity_thresholds=("0.85" "0.85" "0.85" "0.85")
 

# # You can also specify different models and tasks for each GPU if needed
# # models=("llama2" "mistral" "another_model" "yet_another_model")
# # tasks=("task1" "task2" "task3" "task4")

# # Iterate over GPU IDs and specify the model, task, num_transformations, and prompting_type for each GPU
# for gpu in $(seq 0 3); do
#   # Skip GPU numbers 1 and 2 if necessary
#   # if [ $gpu -eq 0 ] || [ $gpu -eq 2 ] || [ $gpu -eq 3 ]; then
#   #   continue
#   # fi
#   # if [ $gpu -eq 1 ] ; then
#   #   continue
#   # fi
#   echo "Starting experiment on GPU $gpu"
#   model=${models[gpu]}
#   task=${tasks[gpu]}
#   num_transformation=${num_transformations[gpu]}
#   prompting_type=${prompting_types[gpu]}
#   similarity_threshold=${similarity_thresholds[gpu]}

#   # Define a unique experiment name
#   experiment_name="experiment_baseline_${prompting_type}_${model}_${task}_NT${num_transformation}_Bs${similarity_threshold}"
#   echo "Experiment_name $experiment_name"
#   # Run the test in the background and log output to a file within the experiment folder
#   run_test $model $task $gpu $experiment_name $num_transformation $prompting_type $similarity_threshold
# done

# # Wait for all background jobs to complete
# wait

# echo "All tests have been completed."



#!/bin/bash

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
#   CUDA_VISIBLE_DEVICES=$gpu python ../../attack_llm_self_prompt.py \
#     --model_type $model_type \
#     --task $task \
#     --prompting_type $prompting_type \
#     --prompt_shot_type $prompt_shot_type \
#     --similarity_threshold $similarity_threshold \
#     --num_transformations $num_transformations \
#     --index_order_technique $index_order_technique \
#     --cache_transformers /mnt/hdd/brian/ \
#     --confidence_type $confidence_type \
#     --k_pred $k_pred \
#     --similarity_technique $similarity_technique \
#     --experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &
# }

# # List of GPU IDs to use
# gpus=(0 1 2 3)

# # Test configurations: each row corresponds to a set of hyperparameters for a specific GPU
# # model type, task, number transformations, prompting_type, similarity threshold, confidence elicitation type, k_pred, semantic sim type, prompting type, top words type
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" 0.5 "weighted_confidence" 20 "USE" "zs" "random")
# declare -a config_gpu_1=("llama3" "strategyQA" 1 "step2_k_pred_avg" 0.5 "weighted_confidence" 3 "USE" "zs" "random")
# declare -a config_gpu_2=("llama3" "ag_news" 1 "step2_k_pred_avg" 0.5 "weighted_confidence" 20 "USE" "zs" "random")
# declare -a config_gpu_3=("llama3" "strategyQA" 1 "step2_k_pred_avg" 0.5 "weighted_confidence" 20 "USE" "zs" "random")

# # List of GPU configurations
# configs=(config_gpu_0 config_gpu_1 config_gpu_2 config_gpu_3)

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
#   experiment_name="experiment_baseline_${prompting_type}_${model}_${task}_NT${num_transformation}_Bs${similarity_threshold}_CT${confidence_type}_KP${k_pred}_ST${similarity_technique}_PST${prompt_shot_type}_IOT${index_order_technique}"
#   echo "Experiment name: $experiment_name on GPU $gpu"

#   # Run the test in the background and log output to a file within the experiment folder
#   run_test $model $task $gpu $experiment_name $num_transformation $prompting_type $similarity_threshold $confidence_type $k_pred $similarity_technique $prompt_shot_type $index_order_technique
# done

# # Wait for all background jobs to complete
# wait

# echo "All tests have been completed."


# Function to run the test
# run_test() {
#   local -n params=$1
#   local gpu=$2
#   local experiment_name=$3

#   # Create the experiment folder if it doesn't exist
#   mkdir -p $experiment_name

#   # Construct the command with the parameter names and values
#   cmd="CUDA_VISIBLE_DEVICES=$gpu python ../../attack_llm_self_prompt.py "
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
# declare -a config_gpu_3=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 3 "USE" "zs" "random")

# # List of GPU configurations
# configs=(config_gpu_0 config_gpu_1 config_gpu_2 config_gpu_3)

# # List of GPU IDs to use
# gpus=(3)

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
  local cmd="CUDA_VISIBLE_DEVICES=$gpu python ../../attack_llm_self_prompt.py "
  cmd+="${params[@]} "
  cmd+="--experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &"

  # Execute the constructed command
  eval "$cmd"
}

# List of hyperparameter names
declare -a param_names=("model_type" "task" "num_transformations" "prompting_type" "search_method" "transformation_method" "n_embeddings" "similarity_threshold" "confidence_type" "k_pred" "similarity_technique" "prompt_shot_type" "index_order_technique" "temperature" "num_examples" "max_iter_i")

# Test configurations: each row corresponds to a set of hyperparameters for a specific GPU
# TextFooler MISTRALv03
# declare -a config_gpu_0=("mistralv03" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_1=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_2=("mistralv03" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)

#SSPAttack MISTRALv03
# declare -a config_gpu_0=("mistralv03" "sst2" 1 "step2_k_pred_avg" "sspattack" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_1=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "sspattack" "sspattack" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_2=("mistralv03" "ag_news" 1 "step2_k_pred_avg" "sspattack" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)

# Self Fool MISTRALv03
# declare -a config_gpu_0=("mistralv03" "sst2" 20 "step2_k_pred_avg" "black_box" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_1=("mistralv03" "strategyQA" 20 "step2_k_pred_avg" "black_box" "self_word_sub" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_2=("mistralv03" "ag_news" 20 "step2_k_pred_avg" "black_box" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)

# TextFooler  LLAMA3
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search" "word_swap_embedding" 50 0.5 "weighted_confidence" 20 "USE" "zs" "delete" 0.001 500 5)
# declare -a config_gpu_1=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_2=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)

# LLAMA3 sspattack
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "sspattack" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_2=("llama3" "strategyQA" 1 "step2_k_pred_avg" "sspattack" "sspattack" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_1=("llama3" "ag_news" 1 "step2_k_pred_avg" "sspattack" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)

#  LLAMA3 selffool
# declare -a config_gpu_0=("llama3" "sst2" 20 "step2_k_pred_avg" "black_box" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_1=("llama3" "strategyQA" 20 "step2_k_pred_avg" "black_box" "self_word_sub" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_2=("llama3" "ag_news" 20 "step2_k_pred_avg" "black_box" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)



# TextFooler Random, delete, top_K
# declare -a config_gpu_0=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_1=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.7 500 5)
# declare -a config_gpu_2=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "delete" 0.7 500 5)
# declare -a config_gpu_3=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "prompt_top_k" 0.7 500 5)


#TextFooler temperature 0.00001
# declare -a config_gpu_1=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 100 5)


# NUMBER OF ITERATIONS max_iter_i = 1, 5, 10, 15, 20 Ablation study
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 1)
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 5)
# declare -a config_gpu_1=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 10)
# declare -a config_gpu_2=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 15)
# declare -a config_gpu_3=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 20)

# NUMBER OF ITERATIONS max_iter_i = 1, 5, 10, 15, 20 Ablation study
# declare -a config_gpu_4=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 1 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 5)
# declare -a config_gpu_5=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 5 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 5)
# declare -a config_gpu_1=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 5)
# declare -a config_gpu_6=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 20 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 5)
declare -a config_gpu_1=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use" "word_swap_embedding" 50 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 100 5)

#No Feedback Ablation
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_1=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_1=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)

# declare -a config_gpu_2=("mistralv03" "sst2" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_1=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5)
# declare -a config_gpu_3=("mistralv03" "ag_news" 1 "step2_k_pred_avg" "greedy_search_use_hardlabel" "word_swap_embedding" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5)


# List of GPU IDs to use and their corresponding configurations
declare -A gpu_config_map=(
  [0]="config_gpu_0"
  [1]="config_gpu_1"
  [2]="config_gpu_2"
  [3]="config_gpu_3"
)

# List of GPU IDs to use
gpus=( 0 1 2 3 )

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
  experiment_name="Attack_model_EN${params[num_examples]}_${params[prompting_type]}_${params[model_type]}_${params[task]}_NT${params[num_transformations]}_Bs${params[similarity_threshold]}_CT${params[confidence_type]}_KP${params[k_pred]}_ST${params[similarity_technique]}_PST${params[prompt_shot_type]}_IOT${params[index_order_technique]}_SM${params[search_method]}_TM${params[transformation_method]}_NE${params[n_embeddings]}_TMP${params[temperature]}_MIT${params[max_iter_i]}"
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