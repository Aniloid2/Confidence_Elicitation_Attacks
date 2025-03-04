# #!/bin/bash
 
function run_test() {
  local gpu=$1
  local experiment_name=$2
  shift 2
  local params=("$@")

  # Create the experiment folder if it doesn't exist
  mkdir -p "$experiment_name"

  # Construct the command with the parameter names and values
  local cmd="CUDA_VISIBLE_DEVICES=$gpu python ../../attack_llm.py "
  cmd+="${params[@]} "
  cmd+="--experiment_name_folder $experiment_name > ${experiment_name}/${experiment_name}.txt 2>&1 &"

  # Execute the constructed command
  eval "$cmd"
}

# List of hyperparameter names
declare -a param_names=("model_type" "task" "num_transformations" "prompting_type" "search_method" "transformation_method" "n_embeddings" "similarity_threshold" "confidence_type" "k_pred" "similarity_technique" "prompt_shot_type" "index_order_technique" "temperature" "num_examples" "max_iter_i" "query_budget" "cache_transformers")




# random word schema

# # llama3 8b instruct
# # TextFooler   llama3 8b instruct
# declare -a config_gpu_0=("llama3" "sst2" 1 "step2_k_pred_avg" "greedy_use_search" "ceattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_1=("llama3" "strategyQA" 1 "step2_k_pred_avg" "greedy_use_search" "ceattack" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_2=("llama3" "ag_news" 1 "step2_k_pred_avg" "greedy_use_search" "ceattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")

# # # llama3 8b instruct  sspattack
# declare -a config_gpu_3=("llama3" "sst2" 2 "step2_k_pred_avg" "sspattack_search" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_4=("llama3" "strategyQA" 2 "step2_k_pred_avg" "sspattack_search" "sspattack" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_5=("llama3" "ag_news" 2 "step2_k_pred_avg" "sspattack_search" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")

# # #   llama3 8b instruct selffool
# declare -a config_gpu_6=("llama3" "sst2" 20 "step2_k_pred_avg" "black_box_search" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_7=("llama3" "strategyQA" 20 "step2_k_pred_avg" "black_box_search" "self_word_sub" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_8=("llama3" "ag_news" 20 "step2_k_pred_avg" "black_box_search" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500  "~/.cache/huggingface/")

# # #  llama3 8b instruct  texthoaxer
# declare -a config_gpu_9=("llama3" "sst2" 2 "step2_k_pred_avg" "texthoaxer_search" "texthoaxer" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_10=("llama3" "strategyQA" 2 "step2_k_pred_avg" "texthoaxer_search" "texthoaxer" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_11=("llama3" "ag_news" 2 "step2_k_pred_avg" "texthoaxer_search" "texthoaxer" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")

 
# # mistralv03 7b CEAttack   
# declare -a config_gpu_0=("mistralv03" "sst2" 1 "step2_k_pred_avg" "greedy_use_search" "ceattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_1=("mistralv03" "strategyQA" 1 "step2_k_pred_avg" "greedy_use_search" "ceattack" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_2=("mistralv03" "ag_news" 1 "step2_k_pred_avg" "greedy_use_search" "ceattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")

# # # mistralv03 7b  sspattack
# declare -a config_gpu_3=("mistralv03" "sst2" 2 "step2_k_pred_avg" "sspattack_search" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_4=("mistralv03" "strategyQA" 2 "step2_k_pred_avg" "sspattack_search" "sspattack" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_5=("mistralv03" "ag_news" 2 "step2_k_pred_avg" "sspattack_search" "sspattack" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")

# # #   mistralv03 7b selffool
# declare -a config_gpu_6=("mistralv03" "sst2" 20 "step2_k_pred_avg" "black_box_search" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_7=("mistralv03" "strategyQA" 20 "step2_k_pred_avg" "black_box_search" "self_word_sub" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_8=("mistralv03" "ag_news" 20 "step2_k_pred_avg" "black_box_search" "self_word_sub" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")

# # #  mistralv03 7b  texthoaxer
# declare -a config_gpu_9=("mistralv03" "sst2" 2 "step2_k_pred_avg" "texthoaxer_search" "texthoaxer" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_10=("mistralv03" "strategyQA" 2 "step2_k_pred_avg" "texthoaxer_search" "texthoaxer" 10 0.5 "weighted_confidence" 6 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")
# declare -a config_gpu_11=("mistralv03" "ag_news" 2 "step2_k_pred_avg" "texthoaxer_search" "texthoaxer" 10 0.5 "weighted_confidence" 20 "USE" "zs" "random" 0.001 500 5 500 "~/.cache/huggingface/")




# List of GPU IDs to use and their corresponding configurations
# Extend or remove lines as needed
declare -A gpu_config_map=(
  [0]="config_gpu_0"
  [1]="config_gpu_1"
  [2]="config_gpu_2"
  [3]="config_gpu_3" 
)

# List of GPU IDs to use
# Needs to match number of gpu config
gpus=( 0 1 2 3  )

# If you want to test on specific GPUs, you can directly assign the `gpus` array like below
# gpus=(3)
 
# Iterate over each GPU ID
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
  experiment_name="Attack_model_EN${params[num_examples]}_${params[prompting_type]}_${params[model_type]}_${params[task]}_NT${params[num_transformations]}_Bs${params[similarity_threshold]}_CT${params[confidence_type]}_KP${params[k_pred]}_ST${params[similarity_technique]}_PST${params[prompt_shot_type]}_IOT${params[index_order_technique]}_SM${params[search_method]}_TM${params[transformation_method]}_NE${params[n_embeddings]}_TMP${params[temperature]}_MIT${params[max_iter_i]}_QB${params[query_budget]}"
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