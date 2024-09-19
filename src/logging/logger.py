import os
from textattack.loggers import AttackLogManager

def log_results(results, test_folder, name_of_test, name_log ,args):
    """
    Logs the results provided without filtering.

    Parameters:
    - results (list): The list of results to be logged.
    - args (object): An object that contains the number of classes (`args.n_classes`).
    - test_folder (str): The folder where the output files will be saved.
    - name_of_test (str): The name to be used for the output files.
    - name_log (str): The name to be used as a prefix for the output files.
    """
    # Log manager to manage attack results
    attack_log_manager = AttackLogManager(metrics=None)
    attack_log_manager.enable_advance_metrics = True

    for i, result in enumerate(results):
        attack_log_manager.log_result(result)

    # print(f'{name_log}_results:', results, len(results))

    # Generate summary
    output_file_name = f'{name_log}_{name_of_test}.txt' 

    attack_log_manager.add_output_file(
        filename=os.path.join(test_folder, output_file_name), 
        color_method="file"
    ) 

    attack_log_manager.enable_stdout()
    attack_log_manager.log_summary()

# Example usage:
# log_results(results, args, test_folder, name_of_test, name_log)

 
import pandas as pd
import textattack
from textattack.attack_results import SuccessfulAttackResult
import numpy as np

def log_results_extension(results, test_folder, name_of_test, name_log, args):
    """
    Extends the logging of results by adding additional metrics and saving them to a CSV file.

    Parameters:
    - test_folder (str): The folder where the CSV file is located and will be saved.
    - name_of_test (str): The name used for the CSV file.
    - results (list): The list of results to be processed.
    """
    # Define the file path
    file_path = os.path.join(test_folder, f'{name_log}_{name_of_test}.csv')

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Add necessary columns if they do not exist
    required_columns = ['original_perplexity', 'attack_perplexity', 'attack_use_score', 'num_words_perturbed']
    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    def count_successful_samples_quereis(results):
        total_queries = 0
        succ_samples = 0
        for i, result in enumerate(results):
            if isinstance(result, SuccessfulAttackResult):
                total_queries += result.num_queries
                succ_samples += 1
        return total_queries / succ_samples if succ_samples != 0 else 0

    successful_samples_queries = 0

    for i, result in enumerate(results):
        print('result queries', result.num_queries)

        # Calculate Perplexity
        try:
            ppl = textattack.metrics.quality_metrics.Perplexity().calculate([result])
            df.at[i, 'original_perplexity'] = ppl['avg_original_perplexity']
            df.at[i, 'attack_perplexity'] = ppl['avg_attack_perplexity']
        except Exception as e:
            df.at[i, 'original_perplexity'] = -1
            df.at[i, 'attack_perplexity'] = -1
            print('Error calculating perplexity:', e)

        # Calculate USE Metric
        try:
            print (',result use point ',result)
            usem = textattack.metrics.quality_metrics.USEMetric().calculate([result])
            print ('usem',usem)
            if usem['avg_attack_use_score'] == np.nan:
                df.at[i, 'attack_use_score'] = 1.0
            else:
                df.at[i, 'attack_use_score'] = usem['avg_attack_use_score']
        except Exception as e:
            df.at[i, 'attack_use_score'] = -1
            print('Error calculating USE score:', e)

        # Calculate Number of Words Perturbed
        try:
            original_text = result.original_result.attacked_text
            perturbed_text = result.perturbed_result.attacked_text
            num_perturbed_words = original_text.words_diff_num(perturbed_text)
            df.at[i, 'num_words_perturbed'] = num_perturbed_words
        except Exception as e:
            df.at[i, 'num_words_perturbed'] = -1
            print('Error calculating number of words perturbed:', e)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

# Example usage:
# log_results_extension(test_folder, name_of_test, results)



from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_results(results, args):
    """
    Evaluates the attack results, computes confusion matrices, accuracies, and attack success rate.

    Parameters:
    - results (list): The list of attack results.
    - args (object): An object that contains the number of classes (`args.n_classes`).
    """
    original_result_true_labels = []
    original_result_predictions = []

    result_true_labels = []
    result_predictions = []

    for i, result in enumerate(results):
        # here we need to distinguish between the result.skipped, successful and failed
        original_result_true_labels.append(result.original_result.ground_truth_output)
        original_result_predictions.append(result.original_result.output)
        result_true_labels.append(result.perturbed_result.ground_truth_output)
        result_predictions.append(result.perturbed_result.output)

    # Output the confusion matrix for original results
    print('Original Result True Labels:', original_result_true_labels)
    print('Original Result Predictions:', original_result_predictions)

    orig_conf_matrix = confusion_matrix(original_result_true_labels, original_result_predictions, labels=list(range(args.n_classes+1)))
    print("Original Confusion Matrix:")
    print(orig_conf_matrix)
    orig_acc = accuracy_score(original_result_true_labels, original_result_predictions) * 100.0
    print("Original Accuracy:", orig_acc)

    null_class = args.n_classes
    # Filter out null class instances for original results
    orig_filtered_predictions = [pred for pred in original_result_predictions if pred != null_class]
    orig_filtered_true_labels = [true_label for true_label, pred in zip(original_result_true_labels, original_result_predictions) if pred != null_class]

    print('Filtered Original Result True Labels:', orig_filtered_true_labels)
    print('Filtered Original Result Predictions:', orig_filtered_predictions)

    # Calculate the accuracy ignoring the null class for original results
    orig_filtered_accuracy = accuracy_score(orig_filtered_true_labels, orig_filtered_predictions) * 100.0
    print(f'Filtered Original Accuracy: {orig_filtered_accuracy:.4f}')

    # Output the confusion matrix for perturbed results
    conf_matrix = confusion_matrix(result_true_labels, result_predictions, labels=list(range(args.n_classes+1)))
    print("Confusion Matrix:")
    print(conf_matrix)
    acc = accuracy_score(result_true_labels, result_predictions) * 100.0
    print("Accuracy:", acc)

    print('Perturbed Result True Labels:', result_true_labels)
    print('Perturbed Result Predictions:', result_predictions)

    # Filter out null class instances for perturbed results
    filtered_predictions = [pred for pred in result_predictions if pred != null_class]
    filtered_true_labels = [true_label for true_label, pred in zip(result_true_labels, result_predictions) if pred != null_class]

    print('Filtered Perturbed Result True Labels:', filtered_true_labels)
    print('Filtered Perturbed Result Predictions:', filtered_predictions)

    # Calculate the accuracy ignoring the null class for perturbed results
    filtered_accuracy = accuracy_score(filtered_true_labels, filtered_predictions) * 100.0
    print(f'Filtered Perturbed Accuracy: {filtered_accuracy:.4f}')

    # successful_attacks_and_original_skipped = [true_label for true_label, pred in zip(filtered_true_labels, filtered_predictions) if true_label != pred] 
    # successful_attacks_and_original_skipped = filter(orginal_skipped, so original missclassifications, so we need the ground truth)
    # filtered_attack_success_rate = (len(successful_attacks) * 100.0) / len(filtered_predictions)
    # print('Filtered Attack Success Rate:', filtered_attack_success_rate)

    successful_samples_queries = count_successful_samples_quereis(results)
    print('Successful Samples Queries:', successful_samples_queries)
    args.logging.info(f"Successful Samples Queries: {successful_samples_queries}")

def count_successful_samples_quereis(results):
    total_queries = 0
    succ_samples = 0
    for i, result in enumerate(results):
        if isinstance(result, SuccessfulAttackResult):
            total_queries += result.num_queries
            succ_samples += 1
    return total_queries / succ_samples if succ_samples != 0 else 0

# Example usage:
# evaluate_results(results, args)