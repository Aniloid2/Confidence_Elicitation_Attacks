from src.utils.shared.misc import environment_setup
import os
from src.inference import Step2KPredAvg
args  = environment_setup() 
from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP,TASK_N_CLASSES,MODEL_INFO  
import numpy as np
args.n_classes =  TASK_N_CLASSES[args.task] 
args.confidence_type_dict = CONFIDENCE_LEVELS[args.confidence_type] 
args.confidence_map_dict = CONFIDENCE_MAP[args.confidence_type] 
model_info = MODEL_INFO[args.model_type]
 
args.model_name =  model_info['model_name']
args.start_prompt_header = model_info['start_prompt_header']
args.end_prompt_footer = model_info['end_prompt_footer']


from src.utils.shared.misc import initialize_model_and_tokenizer

args = initialize_model_and_tokenizer(args)


from src.utils.shared import load_data
dataset_class, label_names = load_data(args)
from src.utils.shared import SimpleDataset
dataset_class =  SimpleDataset(dataset_class,label_names = label_names ) 
args.dataset = dataset_class

import pickle
name_log = 'results'
name_of_test = f'EN{str(args.num_examples)}_MT{args.model_type}_TA{args.task}_PT{args.prompting_type}_PST{args.prompt_shot_type}_ST{args.similarity_technique}_NT{args.num_transformations}'
print ('name_of_test',name_of_test)
file_path = os.path.join(args.test_folder, f'{name_log}_{name_of_test}.pkl')
print ('file_path')


print(f'Results saved to {file_path}')

# To load the `results` list from the file
with open(file_path, 'rb') as file:
    loaded_results = pickle.load(file)

args.k_pred = 20
classifiers = [Step2KPredAvg(**vars(args))]
predictions = []
probabilities = []
correctness=[]
confidences=[]
smaller_true_labels = []
counter_null = 0

# probs_list = []
# second_order_uncertainties = []
# empirical_means = []
for predictor in classifiers: 
    variance_of_means_across_results = []
    smaller_samples = loaded_results[:10]+loaded_results[-10:]
    print ('sub samples',len(smaller_samples))
    for iter, result in enumerate(smaller_samples):
        probs_list = []
        second_order_uncertainties = []
        empirical_means = []
        
        original_text = result.original_result.attacked_text
        perturbed_text = result.perturbed_result.attacked_text
        true_label = result.perturbed_result.ground_truth_output
        print ('original_text',original_text)
        print ('perturbed_text',perturbed_text, true_label)
        for i in range(10):
            guess, probs, confidence = predictor.predict_and_confidence(datapoint = (original_text,true_label))
            print (f'guess{i}',guess)
            print (f'probs{i}',probs)
            print (f'confidences{i}',confidence) 
            probs_list.append(probs)
            second_order_uncertainties.append(predictor.second_order_uncertainty)
            empirical_means.append(predictor.empirical_mean)
        
        # Convert to numpy array for easier manipulation
        second_order_uncertainties = np.array(second_order_uncertainties) 
        empirical_means = np.array(empirical_means) 
        
        # Calculate statistics across calls
        mean_of_means = np.mean(empirical_means, axis=0)
        variance_of_means = np.std(empirical_means, axis=0)
        mean_of_variances = np.mean(second_order_uncertainties, axis=0)
        variance_of_variances = np.std(second_order_uncertainties, axis=0)

        third_order_uncertainties = np.std(second_order_uncertainties, axis=0) # Calculate std.dev. as a measure

        print("Mean of empirical means:", mean_of_means)
        print("Variance of empirical means:", variance_of_means)
        print("Mean of second-order uncertainties:", mean_of_variances)
        print("Variance of second-order uncertainties:", variance_of_variances)
        print('third_order_uncertainties:', third_order_uncertainties)

        variance_of_means_across_results.append(variance_of_means)
        if guess == 'null':
            counter_null+=1
        #     continue
        # prediction_label = 0 if guess == 'negative' elif guess = 'positive' 1 else 2
    
        prediction_label = predictor.prompt_class.task_name_to_label[guess]

        if args.task == 'popQA':
            true_label=1
        elif args.task =='strategyQA': 
            true_label = int(true_label) 

        predictions.append(prediction_label)
        # Take the probability of the positive label for calibration curve
        # probabilities.append(probs[1])
        probabilities.append(probs)

        correctness.append(prediction_label)
        confidences.append(confidence)
        smaller_true_labels.append(true_label)
    variance_of_means_across_results = np.array(variance_of_means_across_results) 
    variance_across_results = np.std(variance_of_means_across_results, axis=0)
    print ('variance_across_results',variance_across_results)

    true_labels = np.array(smaller_true_labels)



    from sklearn.metrics import confusion_matrix, accuracy_score
    conf_matrix = confusion_matrix(true_labels, predictions, labels=list(range(args.n_classes+1)))


    # Output the confusion matrix
    print("Confusion Matrix:", predictor.technique_name)
    print(conf_matrix)
    acc = accuracy_score(true_labels, predictions)
    print("Accuracy:", acc, 'counter_null:',counter_null)

    null_class = args.n_classes
    # Filter out null class instances
    # filtered_true_labels = [label for label in true_labels if label != null_class]
    # filtered_predictions = [pred for true_label, pred in zip(true_labels, predictions) if true_label != null_class]

    filtered_predictions = [pred for pred in predictions if pred != null_class]
    filtered_true_labels = [true_label for true_label, pred in zip(true_labels, predictions) if pred != null_class]
    filtered_probabilities = [probability for probability, pred in zip(probabilities, predictions) if pred != null_class]
    filtered_probabilities = np.array(filtered_probabilities)
    filtered_true_labels =  np.array(filtered_true_labels)
    filtered_probabilities = np.array(filtered_probabilities)
    # Calculate the accuracy ignoring the null class
    filtered_accuracy = accuracy_score(filtered_true_labels, filtered_predictions)

    print(f'Filtered Accuracy: {filtered_accuracy:.4f}')

    probabilities = np.array(probabilities)
    # print ('probabilities',probabilities) 
    correctness = np.array([1 if true_labels[i] == np.argmax(probabilities[i]) else 0 for i in range(len(true_labels))])
    confidences = np.max(probabilities, axis=1)



    name_plot_calibration = f'{predictor.technique_name}_baseline_probs'
    # print ('Base Calibration Metrics')
    # from src.utils.shared import calculate_roc_metrics, plot_roc_curve,  plot_calibration_curve, calculate_ece_for_all_classes
    # calculate_roc_metrics(args, true_labels, probabilities)
    # plot_roc_curve(args, true_labels, probabilities, name_plot = name_plot_calibration)
    # calculate_ece_for_all_classes(args, true_labels, probabilities) 
    # plot_calibration_curve(args, true_labels, probabilities, name_plot =name_plot_calibration) 

    

    # name_plot_roc = f'{predictor.technique_name}_filtered_probs'
    # print ('Filtered Calibration Metrics')
    # calculate_roc_metrics(args, filtered_true_labels, filtered_probabilities)
    # plot_roc_curve(args, filtered_true_labels, filtered_probabilities, name_plot = name_plot_roc)
    # calculate_ece_for_all_classes(args, filtered_true_labels, filtered_probabilities) 
    # plot_calibration_curve(args, filtered_true_labels, filtered_probabilities, name_plot = name_plot_roc) 