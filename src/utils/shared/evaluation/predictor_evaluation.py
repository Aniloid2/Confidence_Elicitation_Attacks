from src.containers import BasePredictorResults
import numpy as np
def predictor_evaluation(args,predictor):
    true_labels = predictor.predictor_container.base_results.true_labels
    probabilities = predictor.predictor_container.base_results.probabilities
    confidences = predictor.predictor_container.base_results.confidences
    print ('true_labels',true_labels)
    print ('probabilities',probabilities)
    print ('confidences',confidences)

    print ('top_k_max_prediction',predictor.predictor_container.classifier_results.top_k_max_prediction)
    print ('top_k_max_prediction_and_confidence',predictor.predictor_container.classifier_results.top_k_max_prediction_and_confidence)
    print ('top_k_dirichlet_mean',predictor.predictor_container.classifier_results.top_k_dirichlet_mean)

    for _, predictor_attribute in vars(predictor.predictor_container).items():
        if not isinstance(predictor_attribute, BasePredictorResults):
            for predictor_name, predictor_predictions in vars(predictor_attribute).items():
                if predictor_predictions:  # Check if the list is not empty
                    # Perform operation if the list is not empty
                    print(f"Performing operation on {predictor_name}:")
                    smaller_true_labels = true_labels
                    predictions = predictor_predictions

                    true_labels = np.array(smaller_true_labels)



                    from sklearn.metrics import confusion_matrix, accuracy_score
                    conf_matrix = confusion_matrix(true_labels, predictions, labels=list(range(args.n_classes+1)))


                    # Output the confusion matrix
                    print("Confusion Matrix:", predictor_name)
                    args.logging.info(f"Confusion Matrix: {predictor_name}")
                    
                    print(conf_matrix)
                    args.logging.info(f"{conf_matrix}")
                    acc = accuracy_score(true_labels, predictions)
                    print(f"Accuracy: {predictor_name}", acc)
                    args.logging.info(f"Accuracy: {predictor_name} {acc}")

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
                    args.logging.info(f'Filtered Accuracy {predictor_name}: {filtered_accuracy:.4f}')

                    probabilities = np.array(probabilities)
                    # print ('probabilities',probabilities) 
                    correctness = np.array([1 if true_labels[i] == np.argmax(probabilities[i]) else 0 for i in range(len(true_labels))])
                    confidences = np.max(probabilities, axis=1)



                    name_plot_calibration = f'{predictor_name}_baseline_probs'
                    print ('Base Calibration Metrics')
                    args.logging.info(f'Base Calibration Metrics {predictor_name}')
                    from src.utils.shared import calculate_roc_metrics, plot_roc_curve,  plot_calibration_curve, calculate_ece_for_all_classes
                    
                    calculate_roc_metrics(args, true_labels, probabilities)
                    plot_roc_curve(args, true_labels, probabilities, name_plot = name_plot_calibration)
                    
                    calculate_ece_for_all_classes(args, true_labels, probabilities) 
                    plot_calibration_curve(args, true_labels, probabilities, name_plot =name_plot_calibration) 

                    

                    name_plot_roc = f'{predictor_name}_filtered_probs'
                    print ('Filtered Calibration Metrics')
                    args.logging.info(f'Filtered Calibration Metrics {predictor_name}')
                    
                    calculate_roc_metrics(args, filtered_true_labels, filtered_probabilities)
                    plot_roc_curve(args, filtered_true_labels, filtered_probabilities, name_plot = name_plot_roc)
                    
                    calculate_ece_for_all_classes(args, filtered_true_labels, filtered_probabilities) 
                    plot_calibration_curve(args, filtered_true_labels, filtered_probabilities, name_plot = name_plot_roc) 





                else:
                    # Skip operation if the list is empty
                    print(f"Skipping {predictor_name} because no results.")


 