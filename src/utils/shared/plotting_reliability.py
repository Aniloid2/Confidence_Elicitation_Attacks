import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize


def plot_calibration_curve(args, true_labels, probabilities, name_plot):
    """
    This function plots the calibration curve for classification models.

    Parameters:
    - true_labels: The ground truth labels.
    - probabilities: The predicted probabilities for each class.
    - name_plot: The name to replace "calibration" in the saved plot filenames.
    - args: An argparse Namespace object that contains various required parameters such as:
        - n_classes: The number of classes.
        - dataset: An object containing label names.
        - task: The classification task.
        - model_type: The model type.
        - confidence_type: The confidence type.
        - test_folder: Folder to save the plots.
        - prompting_type: Prompting type used in the model.
        - k_pred: A parameter used for naming the saved plots.
    """
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(15, 15))

    if args.n_classes > 2:
        true_labels_binarized = label_binarize(true_labels, classes=range(args.n_classes))

        for i in range(args.n_classes):
            true_prob, pred_prob = calibration_curve(true_labels_binarized[:, i], probabilities[:, i], n_bins=10)
            plt.plot(pred_prob, true_prob, marker='o', linewidth=1, label=f'{args.dataset.label_names[i]}')

        # plt.title(
        #     f'Calibration Plot for \n Multi-Class Classification {args.task}/{args.model_type} \n Confidence Type: {args.confidence_type}', 
        #     fontsize=45,
        #     pad=25
        # )
        plt.title(
            f'Calibration Plot for \n Classification {args.task}/{args.model_type}', 
            fontsize=45,
            pad=25
        )
        plt.ylabel('Fraction of correct classifications', fontsize=35)
    else:
        true_prob, pred_prob = calibration_curve(true_labels, probabilities[:, 1], n_bins=10)
        plt.plot(pred_prob, true_prob, marker='o', linewidth=1, label='Calibration Plot')
        # plt.title(
        #     f'Calibration Plot for \n Binary Classification {args.task}/{args.model_type} \n Confidence Type: {args.confidence_type}', 
        #     fontsize=45,
        #     pad=25
        # )
        plt.title(
            f'Calibration Plot for \n Classification {args.task}/{args.model_type}', 
            fontsize=45,
            pad=25
        )
        plt.ylabel('Fraction of positives', fontsize=35)

    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally\ncalibrated')
    plt.xlabel('Mean predicted probability', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(prop={'size': 35})

    # Save the plot to a file
    plt.savefig(os.path.join(args.test_folder, f'{name_plot}_llm_{args.model_type}_{args.task}_{args.prompting_type}_{args.confidence_type}_K{args.k_pred}.jpg'), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.test_folder, f'{name_plot}_llm_{args.model_type}_{args.task}_{args.prompting_type}_{args.confidence_type}_K{args.k_pred}.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_ece(true_labels, probabilities, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    - true_labels: The ground truth labels (binary or one-hot encoded).
    - probabilities: The predicted probabilities.
    - n_bins: Number of bins to use for calibration curve.

    Returns:
    - ECE: A float, the calculated Expected Calibration Error.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (probabilities > bin_edges[i]) & (probabilities <= bin_edges[i + 1])
        if bin_mask.any():
            bin_accuracy = np.mean(true_labels[bin_mask])
            bin_confidence = np.mean(probabilities[bin_mask])
            bin_weight = np.sum(bin_mask) / len(probabilities)
            ece += bin_weight * np.abs(bin_confidence - bin_accuracy)
    return ece


def calculate_ece_for_all_classes(args,true_labels, probabilities):
    """
    Calculate the Expected Calibration Error (ECE) for each class and print the average.

    Parameters:
    - true_labels: The ground truth labels.
    - probabilities: The predicted probabilities for each class.
    - args: An argparse Namespace object that contains various required parameters such as:
        - n_classes: The number of classes.
    """
    ece_total = 0
    for i in range(args.n_classes):
        ece = calculate_ece(true_labels == i, probabilities[:, i])
        ece_total += ece
        print(f"Expected Calibration Error (ECE) for class {i}: {ece:.4f}")
        args.logging.info(f"Expected Calibration Error (ECE) for class {i}: {ece:.4f}")

    # Print average ECE
    average_ece = ece_total / args.n_classes
    print(f"Average Expected Calibration Error (ECE): {average_ece:.4f}")
    args.logging.info(f"Average Expected Calibration Error (ECE): {average_ece:.4f}")
    # Pearson Correlation Coefficient
    correctness = (true_labels == np.argmax(probabilities, axis=1)).astype(np.float32)
    confidences = np.max(probabilities, axis=1)
    correlation_coefficient = np.corrcoef(correctness, confidences)[0, 1]
    print(f"Pearson Correlation Coefficient (r): {correlation_coefficient:.4f}")
    args.logging.info(f"Pearson Correlation Coefficient (r): {correlation_coefficient:.4f}")