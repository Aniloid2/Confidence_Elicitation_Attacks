import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def plot_roc_curve(args,true_labels, probabilities, name_plot):
    """
    This function plots the ROC curve for classification models.

    Parameters:
    - true_labels: The ground truth labels.
    - probabilities: The predicted probabilities for each class.
    - name_plot: The name to replace "roc" in the saved plot filenames.
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
    plt.figure(figsize=(15, 15))
    print(f"true_labels shape: {true_labels.shape}")
    print(f"probabilities shape: {probabilities.shape}")

    if args.n_classes > 2:
        true_labels_binarized = label_binarize(true_labels, classes=range(args.n_classes))
        for i in range(args.n_classes):
            fpr, tpr, _ = roc_curve(true_labels_binarized[:, i], probabilities[:, i])
            plt.plot(fpr, tpr, marker='o', linewidth=1, label=f'{args.dataset.label_names[i]}')
        auroc = roc_auc_score(true_labels_binarized, probabilities[:, :args.n_classes], average='macro', multi_class='ovr')
    else:
        fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
        plt.plot(fpr, tpr, marker='o', linewidth=1, label='ROC Curve')
        auroc = roc_auc_score(true_labels, probabilities[:, 1])

    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Chance')
    # plt.title(f'ROC Curve for \n Classification {args.task}/{args.model_type} \n Confidence Type:{args.confidence_type}', fontsize=45, pad=25)
    plt.title(f'ROC Curve for \n Classification {args.task}/{args.model_type}', fontsize=45, pad=25)
    
    plt.xlabel('False Positive Rate', fontsize=35)
    plt.ylabel('True Positive Rate', fontsize=35)
    plt.legend(prop={'size': 35})
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    # Save the ROC plot to a file with the name_plot variable
    jpg_filename = f'{name_plot}_roc_llm_{args.model_type}_{args.task}_{args.prompting_type}_{args.confidence_type}_K{args.k_pred}.jpg'
    pdf_filename = f'{name_plot}_roc_llm_{args.model_type}_{args.task}_{args.prompting_type}_{args.confidence_type}_K{args.k_pred}.pdf'
    plt.savefig(os.path.join(args.test_folder, jpg_filename), format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.test_folder, pdf_filename), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"AUROC: {auroc:.4f}")
    args.logging.info(f'AUROC: {auroc:.4f}')


def calculate_roc_metrics(args,true_labels, probabilities):
    """
    This function calculates various metrics for classification models, such as AUROC and AUPRC.

    Parameters:
    - true_labels: The ground truth labels.
    - probabilities: The predicted probabilities for each class.
    - args: An argparse Namespace object that contains various required parameters such as:
        - n_classes: The number of classes.
    """
    if args.n_classes > 2:
        true_labels_binarized = label_binarize(true_labels, classes=range(args.n_classes))
        auroc = roc_auc_score(true_labels_binarized, probabilities[:, :args.n_classes], average='macro', multi_class='ovr')
        auprcs_positive = []
        for i in range(args.n_classes):
            auprc_pos = average_precision_score(true_labels_binarized[:, i], probabilities[:, i])
            auprcs_positive.append(auprc_pos)
            print(f"AUPRC for class {i} (positive class): {auprc_pos:.4f}")
            args.logging.info(f"AUPRC for class {i} (positive class): {auprc_pos:.4f}")
    else:
        auroc = roc_auc_score(true_labels, probabilities[:, 1])
        auprc_positive = average_precision_score(true_labels, probabilities[:, 1])
        print(f"AUPRC-Positive: {auprc_positive:.4f}")
        args.logging.info(f"AUPRC-Positive: {auprc_positive:.4f}")
        true_labels_negative = 1 - true_labels
        probabilities_negative = 1 - probabilities[:, 1]
        auprc_negative = average_precision_score(true_labels_negative, probabilities_negative)
        print(f"AUPRC-Negative: {auprc_negative:.4f}")
        args.logging.info(f"AUPRC-Negative: {auprc_negative:.4f}")
    print(f"AUROC: {auroc:.4f}")
    args.logging.info(f'AUROC: {auroc:.4f}')
