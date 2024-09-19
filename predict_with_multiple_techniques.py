from src.utils.shared.misc import environment_setup
import os
from src.inference import Step2KPredAvg
import numpy as np
args  = environment_setup() 
from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP,TASK_N_CLASSES,MODEL_INFO  

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


class BasePredictorResults:
    def __init__(self):
        self.true_labels = []
        self.probabilities = []
        self.confidences = []

    def add_true_label(self, label):
        self.true_labels.append(label)

    def add_probability(self, probability):
        self.probabilities.append(probability)

    def add_confidence(self, confidence):
        self.confidences.append(confidence)


class ClassifierPredictorResults:
    def __init__(self):
        self.top_k_max_prediction = []
        self.top_k_max_prediction_and_confidence = []
        self.top_k_dirichlet_mean = []
        self.vanilla_prediction = []
        self.vanilla_prediction_and_confidence = []
        self.cot_prediction = []
        self.cot_prediction_and_confidence = []

    def add_top_k_max_prediction(self, prediction):
        self.top_k_max_prediction.append(prediction)

    def add_top_k_max_prediction_and_confidence(self, result):
        self.top_k_max_prediction_and_confidence.append(result)

    def add_top_k_dirichlet_mean(self, mean):
        self.top_k_dirichlet_mean.append(mean)

    def add_vanilla_prediction(self, prediction):
        self.vanilla_prediction.append(prediction)

    def add_vanilla_prediction_and_confidence(self, result):
        self.vanilla_prediction_and_confidence.append(result)

    def add_cot_prediction(self, prediction):
        self.cot_prediction.append(prediction)

    def add_cot_prediction_and_confidence(self, result):
        self.cot_prediction_and_confidence.append(result)


class PredictionContainer:
    def __init__(self):
        self.base_results = BasePredictorResults()
        self.classifier_results = ClassifierPredictorResults()

    def add_true_label(self, label):
        self.base_results.add_true_label(label)

    def add_probability(self, probability):
        self.base_results.add_probability(probability)

    def add_confidence(self, confidence):
        self.base_results.add_confidence(confidence)

    def add_top_k_max_prediction(self, prediction):
        self.classifier_results.add_top_k_max_prediction(prediction)

    def add_top_k_max_prediction_and_confidence(self, result):
        self.classifier_results.add_top_k_max_prediction_and_confidence(result)

    def add_top_k_dirichlet_mean(self, mean):
        self.classifier_results.add_top_k_dirichlet_mean(mean)

    def add_vanilla_prediction(self, prediction):
        self.classifier_results.add_vanilla_prediction(prediction)

    def add_vanilla_prediction_and_confidence(self, result):
        self.classifier_results.add_vanilla_prediction_and_confidence(result)

    def add_cot_prediction(self, prediction):
        self.classifier_results.add_cot_prediction(prediction)

    def add_cot_prediction_and_confidence(self, result):
        self.classifier_results.add_cot_prediction_and_confidence(result)

 



predictor1 =Step2KPredAvg(**vars(args))
predictor1.predictor_container = PredictionContainer()

predictors = [predictor1]
 



for predictor in predictors:
    predictions = []
    probabilities = []
    correctness=[]
    confidences=[]
    smaller_true_labels = []
    counter_null = 0
    smaller_samples = loaded_results # loaded_results[:10]+loaded_results[-10:]
    # smaller_samples =  loaded_results[:4]+loaded_results[-4:]
    print ('sub samples',len(smaller_samples))
    for iter, result in enumerate(smaller_samples): 
        original_text = result.original_result.attacked_text
        perturbed_text = result.perturbed_result.attacked_text
        true_label = result.perturbed_result.ground_truth_output
        print ('original_text',original_text)
        print ('perturbed_text',perturbed_text, true_label)
        guess, probs, confidence = predictor.predict_and_confidence(datapoint = (perturbed_text,true_label))
        
        print ('guess',guess)
        print ('probs',probs)
        print ('confidences',confidence)
        if guess == 'null':
            counter_null+=1
        #     continue
        # prediction_label = 0 if guess == 'negative' elif guess = 'positive' 1 else 2
        
    
        # predictor.vanilla_prediction = None
        # predictor.vanilla_prediction_and_confidence = None
        # predictor.cot_prediction = None
        # predictor.cot_prediction_and_confidence = None
        predictor.predictor_container.add_true_label(true_label)
        predictor.predictor_container.add_probability(probs)
        predictor.predictor_container.add_confidence(confidence)
    
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



                    name_plot_calibration = f'{predictor_name}_baseline_probs'
                    print ('Base Calibration Metrics')
                    from src.utils.shared import calculate_roc_metrics, plot_roc_curve,  plot_calibration_curve, calculate_ece_for_all_classes
                    calculate_roc_metrics(args, true_labels, probabilities)
                    plot_roc_curve(args, true_labels, probabilities, name_plot = name_plot_calibration)
                    calculate_ece_for_all_classes(args, true_labels, probabilities) 
                    plot_calibration_curve(args, true_labels, probabilities, name_plot =name_plot_calibration) 

                    

                    name_plot_roc = f'{predictor_name}_filtered_probs'
                    print ('Filtered Calibration Metrics')
                    calculate_roc_metrics(args, filtered_true_labels, filtered_probabilities)
                    plot_roc_curve(args, filtered_true_labels, filtered_probabilities, name_plot = name_plot_roc)
                    calculate_ece_for_all_classes(args, filtered_true_labels, filtered_probabilities) 
                    plot_calibration_curve(args, filtered_true_labels, filtered_probabilities, name_plot = name_plot_roc) 





                else:
                    # Skip operation if the list is empty
                    print(f"Skipping {predictor_name} because no results.")
 