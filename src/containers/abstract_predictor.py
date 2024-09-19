from abc import ABC, abstractmethod

class AbstractPredictor(ABC):
    # Assumes the subclasses will have these attributes
    base_results = None
    classifier_results = None
    
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