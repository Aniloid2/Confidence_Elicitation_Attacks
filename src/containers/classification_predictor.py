
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