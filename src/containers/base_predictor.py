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
