class BasePredictor:
    def predict_and_confidence(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")