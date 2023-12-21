import warnings

from models.encoder import Encoder


class Model:
    def __init__(self, model=None) -> None:
        self.encoder = Encoder()
        self.model = model

    def get_features_dict(self):
        return self.encoder.get_features_dict()

    def predict(self, X, class_names=True):
        if self.model is None:
            warnings.warn("Model is not defined! Load the model first")
            return

        if isinstance(X, dict):
            encoded_X = self.encoder.encode_one_record(X).reshape((1, -1))
        elif isinstance(X, list):
            encoded_X = self.encoder.encode_records(X)
        else:
            raise ValueError("X must be list or dict")

        try:
            y = self.model.predict(encoded_X)
        except AttributeError:
            raise AttributeError("Model must have predict method")

        if class_names:
            y = [self.encoder.outcome_mapping[i] for i in y]

        return y
