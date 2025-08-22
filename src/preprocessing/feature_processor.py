import numpy as np
import logging


class FeatureProcessor():
    """
    Class that contains feature-specific processing functions.
    """

    def __init__(self):
        self.dispatch = {
            "identity": self.identity,
            "flatten": self.flatten,
        }

    def process(self, value, method: str):
        fn = self.dispatch.get(method)
        if fn is None:
            raise Exception(f"Unknown processing method '{method}'.")
        return fn(value)


    def identity(self, x):
        if x is None:
            return [0.0]
        print(f"Identity processing: {x}, shape: {np.array(x).shape}")  # Debug line
        return [float(x)]


    def flatten(self, x):
        if x is None:
            return [0.0]
        return list(np.array(x).flatten())