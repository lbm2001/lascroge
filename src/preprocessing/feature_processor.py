import numpy as np


class FeatureProcessor:
    """
    Class that contains feature-specific processing functions.
    """

    def __init__(self):
        self.dispatch = {
            "identity": self.identity,
            "flatten": self.flatten,
        }

    def process(self, value, method: str, inv=False):
        fn = self.dispatch.get(method)
        if fn is None:
            raise Exception(f"Unknown processing method '{method}'.")
        return fn(x=value, inv=inv)

    def identity(self, x, inv):
        if inv:
            return str(x.item())
        else:
            if x is None:
                return [0.0]
            return [float(x)]

    def flatten(self, x, inv):

        if inv:
            return " ".join(map(str, x))
        else:
            if x is None:
                return [0.0]
            return list(np.array(x).flatten())
