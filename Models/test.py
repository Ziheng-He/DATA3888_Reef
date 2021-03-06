# https://goddoe.github.io/r/machine%20learning/2017/12/17/how-to-use-r-model-in-python.html

import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

r = robjects.r
numpy2ri.activate()

class Model(object):
    """
    R Model Loader

    Attributes
    ----------
    model : R object
    """

    def __init__(self):
        self.model = None

    def load(self, path):
        model_rds_path = "{}.rds".format(path)
        model_dep_path = "{}.dep".format(path)

        self.model = r.readRDS(model_rds_path)

        with open(model_dep_path, "rt") as f:
            model_dep_list = [importr(dep.strip())
                              for dep in f.readlines()
                              if dep.strip()!='']

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        pred_probs : array, shape (n_samples, probs)
        """

        if self.model is None:
            raise Exception("There is no Model")

        if type(X) is not np.ndarray:
            X = np.array(X)

        pred = r.predict(self.model, X, probability=True)
        probs = r.attr(pred, "probabilities")

        return np.array(probs)


from rpy2_wrapper.model import Model

# Constants
MODEL_PATH = "./logistic"

# Example Input
# X = np.array([[5.1, 3.5,  1.4, 0.2], # setosa
#               [6.1, 2.6,  5.6, 1.4]]  # virginica

# Example Run
model = Model().load(MODEL_PATH)
# pred = model.predict(X)

# Example output
# print(pred)
