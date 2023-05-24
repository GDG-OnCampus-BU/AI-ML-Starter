import numpy as np

"""  
Losgistic Regression is Nothing but
creating probability distribution insted of specific values
So that we can clearly classify between two classes


Initailizing Weights and Bias as ZERO 
and with the Iteration changing them
"""


class LogisticRegression:
    def __init__(self, learning_rate=0.001, iter_count=1000):
        """
        Para
        learning_rate = Learning Rate
        iter_count = Number of Iterations (default value = 1000)
        """
        self.learning_rate = learning_rate
        self.iter_count = iter_count
        self.weights = None
        self.bias = None

    def my_sigmoid(z):
        """
        y^ = 1 / (1 + EXP(-z))
        # sigmoid function for variable  (z)
        """
        result = 1 / (1 + np.exp(-z))
        return result

    def fit(self, X, y):
        sample_count, feature_count = X.shape

        self.weights = np.zeros(self.features)
        self.bias = 0

        # z = w.x+b
        # y^ = 1 / (1 + EXP(-z))
        # sigmoid function

        for i in range(self.iter_count):
            linear_prediction = np.dot(X, self.weights) + self.bias
            prediction = LogisticRegression.my_sigmoid(linear_prediction)

            # Calculating Error

            # gradient for weights and bias
            dw = (1 / sample_count) * np.dot(X.T, (prediction - y))
            db = (1 / sample_count) * np.sum(prediction - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict_proba(self, X_test):
        linear_prediction = np.dot(X_test, self.weights) + self.bias
        y_prediction = LogisticRegression.my_sigmoid(linear_prediction)

        return y_prediction

    def predict(self, X_test):
        class_prediction = [0 if y <= 0.5 else 1 for y in self.predict_proba(X_test)]

        return class_prediction
    
    def accuracy(y_pred , y_test):
        
        return np.sum(y_pred == y_test) / len(y_test)
