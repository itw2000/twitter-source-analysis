import numpy as np
import random
import math

random.seed(0)
#Accepts inputs as numpy arrays
def perceptron(x, y):

    #Reshape data and add bias term
    n_samples, n_features = x.shape
    node_zero = [1] * n_samples
    node_zero = np.array(node_zero).reshape(n_samples, 1)
    x = np.hstack((node_zero, x))

    #Parameters
    est_y = []
    alpha = 0.1 / (1 - 0.5)
    threshold = 0.5

    #Initalize weights
    w = ([0] * (n_features+1))
    w = np.array(w).astype(float)



    for j in range(0, n_samples):
        #Predict the class
        est_y.append((sum(w * x[j, :])) > threshold)

        #Compute the correction to weights and update weights
        correction = np.array(alpha * (y[j] - est_y[j]) * x[j, :])
        w = w + correction

    return est_y

def logistic(x):
    return 1 / (1 + math.exp(-x))

class MultiPerceptron:

    def __init__(self, nHidden, nOutput):
        #Declare number of layers
        self.nHidden = nHidden
        self.nOutput = nOutput

        self.weight_matrix_output = np.random.random((self.nOutput, self.nHidden))


    def feed_features(self, features, target, weight_matrix_hidden, weight_matrix_output):
        hidden_activations = []
        threshold = 0.75

        for i in range(self.nHidden):
            x = features.dot(weight_matrix_hidden[:, i])
            hidden_activations.append(logistic(x))

        hidden_activations = np.array([hidden_activations])
        y = hidden_activations.dot(weight_matrix_output[0, :])
        label = (logistic(y) > threshold)
        output_activation = logistic(y)
        print label
        error = label - float(target)

        return hidden_activations, error, output_activation, label

    def compute_delta(self, hidden_activations, error, output_activation, n_features, weight_matrix_output, features):
        alpha = 0.1
        hidden_weight_delta = []
        delta_output = output_activation * (1 - output_activation) * error
        output_weight_delta = -alpha * delta_output * hidden_activations

        delta_hidden = hidden_activations * (1 - hidden_activations) * np.sum(delta_output * weight_matrix_output)

        for i in range(n_features):
            delta_hidden_weight = -alpha * delta_hidden * features[i]
            hidden_weight_delta.append(delta_hidden_weight)
        hidden_weight_delta = np.vstack((hidden_weight_delta[0], hidden_weight_delta[1], hidden_weight_delta[2], hidden_weight_delta[3], hidden_weight_delta[4],
        hidden_weight_delta[5]))

        return output_weight_delta, hidden_weight_delta

    def train(self, features, targets):

        n_samples, n_features = features.shape

        weight_matrix_hidden = np.random.random((n_features, self.nHidden))
        weight_matrix_output = self.weight_matrix_output

        for i in range(1):
            weights_output = np.zeros((1,self.nHidden))
            weights_hidden = np.zeros((6,self.nHidden))

            for j in range(n_samples):
                    fitting = self.feed_features(features[j, :], targets[j], weight_matrix_hidden, weight_matrix_output)

                    hidden_activations = fitting[0]
                    error = fitting[1]
                    output_activations = fitting[2]

                    augment = self.compute_delta(hidden_activations, error, output_activations, n_features, weight_matrix_hidden, features[0,:])

                    weights_output = weights_output + augment[0]
                    weights_hidden = weights_hidden + augment[1]

            weight_matrix_hidden = weight_matrix_hidden + weights_hidden
            weight_matrix_output = weight_matrix_output + weights_output


        return weight_matrix_hidden, weight_matrix_output
























































