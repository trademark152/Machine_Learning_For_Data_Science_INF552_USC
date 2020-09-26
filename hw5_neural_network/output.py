import numpy as np

"""
softmax: providing input and output
see: https://en.wikipedia.org/wiki/Softmax_function
"""
class Softmax:
    # standard softmax function
    def eval(self, input):
        # normalized exponential sum
        exp_scores = np.exp(input)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # calculate the average correct log probabilities
    def eval_error(self, input, true_output):
        num_examples = input.shape[0]

        # convert to probability
        probs = self.eval(input)

        # get the log probability cprressponding to the correct class
        correct_logprobs = -np.log(probs[range(num_examples), true_output])

        # return the average cross-entropy probs, ideal is to maximize this correct_logprobs,
        # meaning the classification classifies true ouput 1 to 1, true output 0 to 0
        return 1./num_examples * np.sum(correct_logprobs)

    # derivative of cross entropy loss
    def calc_diff(self, input, true_output):
        num_examples = input.shape[0]

        # get exponential normalized probs
        probs = self.eval(input)

        # compare to true output: this is error to class 1 (dL_derr = yi - ti)
        # error to class 0 is by -= 0 which stays the same
        probs[range(num_examples), true_output] -= 1
        return probs

"""
Standard square error
"""
class LSE:
    def eval(self, input):
        return (np.greater(input, 0.5))*1
        # return np.argmax(input, axis=1)

    # loss function and its derivative
    def eval_error(self, input, y_true):
        return np.mean(np.power(np.subtract(input,y_true.reshape(-1, 1)), 2))

    # derivative of square error loss
    def calc_diff(self, input, y_true):
        num_examples = input.shape[0]
        return 2/num_examples * np.subtract(input, y_true.reshape(-1, 1))