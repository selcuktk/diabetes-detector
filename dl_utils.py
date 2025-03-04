import numpy as np

class activation:
    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(aL):
        return aL * (1 - aL)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(aL, alpha=0.01):
        daL = np.ones_like(aL)    
        daL[aL < 0] = alpha       
        return daL
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(aL):
        daL = np.ones_like(aL)  
        daL[aL <= 0] = 0       
        return daL
    
class dl_functions:
    @staticmethod
    def initialize_parameters(lsz):
        # lsz is the list that contains size of the hidden layers and output layer
        parameters = {}
        L = len(lsz)
        for l in range(1, L):
            parameters["w" + str(l)] = np.random.randn(lsz[l], lsz[l-1]) * 0.01
            parameters["b" + str(l)] = np.zeros((lsz[l], 1))

            assert(np.shape(parameters["w" + str(l)]) == (lsz[l], lsz[l-1]))
            assert(np.shape(parameters["b" + str(l)]) == (lsz[l], 1))
        return parameters
