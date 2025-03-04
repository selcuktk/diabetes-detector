import numpy as np

# deep learning functions
class dlf:
    @staticmethod
    def sigmoid(Z):
        
        A = 1/(1+np.exp(-Z))
        store = Z
        
        return A, store
    
    @staticmethod
    def sigmoid_backward(dA, store):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        store -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = store
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
        return dZ

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(aL, alpha=0.01):
        daL = np.ones_like(aL)    
        daL[aL < 0] = alpha       
        return daL
    
    @staticmethod
    def relu(Z):
        
        A = np.maximum(0,Z)
        store = Z 

        return A, store

    @staticmethod
    def relu_backward(dA, store):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        store -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        
        Z = store
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        
        return dZ
    
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
