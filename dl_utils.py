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
    
    @staticmethod
    def initialize_parameters(layer_dims):
        
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1), dtype=float)
           
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters
    
    """
    liner_forward()             calculates output of multiplying weights with inputs for one layer
    liner_activation_forward()  calculates activation of obtained output values from that one layer
    L_model_forward()           calculates all layers of activated outputs
    """

    @staticmethod
    def linear_forward(A, W, b):

        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        store = (A, W, b)
        
        return Z, store
    
    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        store -- a python dictionary containing "linear_store" and "activation_store";
                stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_store".
            Z, linear_store = dlf.linear_forward(A_prev, W, b) 
            A, activation_store = dlf.sigmoid(Z)
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_store".
            Z, linear_store = dlf.linear_forward(A_prev, W, b) 
            A, activation_store = dlf.relu(Z) 
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        store = (linear_store, activation_store)

        return A, store
