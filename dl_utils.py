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
    
    @staticmethod
    def L_model_forward(X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        stores -- list of stores containing:
                    every store of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        stores = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "store" to the "stores" list.
        for l in range(1, L):
            A_prev = A 
            A, store = dlf.linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")   
            stores.append(store) 
        
        # Implement LINEAR -> SIGMOID. Add "store" to the "stores" list.
        AL, store = dlf.linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid") 
        stores.append(store)
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, stores

    @staticmethod
    def compute_cost(AL, Y):
        """
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = (-1/m)*np.sum((Y * np.log(AL)) + ((1-Y) * np.log(1-AL)), axis=1)
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost

    @staticmethod
    def linear_backward(dZ, store):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        store -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = store
        m = A_prev.shape[1]

        dW = (1/m)*(np.dot(dZ, A_prev.T)) 
        db = (1/m)*(np.sum(dZ, axis=1, keepdims=True)) 
        dA_prev = np.dot(W.T, dZ) 
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    
    @staticmethod
    def linear_activation_backward(dA, store, activation):   
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        store -- tuple of values (linear_store, activation_store) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_store, activation_store = store
        
        if activation == "relu":
            dZ = dlf.relu_backward(dA, activation_store) 
            dA_prev, dW, db = dlf.linear_backward(dZ, linear_store)
            
        elif activation == "sigmoid":
            dZ = dlf.sigmoid_backward(dA, activation_store)
            dA_prev, dW, db = dlf.linear_backward(dZ, linear_store)  
        
        return dA_prev, dW, db
    
    @staticmethod
    def L_model_backward(AL, Y, stores):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        stores -- list of stores containing:
                    every store of linear_activation_forward() with "relu" (it's stores[l], for l in range(L-1) i.e l = 0...L-2)
                    the store of linear_activation_forward() with "sigmoid" (it's stores[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(stores) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_store". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_store = stores[L-1]  
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = dlf.linear_activation_backward(dAL, current_store, "sigmoid") 
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_store". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_store = stores[l] 
            dA_prev_temp, dW_temp, db_temp = dlf.linear_activation_backward(grads["dA"+str(l+1)], current_store, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp 

        return grads
