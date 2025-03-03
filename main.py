import numpy as np

class Main:
    @staticmethod
    def main():

        # Load the CSV data into a NumPy array
        data = np.genfromtxt("health care diabetes.csv", delimiter=',', skip_header=1)

        # Separate features (X) and target (Y)
        X = data[:, :-1]  # All rows, all columns except the last (for features)
        Y = data[:, -1]   # All rows, last column (for target variable)
        X = X.T

        # Reshaping the target array (y)
        Y = Y.reshape(1, Y.size)
        
        print(X.shape)
        print("--------------")
        print(Y.shape)
        print("--------------")

        # CREATION OF a DEEP LEARNING MODEL in a STATIC WAY

        # This model implements a 3-layered deep learning model with 6-5-1 hidden units numbers recpectively.
        # Also it is non-modular and hardcoded.


        # initialization of the parameters for the model


        # initialization of the weights values
        # in this part of the project, all the number of units for layers are respectively 8, 6, 5, 1
        # 8 refers to number of features (X.shape[0])
        
        wL1 = np.random.randn(6, X.shape[0]) * 0.01      
        wL2 = np.random.randn(5, 6) * 0.01
        wL3 = np.random.randn(1, 5) * 0.01

        # initialization of the bias values

        bL1 = np.zeros((6, 1))
        bL2 = np.zeros((5, 1))
        bL3 = np.zeros((1, 1))

        # cost value, learning rate, number of iterations in for loop respectively
        
        J = 0
        learning_rate = 0.09
        num_iterations = 10000


        
        
            


if __name__ == "__main__":
    Main.main()