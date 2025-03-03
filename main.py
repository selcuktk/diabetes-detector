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

        for i in range (num_iterations):
                m = X.shape[1]  # number of examples
                J = 0

                # Forward propagation
                """
                ZL1 matrix consists of first layer units before activation
                AL1 matrix consists of first layer units after activation and so on...
                """

                ZL1 = np.dot((wL1), X) + bL1
                AL1 = sig(ZL1)

                ZL2 = np.dot((wL2), AL1) + bL2
                AL2 = sig(ZL2)

                ZL3 = np.dot((wL3), AL2) + bL3
                AL3 = sig(ZL3)

                # Calculation of Cost function for logistic regression
                J += -((Y*(np.log(AL3))) + (1-Y)*(np.log(1-AL3)))
                J = np.sum(J)/m

                # Back propagation
                dzL3 = AL3 - Y
                dwL3 = (1/m) * np.dot(dzL3, AL2.T)
                dbL3 = (1/m) * np.sum(dzL3, axis=1, keepdims = True)

                dzL2 = np.dot(wL3.T, dzL3) * (AL2) * (1-AL2)
                dwL2 = (1/m) * np.dot(dzL2, AL1.T)
                dbL2 = (1/m) * np.sum(dzL2, axis=1, keepdims=True)

                dzL1 = np.dot(wL2.T, dzL2) * (AL1) * (1-AL1)
                dwL1 = (1/m) * np.dot(dzL1, X.T)
                dbL1 = (1/m) * np.sum(dzL1, axis=1, keepdims=True)

                # Gradient Descent - Update weights and biases
                wL1 -= learning_rate * dwL1
                bL1 -= learning_rate * dbL1

                wL2 -= learning_rate * dwL2
                bL2 -= learning_rate * dbL2

                wL3 -= learning_rate * dwL3
                bL3 -= learning_rate * dbL3

                # Printing costs for every loop
                print(f"{i} \t {J}")

        
        

def sig(x):
 return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    Main.main()