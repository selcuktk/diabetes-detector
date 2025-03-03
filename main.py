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


        
        
            


if __name__ == "__main__":
    Main.main()