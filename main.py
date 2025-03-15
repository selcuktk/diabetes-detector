import numpy as np
from dl_utils import dlf


class Main:
    @staticmethod
    def main():

        # Load the CSV data into a NumPy array
        data = np.genfromtxt("health care diabetes.csv",
                             delimiter=',', skip_header=1)

        # Separate features (X) and target (Y)
        # All rows, all columns except the last (for features)
        X = data[:, :-1]
        Y = data[:, -1]   # All rows, last column (for target variable)

        X, averages, sdeviation = dlf.normalization(X)
        X = X.T

        # Reshaping the target array (y)
        Y = Y.reshape(1, Y.size)

        print(X.shape)
        print("--------------")
        print(Y.shape)
        print("--------------")

        # CREATION OF a DEEP LEARNING MODEL in a DYNAMIC WAY

        # This model implements a n-layered deep learning model with flexible numbers of hidden units.
        # Also it is modular compared to previous implementation.

    final_parameters = dlf.L_layer_model(X, Y, [X.shape[0], 10, 6, 1], 0.0075, 10000, True)


def sig(x):
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    Main.main()
