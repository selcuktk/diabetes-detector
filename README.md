![Diabetes_Detector](https://github.com/user-attachments/assets/2348cfff-1364-47a2-94b6-b1ca1c7ae409)

# Diabetes Detection Using Deep Learning

Diabetes Detector is a deep learning project designed to determine whether a person has diabetes. Focused on enhancing the programmer's understanding of deep learning, **this project is developed without relying on pre-built libraries like TensorFlow**. It solely revolves around the model, with no backend or frontend integration. After training the model on selected data, users can test their own patient data to detect the presence of diabetes.

## Features

- **Implementation of multiple optimization algorithms:** Gradient Descent, Momentum, RMSprop, Adam.
- **No Backend/Frontend:** A lightweight solution focused solely on the deep learning model.

## Tech Stack

- **Programming Language:** Python
- **Python Libraries**
    - NumPy: For matrix operations and numerical computations.

## Installation

- Clone the project repository:

```bash
git clone https://github.com/selcuktk/diabetes-detector.git
cd diabetes-detector
```

- Install required libraries:

```bash
pip install numpy
```

- Version of the used libraries:

```bash
Package Version
------- -------
numpy   2.2.3
pip     25.0.1
```

## Phase of the Project

The project consists of three phases: non-modular model, modular model, and the implementation of four different optimization algorithms. All four classifiers can be accessed at the leaf branches of the Git graph.

![image](https://github.com/user-attachments/assets/1f7592d0-9f2b-420d-b21f-aacdeae3614d)

### Non-modular Static Model
- Initially, a deep learning algorithm was implemented layer by layer without modular structures (using Gradient Descent).
- The model is not flexible, and the layer sizes are 6, 5, 1 (excluding the input layer).
- Below is an image showing a part of the static model:

![image](https://github.com/user-attachments/assets/d74f1254-7b0e-43de-b25c-c86b97051ef4)

### Modular Dynamic Model
- After implementing the static model, it was converted into a dynamic model by creating `dl_utils.py` (Deep Learning Utilities class). This class contains:

    1. Normalization for data preparation
    2. Activation functions (Sigmoid and ReLU)
    3. Initialization of weights
    4. Calculation of forward propagation for one layer (`linear_forward()`, `linear_activation_forward()`)
    5. Calculation of forward propagation for all layers (`L_model_forward()`)
    6. Computation of cost function
    7. Calculation of backward propagation for one layer (`linear_backward()`, `linear_activation_backward()`)
    8. Calculation of backward propagation for all layers (`L_model_backward()`)
    9. Optimization using Gradient Descent
    10. Integration of the full system (`L_layer_model()`)

### Optimization Algorithms (Gradient Descent, Momentum, RMSprop, Adam)
* The project includes four classifiers that are identical except for the optimization algorithm used.
* There was an issue with optimization caused by vanishing/exploding gradients. This issue was **partially fixed but remains fragile**, so only one example is shared for comparison.
* Below is the test statement used to compare all classifiers:

![image](https://github.com/user-attachments/assets/5d264853-2c7a-437b-9611-c433c897c3a1)

* **Results:**

![image](https://github.com/user-attachments/assets/940de5e4-4e44-438c-b21f-a5b20ea1f079)

### Conclusions Drawn from the Project

* As expected, **Gradient Descent has the highest error rate**, while **Adam optimization has the lowest**.
* When computing the error, **it is crucial to avoid 0 or 1 as error rates**, as this can cause **divide-by-zero errors or vanishing/exploding gradients**.
* **Normalization speeds up training.**
* **Evaluating a classifier solely on training error is not ideal**, but for this project, the goal was to **compare optimization algorithms against each other**.

