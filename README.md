This project demonstrates the use of two optimization techniques—Gradient Descent and Newton's Method—for classifying diabetes based on a medical dataset. The objective is to predict whether a patient is diabetic or healthy using features such as glucose concentration, body mass index (BMI), and other relevant health metrics.

The dataset used for this project is from Kaggle and contains medical data of patients tested for diabetes.

Project Overview
Dataset: The "Diabetes.csv" dataset contains 768 examples and 9 features. The features include:

Number of pregnancies
Plasma glucose concentration
Diastolic blood pressure
Skinfold thickness at triceps
Insulin levels
Body Mass Index (BMI)
Diabetes pedigree function
Age of the patient
Class label (1 - Diabetic, 0 - Healthy)
Optimization Methods: The project applies two optimization techniques:

Gradient Descent: An iterative optimization method for minimizing a loss function.
Newton's Method: A second-order optimization method that utilizes the Hessian matrix to update parameters.
Data Splitting: The dataset is split into training (80%) and testing (20%) subsets. The training data is used to train the models, and the test data is used for evaluation.

Activation Function: The tanh (hyperbolic tangent) activation function is used for both optimization methods to model the predictions.

Objective Function: The binary cross-entropy loss is used as the objective function for optimization, which is minimized during training.

Evaluation Metrics:

Confusion Matrix: Used to assess the accuracy of the model on the test set.
F1-Score: Used as a performance metric, which balances precision and recall.
Features
Gradient Descent:

Iteratively updates model parameters to minimize the loss function.
The program tracks the gradient norm and execution time over iterations.
Newton's Method:

Uses second-order derivatives (Hessian matrix) to converge faster than Gradient Descent.
Tracks the gradient norm and execution time as well.
Visualization:

The project generates plots to visualize the convergence of both methods, including the evolution of the gradient norm and execution time across iterations.
Files
Diabetes.csv: Dataset containing medical data for diabetes classification.
train_folder/: Folder containing the training data (train_data.csv).
test_folder/: Folder containing the test data (test_data.csv).
Code implementation for both Gradient Descent and Newton's Method.

Results
After running the optimization methods, the following results are generated:

Gradient Descent Results:

The progress of the gradient norm is plotted over iterations.
Execution time for each iteration is recorded and plotted.
Confusion matrix and F1-score for the test data are displayed.
Newton's Method Results:

The progress of the gradient norm for Newton's Method is plotted.
Execution time and the norm of the gradient are visualized.
Confusion matrix and F1-score for the test data are displayed.
Example Output
Gradient Descent:

Confusion Matrix: Displaying true positives, false positives, true negatives, and false negatives.
F1-Score: A numerical value indicating the classification performance.
Newton's Method:

Confusion Matrix: Displaying true positives, false positives, true negatives, and false negatives.
F1-Score: A numerical value indicating the classification performance.
Conclusion
This project demonstrates the practical application of optimization methods (Gradient Descent and Newton's Method) in solving a binary classification problem. By comparing the performance and convergence speed of both methods, we gain insight into the efficiency and accuracy of optimization techniques in machine learning tasks.
