## aix-deeplearning_planecrashsurvival.github.io
# Title: 비행기 사고 생존율 분석
# Members
문준영 정보시스템공학과 Liquidangel6922@gmail.com 

Task: 모델구조 설계/ 코드 작성

박준희 융합전자공학부 stayfun.junie@gmail.com

Task: 모델에 대한 설명글 작성/ 블로그 작성 

정수민 컴퓨터소프트웨어학과 min_jr13@naver.com

Task: 모델에 대한 설명글 작성/ 영상촬영

# I. Proposal
### - Motivation
항공 안전 연구는 항공 산업의 지속적인 발전과 대중의 신뢰 확보에 있어 핵심적인 요소입니다. 과거 항공 사고로부터 얻은 교훈은 항공기 설계, 운영절차, 규제 기준의 개선을 이끌어 왔으며, 이는 항공여행을 가장 안전한 교통수단 중 하나로 만드는 데 기여하였습니다. 그럼에도 불구하고, 사고 발생 시 피해를 최소화하고 생존 가능성을 높이기 위한 노력은 끊임없이 이루어져야 합니다. 최근 몇 년간 머신러닝(ML) 기술의 발전은 방대한 양의 사고 데이터 속에 숨겨진 복잡한 패턴을 규명하고 이를 통해 보다 정교한 예측 및 분석을 가능하게 하는 새로운 지평을 열었습니다. 또한 근래에 우리나라에서 비행기 사고로 인한 참사가 발생한 바, 시의적절하다고 판단되어 비행기 사고 생존율 분석 모델을 구현하고자 하였습니다.
### - What do you want to see at the end?
본 프로젝트는 제공된 Kaggle의 "항공 사고 데이터베이스 시놉시스(Abiation Accident Database Synopses)" 데이터셋을 활용하여 항공 사고 발생시 생존율을 시뮬레이션하는 머신러닝모델을 개발하는 것을 목표로 합니다. 이 프로젝트는 적절한 머신러닝 알고리즘 선택 및 검증, 그리고 최종적으로 모델 결과를 해석하고 생존율을 시뮬레이션하는 과정을 다룹니다. 항공 사고 데이터 분석은 그 본질상 민감한 정보를 다루며, 결과 해석에 있어 신중함이 요구되므로 이러한 민감성을 고려하여 체계적이고 과학적인 접근방식을 통해 의미 있는 결과를 도출할 수 있도록 하는데 중점을 두었습니다. 
# II. Datasets
https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses


데이터셋은 Kaggle의 Aviation Accident Database & Synopses, up to 2023 를 이용하였습니다. 이 데이터셋은 1962년부터 현재까지의 민간 항공 사고와 미국, 미국 영토 및 속령, 국제 해역에서 발생한 특정 사건들에 대한 정보를 담고 있습니다.


iris_150.csv 는 로지스틱 회귀모델(이진), 로지스틱 회귀모델(OvA), 결정트리모델, K-NN 모델을 실험할 때 사용하였고, house_prices_100.csv 는 다중선형회귀모델, 단순선형회귀모델을 실험할 때 사용하였습니다. 
# III. Methodology
- 로지스틱 회귀모델(이진), 로지스틱 회귀모델(OvA), 결정트리모델, K-NN 모델, 다중선형회귀 모델, 단순선형회귀모델을 실행해보는 과정입니다.
### 가상 환경 생성 및 활성화
```C++
python3 -m venv venv
source venv/bin/activate
```
### 필요 라이브러리 설치
```C++
requirements.txt:

pandas
numpy
matplotlib
scikit-learn

pip install -r requirements.txt
```
### 스크립트 실행
```C++
python3 SimpleLinear.py
python3 MultipleLinear.py
python3 LogisticRegression.py
python3 LogisticRegression_Ova.py
python3 DecisionTree.py
python3 KNN_Classifier.py
```
### 가상 환경 비활성화
```C++
deactivate
```
- 각 모델에 대한 설명 및 스크립트에 대한 설명입니다.
#### 단순 선형 회귀 (Simple Linear Regression): 하나의 독립 변수를 사용하여 종속 변수를 예측하는 모델입니다.
```C++
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SimpleLinearRegression:
    def __init__(self):
        self.weights = 0
        self.bias = 0
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        self.weights = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
        self.bias = y_mean - self.weights * x_mean

        print(f"Weights: {self.weights}, Bias: {self.bias}")
        print(f"Equation: y = {self.weights}x + {self.bias}")

    def predict(self, X):
        return self.weights * X + self.bias
    
    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    

def main():
    print("Simple Linear Regression using house price prediction")

    df = pd.read_csv("house_prices_100.csv")

    print(df.head())

    # X = df['distance_to_city_km'].values 
    # y = df['price_million_won'].values

    # print(X)
    # print(y)

    # model = SimpleLinearRegression()
    # model.fit(X, y)

    # predictions = model.predict(X)

    # r2_score = model.score(X, y)
    # mse = np.mean((y - predictions) ** 2)

    # print(f"R2 Score: {r2_score}")
    # print(f"MSE: {mse}")

    # print(f"Predictions vs Actual")
    # for i in range(len(y)):
    #     print(f"Predicted: {predictions[i]}, Actual: {y[i]}")

    # plt.scatter(X, y, color='blue', marker='o', label='Training data')
    # plt.plot(X, predictions, color='red', label='Regression line')
    # plt.legend()
    # plt.show()

    X = df['size_sqm'].values
    y = df['price_million_won'].values

    model = SimpleLinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2_score_by_size_sqm = model.score(X, y)
    print(f"R2 Score by size_sqm: {r2_score_by_size_sqm}")

    X = df['distance_to_city_km'].values
    y = df['price_million_won'].values

    model = SimpleLinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2_score_by_distance_to_city_km = model.score(X, y)
    print(f"R2 Score by distance_to_city_km: {r2_score_by_distance_to_city_km}")

    X = df['bedrooms'].values
    y = df['price_million_won'].values

    model = SimpleLinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2_score_by_bedrooms = model.score(X, y)
    print(f"R2 Score by bedrooms: {r2_score_by_bedrooms}")
    
    X = df['age_years'].values
    y = df['price_million_won'].values

    model = SimpleLinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2_score_by_age_years = model.score(X, y)
    print(f"R2 Score by age_years: {r2_score_by_age_years}")    

    # Compare R2 scores with proper train/test split
    print("\n=== Proper Evaluation with Train/Test Split ===")
    features = ['size_sqm', 'distance_to_city_km', 'bedrooms', 'age_years']
    train_r2_scores = []
    test_r2_scores = []
    
    for feature in features:
        X = df[feature].values
        y = df['price_million_won'].values
        
        model = SimpleLinearRegression()
        X_train, X_test, y_train, y_test = model.train_test_split(X, y)
        
        # Train on training data only
        model.fit(X_train, y_train)
        
        # Evaluate on both sets
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        
        overfitting = train_r2 - test_r2
        print(f"{feature}: Train R2={train_r2:.4f}, Test R2={test_r2:.4f}, Overfitting={overfitting:.4f}")
    
    best_feature_index = np.argmax(test_r2_scores)
    best_feature = features[best_feature_index]
    best_test_r2 = test_r2_scores[best_feature_index]
    
    print(f"\nBest feature (by test R2): {best_feature} with Test R2 Score of {best_test_r2:.4f}")
    print(f"Data size: {len(df)} samples (was 20, now 100)")

if __name__ == "__main__":
    main()
```

#### 다중 선형 회귀 (Multiple Linear Regression): 여러 개의 독립 변수를 사용하여 종속 변수를 예측하는 모델입니다.
```C++
import pandas as pd
import numpy as np

# just like simple linear regression, but with multiple features
# Define a class for multiple linear regression
class MultipleLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0
    # Custom method to split dataset into training and test sets
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    # Fit the model using the normal equation
    def fit(self, X, y):
        # Add bias column (1s) to the input features
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        # Calcuate weights using the normal equation
        self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    # Predict values based on the input features
    def predict(self, X):
        # Add bias column to match training format
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_with_bias @ self.weights
    # Calculate R-squared score to evaluate the model
    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2) # Sum of squared residuals
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of sqaures
        return 1 - ss_res / ss_tot
    
def main():
    print("Multiple Linear Regression using house price prediction")
    # Load the dataset
    df = pd.read_csv("house_prices_100.csv")
    # Display the first few rows of the dataset
    print(df.head())
    # Select input features and target variable
    X = df[['size_sqm', 'distance_to_city_km', 'bedrooms', 'age_years']].values
    y = df['price_million_won'].values
    # Create model instance
    model = MultipleLinearRegression()
    # Split data into training and test sets
    # Use train/test split for proper evaluation
    X_train, X_test, y_train, y_test = model.train_test_split(X, y)
    
    # Train on training data only
    model.fit(X_train, y_train)

    # Evaluate on both training and test data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2_score = model.score(X_train, y_train)
    test_r2_score = model.score(X_test, y_test)
    
    print(f"Train R2 Score: {train_r2_score:.4f}")
    print(f"Test R2 Score: {test_r2_score:.4f}")
    print(f"Overfitting: {train_r2_score - test_r2_score:.4f}")
    
    # Show learned weights
    feature_names = ['bias', 'size_sqm', 'distance_to_city_km', 'bedrooms', 'age_years']
    print("\nLearned weights:")
    for i, (name, weight) in enumerate(zip(feature_names, model.weights)):
        print(f"{name}: {weight:.4f}")

if __name__ == "__main__":
    main()


```
#### 로지스틱 회귀 (Binary Logistic Regression):  이진 분류 문제를 해결하기 위한 모델입니다. (예: Iris 데이터셋에서 특정 품종인지 아닌지 분류)
```C++
# Using iris_150.csv, do logistic regression to predict the species of the iris

import pandas as pd # Import the pandas library for data maipulation
import numpy as np  # Import the numpy library for numerical operations

class LogisticRegression:
    # Initializes the learning rate and maximum number of iterations, and sets the initial weights and bias of the model to None
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    # Sigmoid function: Used as the activation function in logistic regression
    def sigmoid(self, z):
        # Clip z to avoid overflow/underflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    # Custom fuction to split the dataset into training and testing sets
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        n_test = int(test_size * len(X))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test
    # Trains the logistic regression model using the provided training data X and labels y
    # Optimizes the weights and bias using Gradient Descent
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        print(f"Starting Logistic Regression training...")
        print(f"   Number of samples: {n_samples}, Number of features: {n_features}")
        print(f"   Learning rate: {self.learning_rate}, Max iterations: {self.max_iterations}")

        # Gradient Descent
        for epoch in range(self.max_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 100 == 0:
                # Calculate and print the Binary Cross-Entropy Cost function
                cost = - (1 / n_samples) * np.sum(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9)) # Added 1e-9 for log stability
                print(f"   Epoch {epoch+1}/{self.max_iterations}, Cost: {cost:.4f}")
        
        print(f"Training complete!")
        print(f"   Final weights: {self.weights}")
        print(f"   Final bias: {self.bias}")
    # Returns the predicted probabilities for the input X
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    # Returns binary predictions (0 or 1) for the input X
    # Classifies as 1 if the predicted probability is greater than or equal to the threshold (default 0.5), otherwise 0
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    # Calculates the accuracy of the model
    # Compares predicted labels with actual labels and returns the proportion of correct matches
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
# main fuction: Defines the overall execution flow of the logistic regression model
def main():
    print("Iris Dataset Logistic Regression for Setosa vs. Others")
    print("-" * 50)

    # Load the dataset
    df = pd.read_csv("iris_150.csv") # Changed to relative path
    print("Dataset head:")
    print(df.head())
    print()

    # Prepare data: Predict 'Iris-setosa' (binary classification)
    # Iris-setosa -> 1, others (Iris-versicolor, Iris-virginica) -> 0
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = (df['species'] == 'Iris-setosa').astype(int).values
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # Creates and instance of the LogisticRegression model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000) # Adjusted learning rate for potentially faster convergence
    # Splits the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.2, random_state=42)
    # Trains the model using the training data
    model.fit(X_train, y_train)
    # Evaluates and prints the model's accuracy on the test data
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Performance Evaluation:")
    print(f"   Accuracy: {accuracy:.4f} (Correctly predicted {int(accuracy * len(y_test))} out of {len(y_test)} samples)")

    # Weights interpretation
    print(f"\nWeights Interpretation (Influence on Setosa Prediction):")
    for i, feature in enumerate(feature_names):
        weight = model.weights[i]
        change_direction = "increases" if weight > 0 else ("decreases" if weight < 0 else "does not change")
        print(f"   {feature:15s}: {weight:8.4f} (feature value increase -> Setosa probability {change_direction})")
    print(f"   bias              : {model.bias:8.4f}")
    print()

    # Detailed prediction results for the first 10 test samples
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    print(f"Test Data Prediction Results (random 10 samples):")
    #get random 10 samples and test, print the results
    random_indices = np.random.choice(len(X_test), 10, replace=False)
    for i in random_indices:
        print(f"{y_test[i]:>10d} {predictions[i]:>10d} {probabilities[i]:15.4f}")
    print()

if __name__ == "__main__":
    main()
```
#### 로지스틱 회귀 (One-vs-All, OvA for Multiclass): 다중 클래스 분류 문제를 해결하기 위해 이진 분류기를 여러 개 사용하는 OvA 방식의 로지스틱 회귀 모델입니다.
```C++
import numpy as np
import pandas as pd

class LogisticRegressionOvA:
    # Initializes learning rate, number of iterations, an empty list for models, and stores the provided class names
    def __init__(self, class_names, learning_rate=0.1, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.models = []
        self.class_names_ = class_names
        self.unique_classes_ = None
    # Sigmoid function
    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    # Shuffles the data and divides it based on the specified test_size
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    # Trains a single binary logistic regression classifier for a specific class (One-vs-All approach)
    # Performs gradient descent to find the optimal weights and bias for distinguishing 'class_value' from all other classes
    def _train_binary_classifier(self, X_train_ova, y_train_ova_binary, class_value):
        n_samples, n_features = X_train_ova.shape
        weights = np.zeros(n_features)
        bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X_train_ova, weights) + bias
            y_predicted_proba = self._sigmoid(linear_model)
            # Calculate gradients(derivatives of the cost function) for weights and bias
            dw = (1 / n_samples) * np.dot(X_train_ova.T, (y_predicted_proba - y_train_ova_binary))
            db = (1 / n_samples) * np.sum(y_predicted_proba - y_train_ova_binary)
            # Update weights and bias using the learning rate and gradients
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
        # Return the trained weights, bias, and the class value this model was trained for
        return {'weights': weights, 'bias': bias, 'class_value': class_value}
    # Fits the One-vs-All logistic regression model by training a separate binary classifier for each unique class present in the training labels
    def fit(self, X_train, y_train):
        self.unique_classes_ = np.unique(y_train) # Identify all unique classes in the target variable
        self.models = [] # Reset the list of models

        print("\n--- Individual OvA Classifier Training Info ---")
        for class_value in self.unique_classes_:
            class_value_int = int(class_value)
            current_class_name = self.class_names_[class_value_int]
            
            y_train_ova_binary = np.where(y_train == class_value, 1, 0)
            # Train a binary classifier for the current class
            model_info = self._train_binary_classifier(X_train, y_train_ova_binary, class_value)
            self.models.append(model_info)
 
            print(f"\n--- Classifier: {current_class_name} vs. Others ---")
            print(f"Weights: {model_info['weights']}")
            print(f"Bias: {model_info['bias']:.4f}")
            # Calculate and print the training accuracy for this specific binary classifier
            predictions_binary_train = (self._sigmoid(np.dot(X_train, model_info['weights']) + model_info['bias']) >= 0.5).astype(int)
            accuracy_binary_train = self.accuracy(y_train_ova_binary, predictions_binary_train)
            print(f"Training Accuracy (Binary '{current_class_name} vs Others'): {accuracy_binary_train:.4f}")
    # Predicts the probabilityof each input sample belonging to each class
    def predict_proba(self, X):
        probas_list = []
        for model_info in self.models:
            linear_model = np.dot(X, model_info['weights']) + model_info['bias']
            probas = self._sigmoid(linear_model)
            probas_list.append(probas)
        return np.array(probas_list).T # Transpose to have samples as rows and classes as columns
    # Predicts the class label for each input sample
    def predict(self, X):
        probabilities = self.predict_proba(X)
        argmax_indices = np.argmax(probabilities, axis=1)
        return self.unique_classes_[argmax_indices]
    # Calcuates the accuracy score by  comparing true labels with predicted labels
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
# Main function to execute the Logistic Regression OvA classification process 
def main():
    np.random.seed(42) # Set random seed for reproducibility

    csv_file_path = 'iris_150.csv'
    
    iris_dataframe = pd.read_csv(csv_file_path) # Load the dataset
    # Define class names and create mappings between names and numerical labels
    species_names_list = ['setosa', 'versicolor', 'virginica']
    species_name_to_label_map = {name: i for i, name in enumerate(species_names_list)}
    label_to_species_name_map = {i: name for i, name in enumerate(species_names_list)}
    # Convert species names to numerical labels in the dataframe
    iris_dataframe['species_numerical'] = iris_dataframe['species'].map(species_name_to_label_map)

    class_names_list_for_model = species_names_list
    # Extract feature(X) and target (y) as numpy arrays
    original_features = iris_dataframe[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    original_target_numerical = iris_dataframe['species_numerical'].values
    # Suffle the dataset(features and target together)
    permutation_indices = np.random.permutation(len(original_features))
    shuffled_features = original_features[permutation_indices]
    shuffled_target_numerical = original_target_numerical[permutation_indices]
    # Create a shuffled DataFrame for display purposes
    shuffled_df_for_display = iris_dataframe.iloc[permutation_indices].reset_index(drop=True)
    
    print("--- Shuffled Dataset (first 5 samples from CSV) ---")
    print(shuffled_df_for_display[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']].head())
    print("-" * 70)
    # Initialize the Logistic Regression OvA classifier with specified parameters
    ova_classifier = LogisticRegressionOvA(
        class_names=class_names_list_for_model,
        learning_rate=0.1, 
        n_iterations=3000 
    )
    # Split the shuffled data into training and testing sets
    X_train, X_test, y_train, y_test = ova_classifier.train_test_split(
        shuffled_features, shuffled_target_numerical, test_size=0.2, random_state=42
    )
    print(f"\nTraining data size: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test data size: X_test {X_test.shape}, y_test {y_test.shape}")
    # Train the One-vs-All classifier
    ova_classifier.fit(X_train, y_train)
    # Make predictions on the test set
    test_set_predictions_idx = ova_classifier.predict(X_test)
    # Print detailed prediction results for the first 10 sampled of the test set
    print("\n\n--- Test Set Prediction Results (first 10 samples) ---")
    print(f"{'Actual Species':<20} | {'Predicted Species':<20} | {'Correct?':<10}")
    print("-" * 60) 
    for i in range(min(10, len(X_test))):
        true_species_name = label_to_species_name_map[y_test[i]] 
        pred_species_name = label_to_species_name_map[test_set_predictions_idx[i]] 
        is_correct = "Yes" if y_test[i] == test_set_predictions_idx[i] else "No"
        print(f"{true_species_name:<20} | {pred_species_name:<20} | {is_correct:<10}")
    # Calculate and print the overall accuracy on the test set
    overall_test_accuracy = ova_classifier.accuracy(y_test, test_set_predictions_idx)
    num_correct_total = np.sum(y_test == test_set_predictions_idx)
    num_total_test = len(y_test)
    print(f"\n--- Overall Test Set Final Accuracy ---")
    print(f"Accuracy: {overall_test_accuracy:.4f} ({num_correct_total}/{num_total_test} correct)")

if __name__ == "__main__":
    main()
```

#### 결정 트리 (Decision Tree): 데이터의 특징을 기반으로 트리 구조의 분류/회귀 모델을 만듭니다.
```C++
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Node class to represent each node in the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, gini=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini
# Custom decision tree classifier implementation
class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
    # Calculate Gini impurity for a label array
    def _calculate_gini(self, y):
        if y.size == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.size
        gini = 1 - np.sum(probabilities**2)
        return gini
    # Find the best featureand threshold to split the dataset
    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None

        current_gini = self._calculate_gini(y)
        
        best_split_info = None
        max_gini_gain = 0 

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                if left_indices.size == 0 or right_indices.size == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                
                gini_left = self._calculate_gini(y_left)
                gini_right = self._calculate_gini(y_right)
                
                weighted_gini = (left_indices.size / n_samples) * gini_left + \
                                (right_indices.size / n_samples) * gini_right
                
                gini_gain = current_gini - weighted_gini

                if gini_gain > max_gini_gain:
                    max_gini_gain = gini_gain
                    best_split_info = {
                        'feature_index': feature_idx,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'gini_gain': gini_gain 
                    }
        
        return best_split_info
    # Recursively buil the decision tree
    def _build_tree(self, X, y, depth):
        n_samples, _ = X.shape
        current_node_gini = self._calculate_gini(y)
        
        if depth >= self.max_depth or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, gini=current_node_gini)

        best_split_info = self._find_best_split(X, y)

        if best_split_info is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, gini=current_node_gini)

        left_subtree = self._build_tree(
            X[best_split_info['left_indices']],
            y[best_split_info['left_indices']],
            depth + 1
        )
        right_subtree = self._build_tree(
            X[best_split_info['right_indices']],
            y[best_split_info['right_indices']],
            depth + 1
        )
        return Node(
            feature_index=best_split_info['feature_index'],
            threshold=best_split_info['threshold'],
            left=left_subtree,
            right=right_subtree,
            gini=current_node_gini
        )
    # Determine the most common label in a given array
    def _most_common_label(self, y):
        if y.size == 0:
            return None 
        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]
    # Train the decision tree model
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
        
        self.n_features_in_ = X_values.shape[1]
        self.root = self._build_tree(X_values, y_values, depth=0)
    # Traverse the tree for a single sample
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    # Predict classes for a dataset 
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        predictions = [self._traverse_tree(sample, self.root) for sample in X_values]
        return np.array(predictions)

def main():
    np.random.seed(42)
    csv_file_path = 'iris_150.csv'

    try:
        iris_dataframe = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: '{csv_file_path}' file not found. Please check the file path.")
        return

    iris_dataframe = iris_dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("--- Shuffled Dataset (first 5 samples) ---")
    print(iris_dataframe.head())
    print("-" * 70)

    X_df = iris_dataframe[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_series = iris_dataframe['species']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    
    class_names_list = label_encoder.classes_.tolist() 
    label_to_species_name_map = {i: name for i, name in enumerate(class_names_list)}

    print("\n--- Target Variable Encoding ---")
    print(f"Original species names: {class_names_list}")
    print(f"Encoded labels for first 5 samples: {y_encoded[:5]}")
    print(f"Mapping: {label_to_species_name_map}")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print(f"\nTraining data size: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test data size: X_test {X_test.shape}, y_test {y_test.shape}")
    print("-" * 70)

    custom_dt_classifier = CustomDecisionTreeClassifier(max_depth=5, min_samples_split=2)
    
    print("\n--- Training Custom Decision Tree Classifier ---")
    custom_dt_classifier.fit(X_train, y_train)
    print("Training complete.")
    print("-" * 70)

    train_predictions = custom_dt_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"\n--- Training Set Performance ---")
    print(f"Accuracy: {train_accuracy:.4f}")
    
    test_predictions = custom_dt_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\n--- Test Set Performance ---")
    print(f"Accuracy: {test_accuracy:.4f}")
    
    num_correct_total = np.sum(y_test == test_predictions)
    num_total_test = len(y_test)
    print(f"Correct predictions on test set: {num_correct_total}/{num_total_test}")
    print("-" * 70)

    print("\n--- Test Set Prediction Details (first 10 samples) ---")
    print(f"{'Actual Species':<20} | {'Predicted Species':<20} | {'Correct?':<10}")
    print("-" * 60)
    for i in range(min(10, len(X_test))):
        true_label = y_test[i]
        pred_label = test_predictions[i]
        true_species_name = label_to_species_name_map[true_label]
        pred_species_name = label_to_species_name_map.get(pred_label, "Unknown")
        is_correct = "Yes" if true_label == pred_label else "No"
        print(f"{true_species_name:<20} | {pred_species_name:<20} | {is_correct:<10}")
    print("-" * 70)

    # test set accuracy
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test set accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
```
#### K-최근접 이웃 (K-Nearest Neighbors, K-NN): 새로운 데이터 포인트와 가장 가까운 K개의 훈련 데이터 포인트를 기반으로 분류하는 모델입니다.
```C++
#Build a K-NN classifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def _euclidean_distance(self, point1, point2): # Calculate distance between two vectors(point)
        return np.sqrt(np.sum((point1 - point2)**2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict_single_instance(self, x_test_instance): # Calculate all distances, sort them, select k smallest distances, find the 'mode'
        distances = []
        for i, train_point in enumerate(self.X_train):
            distance = self._euclidean_distance(x_test_instance, train_point)
            distances.append((distance, self.y_train[i]))
        
        distances.sort(key=lambda tup: tup[0])
        
        neighbors_labels = []

        for i in range(self.k):
            if i < len(distances):
                neighbors_labels.append(distances[i][1])
            else:
                break 
        
        unique_labels, counts = np.unique(neighbors_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        return predicted_label

    def predict(self, X_test):
        predictions = [self._predict_single_instance(test_instance) for test_instance in X_test]

        return np.array(predictions)

def main():
    np.random.seed(42)
    csv_file_path = 'iris_150.csv'

    try:
        iris_dataframe = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: '{csv_file_path}' file not found. Please check the file path.")
        return
    # Sampling all datas in iris_daaframe(shuffling all indexes and discard original indexes)
    iris_dataframe = iris_dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    # Print head of dataframe(shuffled)
    print("--- Shuffled Dataset (first 5 samples) ---")
    print(iris_dataframe.head())
    print("-" * 70)
    # Making new dataframe which includes 4 columns; lengths and widths.
    X_df = iris_dataframe[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_series = iris_dataframe['species']
    # Using LabelEncoder, make 'species'series to integer series:[0,2,2,1,0,....]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    # Make dictionary between 'members of y_series' and 'integers'
    class_names_list = label_encoder.classes_.tolist() 
    label_to_species_name_map = {i: name for i, name in enumerate(class_names_list)}
    # Print those mappings
    print("\n--- Target Variable Encoding ---")
    print(f"Original species names: {class_names_list}")
    print(f"Encoded labels for first 5 samples: {y_encoded[:5]}")
    print(f"Mapping: {label_to_species_name_map}")
    print("-" * 70)
    # Split datas into 'for train'(80%) and 'for test'(20%), keeping ratio between each labels in splited datas
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    # Use StandardScaler to make datas 'scaled'
    # Use fit_transform scaling X_train to 'train(fit)' mean of data
    # Use only transform scaling X_test to keep independency of test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    # Print how datas are scaled
    print(f"\n--- Data Scaling Applied ---")
    print(f"Training data size after scaling: X_train {X_train_scaled.shape}, y_train {y_train.shape}")
    print(f"Test data size after scaling: X_test {X_test_scaled.shape}, y_test {y_test.shape}")
    if X_train_scaled.shape[0] > 0:
        print(f"Sample of scaled training data (first row): {X_train_scaled[0]}")
    print("-" * 70)
    # Set value of K in K-NN, and use KNNClassifier class
    k_value = 5
    knn_classifier = KNNClassifier(k=k_value)
    # knn_classifier.fit -> it saves those train datas
    print(f"\n--- Training Custom K-NN Classifier (K={k_value}) ---")
    knn_classifier.fit(X_train_scaled, y_train)
    print("Training complete (data stored).")
    print("-" * 70)
    # knn_classifier.predict -> it predicts labels of train sets, compares those labels and calculates accuracy.
    train_predictions = knn_classifier.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"\n--- Training Set Performance ---")
    print(f"Accuracy: {train_accuracy:.4f}")
    # Using test data, does same thing(calculate test accuracy)
    test_predictions = knn_classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\n--- Test Set Performance ---")
    print(f"Accuracy: {test_accuracy:.4f}")
    # Count how many right answers from test data are
    # Count how many test data are and print as ratio
    num_correct_total = np.sum(y_test == test_predictions)
    num_total_test = len(y_test)
    print(f"Correct predictions on test set: {num_correct_total}/{num_total_test}")
    print("-" * 70)
    # print detailed predicion from test data, using dicionary from y_series to show 'integer' and 'species name'
    print("\n--- Test Set Prediction Details (first 10 samples) ---")
    print(f"{'Actual Species':<20} | {'Predicted Species':<20} | {'Correct?':<10}")
    print("-" * 60)
    for i in range(min(10, len(X_test_scaled))): # Ensure we don't go out of bounds
        true_label = y_test[i]
        pred_label = test_predictions[i]
        true_species_name = label_to_species_name_map[true_label]
        pred_species_name = label_to_species_name_map.get(pred_label, "Unknown")
        is_correct = "Yes" if true_label == pred_label else "No"
        print(f"{true_species_name:<20} | {pred_species_name:<20} | {is_correct:<10}")
    print("-" * 70)
 
    #print all test accuracy
    print(f"\n--- Test Set Accuracy ---")
    print(f"Accuracy: {test_accuracy:.4f} ({num_correct_total}/{num_total_test})")

if __name__ == "__main__":
    main()
```


# IV. Evaluation & Analysis
- 각 모델별 실행결과입니다.
#### 단순 선형 회귀 (Simple Linear Regression)

![image](https://github.com/user-attachments/assets/3120c897-c794-4e65-98f5-34e5ab0d26ff)
✅ 실행 결과 분석:
독립 변수 1개와 종속 변수 간의 선형 관계를 모델링하였습니다.

예측 직선이 데이터 포인트들을 평균적으로 잘 통과하였습니다.

R² 값이 0.85 이상이면 설명력이 꽤 좋은 편입니다.

✅ 장점:
매우 직관적이고 해석이 용이합니다.

계산이 빠르고 구현이 쉽습니다.

❌ 단점:
변수 간 선형 관계만 모델링 가능합니다 (비선형 관계는 잘 설명 못함).

이상치(outlier)에 민감합니다.
  
#### 다중 선형 회귀 (Multiple Linear Regression)

![image](https://github.com/user-attachments/assets/a037867c-7b29-458e-aa03-5890bdfe1a04)

✅ 실행 결과 분석:
여러 개의 독립 변수를 활용해 종속 변수를 예측합니다.

R² 값이 0.9에 가까울수록 설명력이 좋습니다.

회귀 계수가 유의미한(p < 0.05) 변수는 모델에 중요한 영향을 미칩니다.

✅ 장점:
다양한 변수 고려 가능 → 복잡한 관계를 모델링할 수 있습니다.

해석 가능성과 설명력을 유지합니다.

❌ 단점:
다중공선성 문제가 생길 수 있습니다 (변수 간 상관관계가 높을 경우).

변수 선택이 잘못되면 모델 성능이 떨어집니다.
#### 로지스틱 회귀 (Binary Logistic Regression)

![image](https://github.com/user-attachments/assets/25f48eb8-b2d8-4ea7-af5b-aa7bade66d5b)
![image](https://github.com/user-attachments/assets/6e2f4b10-3974-4b73-ba2d-7cc263069082)

✅ 실행 결과 분석:
두 개의 클래스(예: 0 vs 1)를 예측합니다.

시그모이드 함수 기반으로 확률을 출력하고, 임계값(보통 0.5)으로 분류합니다.

혼동 행렬 및 정확도(accuracy), 정밀도(precision), 재현율(recall) 등 지표로 평가합니다.

✅ 장점:
확률 기반 출력 → 해석이 용이합니다.

이진 분류 문제에서 매우 효과적입니다.

❌ 단점:
선형 분리 가능한 경우에만 좋은 성능을 가집니다.

다중 클래스 문제에서는 직접 사용 불가합니다 (One-vs-All 등 기법 필요).


#### 로지스틱 회귀 (One-vs-All, OvA for Multiclass)

![image](https://github.com/user-attachments/assets/5ae9853c-d416-457a-a296-64fad56ffd70)
![image](https://github.com/user-attachments/assets/6d0b7dcd-194f-446d-afa9-ef5e3b2c801d)

✅ 실행 결과 분석:
각 클래스를 한 번에 하나씩 다른 클래스와 비교하여 이진 분류기를 학습합니다.

Softmax가 아닌 OvA 방식 사용 시, 각 클래스별 확률 중 가장 높은 값을 선택합니다.

✅ 장점:
단순한 이진 로지스틱 회귀를 여러 개 조합해 다중 클래스 분류 가능합니다.

구현이 비교적 쉽습니다.

❌ 단점:
클래스 수만큼 모델을 학습해야 하므로 비효율적일 수 있습니다.

클래스 간 불균형이 있는 경우 성능이 나쁠 수 있습니다.



#### 결정 트리 (Decision Tree)

![image](https://github.com/user-attachments/assets/1629f3ab-bc05-4050-bb83-4622e27f797f)
![image](https://github.com/user-attachments/assets/e6073309-5345-4eab-a11d-e6dc0c202787)

✅ 실행 결과 분석:
트리 구조를 따라 분류 또는 예측을 수행합니다.

각 분기점에서 조건을 기준으로 데이터를 나눕니다.

시각화된 트리에서 조건과 분기 구조를 확인 가능합니다.

✅ 장점:
모델이 직관적이며 시각화와 해석이 쉽습니다.

비선형 관계나 범주형 변수도 처리 가능합니다.

❌ 단점:
과적합(overfitting)의 위험이 큽니다 (특히 깊은 트리일 경우).

데이터가 조금만 달라져도 트리 구조가 많이 바뀔 수 있습니다 (불안정성).



#### K-최근접 이웃 (K-Nearest Neighbors, K-NN)

![image](https://github.com/user-attachments/assets/574fd70c-9572-4443-8061-c1049bafb6a7)
![image](https://github.com/user-attachments/assets/e1a747ea-03b8-45e4-82e0-71fa0921e065)

✅ 실행 결과 분석:
테스트 데이터가 주어졌을 때, 학습 데이터 중 가장 가까운 K개의 이웃을 기반으로 분류합니다.

K=3, 5 등 여러 값으로 실험 가능. 결정 경계는 복잡할 수 있습니다.

✅ 장점:
비모수 모델입니다 (모델 학습이 없고, 메모리에 데이터 저장 후 예측 시 계산).

데이터 분포를 잘 반영하며 복잡한 결정 경계도 학습 가능합니다.

❌ 단점:
계산 비용이 큽니다 (모든 데이터와 거리 계산 필요).

고차원 데이터에서는 거리 계산의 효율성이 떨어집니다 (차원의 저주).

이상치나 잡음에 민감할 수 있습니다.

### 로지스틱 회귀모델(이진), 로지스틱 회귀모델(OvA), 결정트리모델, K-NN 모델, 다중선형회귀 모델, 단순선형회귀모델을 실행해본 결과, RandomForest 모델을 사용하여 비행기 사고 생존율 예측 모델을 구현하기로 하였습니다.

### RandomForest 모델을 선택한 이유

1. 복잡한 데이터 구조에 강함
   
- 비행기 사고 데이터는 다양한 변수(날짜, 장소, 사고원인, 기종, 기상, 탑승 인원 등)가 얽혀 있어 비선형적이고 복잡한 상호작용이 존재할 가능성이 큽니다.

- RandomForest는 비선형 관계와 변수 간 복잡한 상호작용을 자동으로 모델링해줍니다.

2. 과적합(overfitting)에 강함
   
- 단일 결정 트리는 매우 직관적이지만 과적합의 문제가 있습니다.
  
- RandomForest는 여러 개의 결정 트리를 만들고 결과를 앙상블(다수결)하여 예측하기 때문에 과적합을 효과적으로 방지할 수 있습니다.

3. 범주형/수치형 데이터 모두 처리 가능
- 사고 기록에는 날짜, 위치(범주형), 사망자 수 (수치형), 사고 원인(범주형) 등 다양한 형식의 데이터가 있습니다.
  
- RandomForest는 이러한 다양한 형식의 변수들을 별도의 전처리 없이 자연스럽게 처리할 수 있습니다.

### RandomForest 모델의 작동 원리

1. 부트스트랩 샘플링:

데이터셋에서 여러 개의 샘플을 랜덤하게 뽑아 서브셋을 생성합니다. 이 과정은 '부트스트랩 샘플링'이라고 하며, 각 서브셋은 원본 데이터셋에서 중복을 허용하여 생성됩니다.

2. 결정 트리 생성:

각 부트스트랩 샘플로부터 결정 트리를 만듭니다. 이때, 트리의 각 노드에서 최적의 분할을 찾을 때 전체 특징 중 임의로 선택된 일부 특징만을 고려합니다. 이로 인해 트리들이 서로 다르게 성장하게 됩니다.

3. 앙상블:

여러 개의 결정 트리로 구성된 Random Forest 모델은 각 트리의 예측을 결합하여 최종 예측을 수행합니다. 분류 문제의 경우, 각 트리의 예측 중 가장 많이 나온 클래스를 선택하여 최종 클래스를 결정합니다(다수결 투표). 회귀 문제의 경우, 각 트리의 예측값을 평균하여 최종 예측값을 결정합니다.

### RandomForest 모델 실행결과 분석

![image](https://github.com/user-attachments/assets/9fd0030b-2a984ed0-a794-9aa1a4ca8eee)


- 데이터 로딩

88,889개의 기록과 31개의 열이 포함된 데이터셋이 성공적으로 로드되었습니다.
지리적 분석을 돕기 위해 주 코드도 로드되었습니다.

- 데이터 탐색

데이터는 1948년부터 2022년까지의 범위를 포함하며, 메모리 사용량은 122.1 MB입니다.
주요 생존 데이터 포인트:
치명적 부상: 평균 0.6
심각한 부상: 평균 0.3
경미한 부상: 평균 0.4
부상 없음: 평균 5.3

- 결측치

일부 열은 많은 결측치를 가지고 있으며, 예를 들어 Schedule은 85.8%, Air.carrier는 81.3%입니다.

- 생존율 분석

유효한 사례: 98.5%
평균 생존율: 81.6%
분포에 따르면 79.7%의 사례가 완전 생존을 보였습니다.

- 특성 선택

결측치가 적고 높은 관련성을 가진 8개의 주요 특성이 모델에 선택되었습니다:
Injury.Severity
Aircraft.Category
Weather.Condition
Broad.phase.of.flight
Engine.Type
Amateur.Built
Purpose.of.flight
Number.of.Engines

- 모델 학습

데이터셋은 학습용(70,064 샘플)과 테스트용(17,516 샘플)으로 분할되었습니다.
랜덤 포레스트 모델은 200개의 트리를 사용했으며 좋은 일반화 성능을 보였습니다:
학습 RMSE: 9.01%
테스트 RMSE: 9.27%
학습 R²: 0.9428
테스트 R²: 0.9395

- 특성 중요도

가장 중요한 특성은 Injury.Severity로, 중요도 점수는 0.9080입니다.
Weather.Condition 및 Broad.phase.of.flight도 중요한 특성으로 나타났습니다.

- 예측 예시

모델은 생존율을 최소한의 오차로 예측하며, 그 효과성을 입증했습니다.

# V. Conclusion: Discussion
랜덤 포레스트 모델은 항공 사고 생존율을 효과적으로 예측하며, Injury.Severity가 가장 중요한 특성입니다. 모델의 높은 R² 값과 낮은 오차율은 강력한 예측 성능을 나타냅니다. 향후 개선점으로는 특정 특성의 결측 데이터를 해결하여 모델의 견고성을 향상시키는 것이 있습니다.

- 모델의 강점
  
복잡한 상호작용: 비선형 및 복잡한 상호작용을 잘 처리하여 다양한 변수 간의 관계를 효과적으로 모델링합니다.

과적합 방지: 여러 결정 트리를 앙상블하여 단일 트리보다 일반화 성능이 뛰어납니다.

다양한 데이터 처리: 수치형 및 범주형 데이터를 모두 자연스럽게 처리할 수 있어 다양한 형식의 데이터를 다루는 데 유리합니다.

- 개선 및 고려 사항
  
결측치 처리: 일부 변수에서 높은 결측치 비율이 있으므로 이를 보완하기 위한 추가적인 데이터 전처리 기법이 필요합니다. 결측치를 효과적으로 처리하면 모델의 성능을 더욱 향상시킬 수 있습니다.

특성 확장: 현재 모델은 8개의 주요 특성을 사용하고 있는데, 추가적인 데이터 수집이나 새로운 특성 도출을 통해 모델의 예측력을 강화할 수 있습니다.

모델 해석 가능성: 랜덤 포레스트는 블랙박스 모델로 간주될 수 있으므로, 설명 가능한 AI 기법을 활용하여 모델의 예측 결과를 더 잘 이해할 수 있는 방안을 고려할 수 있습니다.

실시간 적용 가능성: 항공 사고의 경우 실시간 데이터가 중요할 수 있으므로, 모델의 실시간 적용 가능성을 검토하고 필요한 최적화 작업을 진행할 수 있습니다.

- 미래 연구 방향
  
다른 모델과의 비교: XGBoost, LightGBM 등 다른 앙상블 기법과의 성능 비교를 통해 최적의 모델을 선택할 수 있습니다.
시나리오 분석: 다양한 사고 시나리오에 대한 시뮬레이션을 통해 모델의 적용 범위를 확장할 수 있습니다.
도메인 전문가 협업: 항공 안전 전문가와의 협력을 통해 모델의 해석을 개선하고, 실제 항공 안전 정책에 기여할 수 있는 방안을 모색할 수 있습니다.




