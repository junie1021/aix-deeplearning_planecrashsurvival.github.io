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
```
python3 -m venv venv
source venv/bin/activate
```
### 필요 라이브러리 설치
```
requirements.txt:

pandas
numpy
matplotlib
scikit-learn

pip install -r requirements.txt
```
### 스크립트 실행
```
python3 SimpleLinear.py
python3 MultipleLinear.py
python3 LogisticRegression.py
python3 LogisticRegression_Ova.py
python3 DecisionTree.py
python3 KNN_Classifier.py
```
### 가상 환경 비활성화
```
deactivate
```
- 각 모델에 대한 설명 및 스크립트에 대한 설명입니다.
#### 단순 선형 회귀 (Simple Linear Regression): 하나의 독립 변수를 사용하여 종속 변수를 예측하는 모델입니다.
```
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
```
import pandas as pd
import numpy as np

#just like simple linear regression below, but with multiple features

class MultipleLinearRegression:
    def __init__(self):
        self.weights = None
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
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

    def predict(self, X):
        # Add bias column to match training format
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_with_bias @ self.weights
    
    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
def main():
    print("Multiple Linear Regression using house price prediction")

    df = pd.read_csv("house_prices_100.csv")

    print(df.head())

    X = df[['size_sqm', 'distance_to_city_km', 'bedrooms', 'age_years']].values
    y = df['price_million_won'].values

    model = MultipleLinearRegression()
    
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
```
# Using iris_150.csv, do logistic regression to predict the species of the iris

import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Clip z to avoid overflow/underflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
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
                cost = - (1 / n_samples) * np.sum(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9)) # Added 1e-9 for log stability
                print(f"   Epoch {epoch+1}/{self.max_iterations}, Cost: {cost:.4f}")
        
        print(f"Training complete!")
        print(f"   Final weights: {self.weights}")
        print(f"   Final bias: {self.bias}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

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

    model = LogisticRegression(learning_rate=0.1, max_iterations=1000) # Adjusted learning rate for potentially faster convergence

    # Split data
    X_train, X_test, y_train, y_test = model.train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
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
```
mport numpy as np
import pandas as pd

class LogisticRegressionOvA:
    def __init__(self, class_names, learning_rate=0.1, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.models = []
        self.class_names_ = class_names
        self.unique_classes_ = None

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

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

    def _train_binary_classifier(self, X_train_ova, y_train_ova_binary, class_value):
        n_samples, n_features = X_train_ova.shape
        weights = np.zeros(n_features)
        bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X_train_ova, weights) + bias
            y_predicted_proba = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X_train_ova.T, (y_predicted_proba - y_train_ova_binary))
            db = (1 / n_samples) * np.sum(y_predicted_proba - y_train_ova_binary)

            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
        
        return {'weights': weights, 'bias': bias, 'class_value': class_value}

    def fit(self, X_train, y_train):
        self.unique_classes_ = np.unique(y_train)
        self.models = []

        print("\n--- Individual OvA Classifier Training Info ---")
        for class_value in self.unique_classes_:
            class_value_int = int(class_value)
            current_class_name = self.class_names_[class_value_int]
            
            y_train_ova_binary = np.where(y_train == class_value, 1, 0)
            
            model_info = self._train_binary_classifier(X_train, y_train_ova_binary, class_value)
            self.models.append(model_info)

            print(f"\n--- Classifier: {current_class_name} vs. Others ---")
            print(f"Weights: {model_info['weights']}")
            print(f"Bias: {model_info['bias']:.4f}")

            predictions_binary_train = (self._sigmoid(np.dot(X_train, model_info['weights']) + model_info['bias']) >= 0.5).astype(int)
            accuracy_binary_train = self.accuracy(y_train_ova_binary, predictions_binary_train)
            print(f"Training Accuracy (Binary '{current_class_name} vs Others'): {accuracy_binary_train:.4f}")

    def predict_proba(self, X):
        probas_list = []
        for model_info in self.models:
            linear_model = np.dot(X, model_info['weights']) + model_info['bias']
            probas = self._sigmoid(linear_model)
            probas_list.append(probas)
        return np.array(probas_list).T

    def predict(self, X):
        probabilities = self.predict_proba(X)
        argmax_indices = np.argmax(probabilities, axis=1)
        return self.unique_classes_[argmax_indices]

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

def main():
    np.random.seed(42)

    csv_file_path = 'iris_150.csv'
    
    iris_dataframe = pd.read_csv(csv_file_path)

    species_names_list = ['setosa', 'versicolor', 'virginica']
    species_name_to_label_map = {name: i for i, name in enumerate(species_names_list)}
    label_to_species_name_map = {i: name for i, name in enumerate(species_names_list)}
    
    iris_dataframe['species_numerical'] = iris_dataframe['species'].map(species_name_to_label_map)

    class_names_list_for_model = species_names_list

    original_features = iris_dataframe[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    original_target_numerical = iris_dataframe['species_numerical'].values
    
    permutation_indices = np.random.permutation(len(original_features))
    shuffled_features = original_features[permutation_indices]
    shuffled_target_numerical = original_target_numerical[permutation_indices]

    shuffled_df_for_display = iris_dataframe.iloc[permutation_indices].reset_index(drop=True)
    
    print("--- Shuffled Dataset (first 5 samples from CSV) ---")
    print(shuffled_df_for_display[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']].head())
    print("-" * 70)

    ova_classifier = LogisticRegressionOvA(
        class_names=class_names_list_for_model,
        learning_rate=0.1, 
        n_iterations=3000 
    )

    X_train, X_test, y_train, y_test = ova_classifier.train_test_split(
        shuffled_features, shuffled_target_numerical, test_size=0.2, random_state=42
    )
    print(f"\nTraining data size: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test data size: X_test {X_test.shape}, y_test {y_test.shape}")

    ova_classifier.fit(X_train, y_train)

    test_set_predictions_idx = ova_classifier.predict(X_test)

    print("\n\n--- Test Set Prediction Results (first 10 samples) ---")
    print(f"{'Actual Species':<20} | {'Predicted Species':<20} | {'Correct?':<10}")
    print("-" * 60) 
    for i in range(min(10, len(X_test))):
        true_species_name = label_to_species_name_map[y_test[i]] 
        pred_species_name = label_to_species_name_map[test_set_predictions_idx[i]] 
        is_correct = "Yes" if y_test[i] == test_set_predictions_idx[i] else "No"
        print(f"{true_species_name:<20} | {pred_species_name:<20} | {is_correct:<10}")

    overall_test_accuracy = ova_classifier.accuracy(y_test, test_set_predictions_idx)
    num_correct_total = np.sum(y_test == test_set_predictions_idx)
    num_total_test = len(y_test)
    print(f"\n--- Overall Test Set Final Accuracy ---")
    print(f"Accuracy: {overall_test_accuracy:.4f} ({num_correct_total}/{num_total_test} correct)")

if __name__ == "__main__":
    main()
```

#### 결정 트리 (Decision Tree): 데이터의 특징을 기반으로 트리 구조의 분류/회귀 모델을 만듭니다.
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, gini=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini

class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def _calculate_gini(self, y):
        if y.size == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / y.size
        gini = 1 - np.sum(probabilities**2)
        return gini

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

    def _most_common_label(self, y):
        if y.size == 0:
            return None 
        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]

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

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

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
```
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

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict_single_instance(self, x_test_instance):
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

    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    print(f"\n--- Data Scaling Applied ---")
    print(f"Training data size after scaling: X_train {X_train_scaled.shape}, y_train {y_train.shape}")
    print(f"Test data size after scaling: X_test {X_test_scaled.shape}, y_test {y_test.shape}")
    if X_train_scaled.shape[0] > 0:
        print(f"Sample of scaled training data (first row): {X_train_scaled[0]}")
    print("-" * 70)

    k_value = 5
    knn_classifier = KNNClassifier(k=k_value)
    
    print(f"\n--- Training Custom K-NN Classifier (K={k_value}) ---")
    knn_classifier.fit(X_train_scaled, y_train)
    print("Training complete (data stored).")
    print("-" * 70)

    train_predictions = knn_classifier.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"\n--- Training Set Performance ---")
    print(f"Accuracy: {train_accuracy:.4f}")
    
    test_predictions = knn_classifier.predict(X_test_scaled)
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
# V. Conclusion: Discussion
