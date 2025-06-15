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