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


