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
