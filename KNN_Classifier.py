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
