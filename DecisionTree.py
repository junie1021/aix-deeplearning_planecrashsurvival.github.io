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
