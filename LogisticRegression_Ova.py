import numpy as np
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
