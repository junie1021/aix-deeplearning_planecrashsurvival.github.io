import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available, using matplotlib only")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class AviationSurvivalPredictor:
    
    def __init__(self):
        self.aviation_data = None
        self.state_codes = None
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'survival_rate'
        
    def load_data(self):
        print("Loading aviation accident dataset from local files...")
        print("=" * 60)
        
        try:
            self.aviation_data = pd.read_csv('AviationData.csv', encoding='cp1252', low_memory=False)
            print(f"Aviation data loaded: {self.aviation_data.shape}")
            
            self.state_codes = pd.read_csv('USState_Codes.csv')
            print(f"State codes loaded: {self.state_codes.shape}")
            
            print(f"\nAvailable Columns ({len(self.aviation_data.columns)}):")
            for i, col in enumerate(self.aviation_data.columns, 1):
                print(f"  {i:2d}. {col}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure AviationData.csv and USState_Codes.csv are in the current directory")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def explore_data(self):
        print("\n" + "=" * 60)
        print("AVIATION ACCIDENT DATA EXPLORATION")
        print("=" * 60)
        
        print(f"\nDataset Overview:")
        print(f"  Total accidents: {len(self.aviation_data):,}")
        print(f"  Date range: {self.aviation_data['Event.Date'].min()} to {self.aviation_data['Event.Date'].max()}")
        print(f"  Memory usage: {self.aviation_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        survival_cols = ['Total.Fatal.Injuries', 'Total.Serious.Injuries', 
                        'Total.Minor.Injuries', 'Total.Uninjured']
        
        print(f"\nSurvival Data Analysis:")
        for col in survival_cols:
            if col in self.aviation_data.columns:
                non_null = self.aviation_data[col].notna().sum()
                total = len(self.aviation_data)
                print(f"  {col}: {non_null:,}/{total:,} ({non_null/total*100:.1f}%) non-null")
                
                numeric_values = pd.to_numeric(self.aviation_data[col], errors='coerce')
                if numeric_values.notna().sum() > 0:
                    print(f"    Range: {numeric_values.min():.0f} - {numeric_values.max():.0f}")
                    print(f"    Mean: {numeric_values.mean():.1f}")
        
        print(f"\nMissing Values (Top 10):")
        missing_pct = (self.aviation_data.isnull().sum() / len(self.aviation_data) * 100).sort_values(ascending=False)
        for col, pct in missing_pct.head(10).items():
            print(f"  {col}: {pct:.1f}%")
        
        categorical_cols = ['Injury.Severity', 'Aircraft.Category', 'Weather.Condition', 
                           'Broad.phase.of.flight', 'Engine.Type']
        
        print(f"\nKey Categorical Variables:")
        for col in categorical_cols:
            if col in self.aviation_data.columns:
                unique_count = self.aviation_data[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                
                if unique_count <= 10:
                    value_counts = self.aviation_data[col].value_counts()
                    print(f"    Values: {', '.join(value_counts.head(5).index.tolist())}")
    
    def create_survival_rate_target(self):
        print("\n" + "=" * 60)
        print("CREATING SURVIVAL RATE TARGET VARIABLE")
        print("=" * 60)
        
        fatal_col = 'Total.Fatal.Injuries'
        serious_col = 'Total.Serious.Injuries'
        minor_col = 'Total.Minor.Injuries'
        uninjured_col = 'Total.Uninjured'
        
        print(f"Using columns:")
        print(f"  Fatal: {fatal_col}")
        print(f"  Serious: {serious_col}")
        print(f"  Minor: {minor_col}")
        print(f"  Uninjured: {uninjured_col}")
        
        try:
            fatal = pd.to_numeric(self.aviation_data[fatal_col], errors='coerce').fillna(0)
            serious = pd.to_numeric(self.aviation_data[serious_col], errors='coerce').fillna(0)
            minor = pd.to_numeric(self.aviation_data[minor_col], errors='coerce').fillna(0)
            uninjured = pd.to_numeric(self.aviation_data[uninjured_col], errors='coerce').fillna(0)
            
            total_aboard = fatal + serious + minor + uninjured
            survivors = serious + minor + uninjured
            
            survival_rate = np.where(total_aboard > 0, (survivors / total_aboard * 100), np.nan)
            survival_rate = np.clip(survival_rate, 0, 100)
            
            self.aviation_data['total_aboard'] = total_aboard
            self.aviation_data['survivors'] = survivors
            self.aviation_data['survival_rate'] = survival_rate
            
            valid_cases = (total_aboard > 0) & (~np.isnan(survival_rate))
            valid_data = self.aviation_data[valid_cases].copy()
            
            print(f"\nSurvival Rate Statistics:")
            print(f"  Valid cases: {valid_cases.sum():,} / {len(self.aviation_data):,} ({valid_cases.sum()/len(self.aviation_data)*100:.1f}%)")
            print(f"  Mean survival rate: {valid_data['survival_rate'].mean():.1f}%")
            print(f"  Std deviation: {valid_data['survival_rate'].std():.1f}%")
            print(f"  Min: {valid_data['survival_rate'].min():.1f}%")
            print(f"  Max: {valid_data['survival_rate'].max():.1f}%")
            
            print(f"\nSurvival Rate Distribution:")
            survival_rates = valid_data['survival_rate']
            print(f"  0% (Complete fatality): {(survival_rates == 0).sum():,} cases ({(survival_rates == 0).mean()*100:.1f}%)")
            print(f"  1-20% (Very low): {((survival_rates > 0) & (survival_rates <= 20)).sum():,} cases")
            print(f"  21-40% (Low): {((survival_rates > 20) & (survival_rates <= 40)).sum():,} cases")
            print(f"  41-60% (Medium): {((survival_rates > 40) & (survival_rates <= 60)).sum():,} cases")
            print(f"  61-80% (High): {((survival_rates > 60) & (survival_rates <= 80)).sum():,} cases")
            print(f"  81-99% (Very high): {((survival_rates > 80) & (survival_rates < 100)).sum():,} cases")
            print(f"  100% (Complete survival): {(survival_rates == 100).sum():,} cases ({(survival_rates == 100).mean()*100:.1f}%)")
            
            self.aviation_data = valid_data
            
            return True
            
        except Exception as e:
            print(f"Error creating survival rate: {e}")
            return False
    
    def prepare_features(self):
        print("\n" + "=" * 60)
        print("FEATURE PREPARATION")
        print("=" * 60)
        
        potential_features = {
            'categorical': [
                'Injury.Severity',
                'Aircraft.Category', 
                'Weather.Condition',
                'Broad.phase.of.flight',
                'Engine.Type',
                'Make',
                'Amateur.Built',
                'Schedule',
                'Purpose.of.flight'
            ],
            'numerical': [
                'Number.of.Engines',
                'Latitude',
                'Longitude'
            ]
        }
        
        selected_features = []
        
        print("Checking feature availability and quality:")
        
        for feature in potential_features['categorical']:
            if feature in self.aviation_data.columns:
                missing_pct = self.aviation_data[feature].isnull().mean() * 100
                unique_count = self.aviation_data[feature].nunique()
                
                if missing_pct < 70 and unique_count > 1 and unique_count < 1000:
                    selected_features.append(feature)
                    print(f"  Feature {feature}: {missing_pct:.1f}% missing, {unique_count} unique values")
                else:
                    print(f"  Skip {feature}: {missing_pct:.1f}% missing, {unique_count} unique values")
            else:
                print(f"  Not found {feature}")
        
        for feature in potential_features['numerical']:
            if feature in self.aviation_data.columns:
                missing_pct = self.aviation_data[feature].isnull().mean() * 100
                if missing_pct < 70:
                    numeric_values = pd.to_numeric(self.aviation_data[feature], errors='coerce')
                    if numeric_values.notna().sum() > len(self.aviation_data) * 0.3:
                        selected_features.append(feature)
                        print(f"  Feature {feature}: {missing_pct:.1f}% missing (numerical)")
                    else:
                        print(f"  Skip {feature}: Too many invalid numerical values")
                else:
                    print(f"  Skip {feature}: {missing_pct:.1f}% missing")
        
        print(f"\nFinal selected features ({len(selected_features)}):")
        for i, feature in enumerate(selected_features, 1):
            print(f"  {i}. {feature}")
        
        self.feature_columns = selected_features
        return selected_features
    
    def preprocess_features(self, features):
        print(f"\nPreprocessing features...")
        
        processed_data = self.aviation_data[features + ['survival_rate']].copy()
        
        for feature in features:
            print(f"  Processing {feature}...")
            
            if self.aviation_data[feature].dtype == 'object':
                processed_data[feature] = processed_data[feature].fillna('Unknown')
                
                le = LabelEncoder()
                processed_data[feature] = le.fit_transform(processed_data[feature].astype(str))
                self.label_encoders[feature] = le
                
                print(f"    Categorical: {len(le.classes_)} categories")
                
            else:
                median_value = pd.to_numeric(processed_data[feature], errors='coerce').median()
                processed_data[feature] = pd.to_numeric(processed_data[feature], errors='coerce').fillna(median_value)
                
                print(f"    Numerical: filled with median {median_value:.2f}")
        
        processed_data = processed_data.dropna(subset=['survival_rate'])
        
        print(f"  Final processed data shape: {processed_data.shape}")
        
        return processed_data
    
    def train_random_forest_model(self, features):
        print("\n" + "=" * 60)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 60)
        
        processed_data = self.preprocess_features(features)
        
        X = processed_data[features]
        y = processed_data['survival_rate']
        
        print(f"Training data: {X.shape[0]:,} samples, {X.shape[1]} features")
        print(f"Target range: {y.min():.1f}% - {y.max():.1f}%")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.digitize(y, bins=[0, 25, 50, 75, 100])
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        print(f"\nTraining Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        test_pred = np.clip(test_pred, 0, 100)
        train_pred = np.clip(train_pred, 0, 100)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nModel Performance:")
        print(f"  Training RMSE: {train_rmse:.2f}%")
        print(f"  Test RMSE: {test_rmse:.2f}%")
        print(f"  Training MAE: {train_mae:.2f}%")
        print(f"  Test MAE: {test_mae:.2f}%")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        if train_rmse < test_rmse * 0.7:
            print(f"  Possible overfitting detected")
        else:
            print(f"  Good generalization")
        
        self.analyze_feature_importance(features)
        
        self.show_prediction_examples(X_test, y_test, test_pred)
        
        return X_test, y_test, test_pred
    
    def analyze_feature_importance(self, features):
        print(f"\nFeature Importance Analysis:")
        
        importance_scores = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        if HAS_SEABORN:
            sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
        else:
            plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance Score')
        
        print(f"\nTop 3 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(3).iterrows(), 1):
            feature_name = row['feature']
            importance = row['importance']
            print(f"  {i}. {feature_name} (importance: {importance:.4f})")
            
            if feature_name in self.label_encoders:
                le = self.label_encoders[feature_name]
                print(f"     Categories: {', '.join(le.classes_[:5])}{'...' if len(le.classes_) > 5 else ''}")
            else:
                values = pd.to_numeric(self.aviation_data[feature_name], errors='coerce')
                print(f"     Range: {values.min():.2f} - {values.max():.2f}")
        
        plt.subplot(2, 1, 2)
        plt.hist(self.aviation_data['survival_rate'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Survival Rates')
        plt.xlabel('Survival Rate (%)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def show_prediction_examples(self, X_test, y_test, y_pred):
        print(f"\nPrediction Examples (Random 10 cases):")
        print(f"{'Actual':>8} {'Predicted':>10} {'Error':>8} {'Category'}")
        print("-" * 45)
        
        sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
        
        for idx in sample_indices:
            actual = y_test.iloc[idx]
            predicted = y_pred[idx]
            error = abs(actual - predicted)
            
            if actual == 0:
                category = "Complete fatality"
            elif actual == 100:
                category = "Complete survival"
            elif actual < 25:
                category = "Low survival"
            elif actual < 75:
                category = "Medium survival"
            else:
                category = "High survival"
            
            print(f"{actual:8.1f}% {predicted:9.1f}% {error:7.1f}% {category}")
    
    def run_complete_analysis(self):
        print("AVIATION ACCIDENT SURVIVAL RATE PREDICTION")
        print("Using Random Forest Machine Learning")
        print("=" * 60)
        
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return
        
        self.explore_data()
        
        if not self.create_survival_rate_target():
            print("Failed to create survival rate target. Exiting.")
            return
        
        features = self.prepare_features()
        if not features:
            print("No suitable features found. Exiting.")
            return
        
        try:
            X_test, y_test, y_pred = self.train_random_forest_model(features)
            
            print(f"\nAnalysis completed successfully!")
            print(f"Model trained on {len(self.aviation_data):,} aviation accidents")
            print(f"{len(features)} features used for prediction")
            print(f"Random Forest with 200 trees")
            
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()

def main():
    predictor = AviationSurvivalPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()
