# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import warnings
from config import HYPERPARAMETER_GRIDS, MODELING_FEATURE_COLUMNS

warnings.filterwarnings('ignore')

class HeatingRetrofitModel:
    """
    Manages training, evaluation, and tuning of machine learning models
    for heating retrofit potential.
    """
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None

    def train_models(self, X, y):
        """Train multiple ML models and compare performance"""
        print("=== TRAINING MULTIPLE ML MODELS ===")

        if X.empty or y.empty:
            print("Warning: Input data for training is empty. Skipping model training.")
            return {}, None, None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }

        # Train and evaluate models
        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            try:
                # Fit model
                model.fit(X_train, y_train)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

                # Test predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                # Store results
                results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_predictions': y_pred,
                    'test_probabilities': y_pred_proba,
                    'X_test': X_test,
                    'y_test': y_test
                }
                print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error training {name}: {e}. Skipping this model.")
                # Store placeholder results for failed models
                results[name] = {'model': None, 'cv_mean': -1, 'cv_std': 0, 'test_predictions': None, 'test_probabilities': None, 'X_test': X_test, 'y_test': y_test}


        # Filter out models that failed training before finding the best one
        successful_models = {name: res for name, res in results.items() if res['model'] is not None}

        if not successful_models:
            print("No models were successfully trained. Cannot determine best model.")
            self.best_model = None
            return results, X_test, y_test

        best_model_name = max(successful_models.keys(), key=lambda x: successful_models[x]['cv_mean'])
        self.best_model = successful_models[best_model_name]['model']
        self.models = successful_models

        print(f"\nBest Model: {best_model_name}")
        print(f"Best CV F1 Score: {successful_models[best_model_name]['cv_mean']:.4f}")

        # Detailed evaluation of best model
        print(f"\n=== DETAILED EVALUATION - {best_model_name} ===")
        if successful_models[best_model_name]['test_predictions'] is not None:
            best_pred = successful_models[best_model_name]['test_predictions']
            print(classification_report(y_test, best_pred))
        else:
            print("No test predictions available for the best model.")


        # Feature importance for tree-based models
        if self.best_model and hasattr(self.best_model, 'feature_importances_') and not X.empty:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.feature_importance = feature_importance
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        elif self.best_model and hasattr(self.best_model, 'coef_') and not X.empty:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(self.best_model.coef_[0])
            }).sort_values('importance', ascending=False)
            self.feature_importance = feature_importance
            print("\nTop 10 Most Important Features (based on coefficients):")
            print(feature_importance.head(10))
        else:
            print("Feature importance not available for the best model type or data is empty.")


        return results, X_test, y_test

    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning on the best model"""
        print("=== HYPERPARAMETER TUNING ===")

        if self.best_model is None or X.empty or y.empty:
            print("Please train models first or provide non-empty data for tuning.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model_type_name = type(self.best_model).__name__
        param_grid = None

        if "RandomForestClassifier" in model_type_name:
            param_grid = HYPERPARAMETER_GRIDS.get('Random Forest')
        elif "XGBClassifier" in model_type_name:
            param_grid = HYPERPARAMETER_GRIDS.get('XGBoost')
        else:
            print(f"Hyperparameter tuning not implemented or configured for {model_type_name}")
            return None

        if param_grid:
            try:
                grid_search = GridSearchCV(
                    self.best_model,
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )

                print("Performing grid search...")
                grid_search.fit(X_train, y_train)

                self.best_model = grid_search.best_estimator_

                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")

                y_pred = self.best_model.predict(X_test)
                print("\nTuned Model Performance:")
                print(classification_report(y_test, y_pred))

                if hasattr(self.best_model, 'feature_importances_') and not X.empty:
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': self.best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importance = feature_importance
                    print("\nTop 10 Most Important Features (after tuning):")
                    print(feature_importance.head(10))
                elif hasattr(self.best_model, 'coef_') and not X.empty:
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.abs(self.best_model.coef_[0])
                    }).sort_values('importance', ascending=False)
                    self.feature_importance = feature_importance
                    print("\nTop 10 Most Important Features (after tuning, based on coefficients):")
                    print(feature_importance.head(10))
                else:
                    print("Feature importance not available for the tuned model type or data is empty.")

                return grid_search.best_params_
            except Exception as e:
                print(f"Error during hyperparameter tuning: {e}. Skipping tuning.")
                return None
        else:
            print("No hyperparameter grid found for the best model type.")
            return None

    def predict_retrofit_potential(self, X_transformed_data):
        """Predict retrofit potential for new building data"""
        if self.best_model is None:
            print("Please train models first.")
            return None, None
        if X_transformed_data.empty:
            print("Warning: Input data for prediction is empty.")
            return np.array([]), np.array([])

        predictions = self.best_model.predict(X_transformed_data)
        probabilities = self.best_model.predict_proba(X_transformed_data)[:, 1] if hasattr(self.best_model, 'predict_proba') else None

        return predictions, probabilities

    def get_best_model(self):
        return self.best_model

    def get_feature_importance(self):
        return self.feature_importance
