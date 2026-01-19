"""
Trade Surveillance ML Model Training
XGBoost classifier for alert risk scoring
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, average_precision_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class SurveillanceModel:
    """ML Model for surveillance alert classification"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        
    def prepare_features(self, df):
        """Extract numeric features for model training"""
        # Identify feature columns (exclude metadata and label)
        exclude_cols = ['alert_id', 'timestamp', 'trader_id', 'symbol', 
                       'asset_class', 'is_violation']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns]
        y = df['is_violation'] if 'is_violation' in df.columns else None
        
        return X, y
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train XGBoost model"""
        print(" Training XGBoost Alert Classification Model\n")
        
        # Prepare data
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f" Training set: {len(X_train)} alerts")
        print(f" Test set: {len(X_test)} alerts")
        print(f"   - Violations: {y_test.sum()} ({y_test.mean()*100:.1f}%)\n")
        
        # Configure XGBoost with optimized hyperparameters
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=3,
            scale_pos_weight=5.67,  # Class imbalance (850/150)
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Train model
        print("⚙️  Training model...")
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        print("\n 5-Fold Cross-Validation:")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        print(f"   CV F1 Scores: {cv_scores}")
        print(f"   Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Evaluate on test set
        print("\n Test Set Performance:")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Classification metrics
        print(classification_report(y_test, y_pred, target_names=['False Positive', 'Violation']))
        
        # Additional metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n Advanced Metrics:")
        print(f"   ROC-AUC Score: {roc_auc:.4f}")
        print(f"   Average Precision: {avg_precision:.4f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n Top 5 Most Important Features:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n Confusion Matrix:")
        print(f"   True Negatives:  {cm[0,0]}")
        print(f"   False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}")
        print(f"   True Positives:  {cm[1,1]}")
        
        # Calculate metrics for ROI
        false_positive_reduction = 1 - (cm[0,1] / (cm[0,0] + cm[0,1]))
        print(f"\n Business Impact:")
        print(f"   False Positive Reduction: {false_positive_reduction*100:.1f}%")
        print(f"   Estimated Time Savings: ~{false_positive_reduction*40:.0f}% reduction in review time")
        
        return {
            'model': self.model,
            'test_accuracy': (cm[0,0] + cm[1,1]) / cm.sum(),
            'precision': cm[1,1] / (cm[1,1] + cm[0,1]),
            'recall': cm[1,1] / (cm[1,1] + cm[1,0]),
            'f1_score': 2 * (cm[1,1] / (cm[1,1] + cm[0,1])) * (cm[1,1] / (cm[1,1] + cm[1,0])) / 
                       ((cm[1,1] / (cm[1,1] + cm[0,1])) + (cm[1,1] / (cm[1,1] + cm[1,0]))),
            'roc_auc': roc_auc,
            'feature_importance': self.feature_importance
        }
    
    def predict_risk_score(self, df):
        """Generate 0-100 risk scores for new alerts"""
        X, _ = self.prepare_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        risk_scores = (probabilities * 100).round(0).astype(int)
        return risk_scores
    
    def save_model(self, filepath='models/surveillance_model.pkl'):
        """Save trained model to disk"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"\n Model saved to: {filepath}")
    
    def load_model(self, filepath='models/surveillance_model.pkl'):
        """Load trained model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.feature_importance = data['feature_importance']
        print(f" Model loaded from: {filepath}")


def plot_feature_importance(feature_importance, save_path='diagrams/feature_importance.png'):
    """Visualize top features"""
    plt.figure(figsize=(10, 6))
    top_10 = feature_importance.head(10)
    sns.barplot(data=top_10, x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance - XGBoost Alert Classifier', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f" Feature importance plot saved: {save_path}")


if __name__ == "__main__":
    # Load training data
    df = pd.read_csv('data/surveillance_alerts.csv')
    
    # Train model
    model = SurveillanceModel()
    results = model.train(df)
    
    # Save model
    model.save_model('models/surveillance_model.pkl')
    
    # Plot feature importance
    plot_feature_importance(results['feature_importance'], 
                           'diagrams/feature_importance.png')
    
    # Demo: Score some sample alerts
    print("\n Demo - Scoring Sample Alerts:")
    sample_alerts = df.sample(5)
    risk_scores = model.predict_risk_score(sample_alerts)
    
    for idx, (_, alert) in enumerate(sample_alerts.iterrows()):
        print(f"\nAlert ID: {alert['alert_id']}")
        print(f"  Symbol: {alert['symbol']}")
        print(f"  Order-to-Trade Ratio: {alert['order_to_trade_ratio']:.1f}")
        print(f"  Trade Velocity: {alert['trade_velocity']:.1f}/min")
        print(f"  RISK SCORE: {risk_scores[idx]}/100")
        print(f"  Actual: {'VIOLATION' if alert['is_violation'] else 'False Positive'}")
