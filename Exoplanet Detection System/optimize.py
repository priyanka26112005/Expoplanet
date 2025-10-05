"""
PHASE 5: Advanced Model Optimization & Hyperparameter Tuning
Improve model accuracy through systematic optimization
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelOptimizer:
    """Optimize and fine-tune exoplanet detection models"""
    
    def __init__(self):
        self.best_models = {}
        self.optimization_results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        
        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"✅ Loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def optimize_random_forest(self, X_train, y_train, X_val, y_val):
        """Optimize Random Forest with Grid Search"""
        print("\n🌲 Optimizing Random Forest...")
        print("⏱️ This may take 10-15 minutes...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Initialize base model
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            rf_base, 
            param_grid, 
            cv=3, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Evaluate
        train_score = accuracy_score(y_train, best_rf.predict(X_train))
        val_score = accuracy_score(y_val, best_rf.predict(X_val))
        
        print(f"\n✅ Best Random Forest Parameters: {grid_search.best_params_}")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Validation Accuracy: {val_score:.4f}")
        
        self.best_models['random_forest'] = best_rf
        self.optimization_results['random_forest'] = {
            'best_params': grid_search.best_params_,
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        return best_rf
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost with Randomized Search"""
        print("\n🚀 Optimizing XGBoost...")
        print("⏱️ This may take 15-20 minutes...")
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [200, 300, 500, 700],
            'max_depth': [6, 8, 10, 12, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0.1, 0.5, 1, 2]
        }
        
        # Initialize base model
        xgb_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        
        # Randomized search (faster than grid search)
        random_search = RandomizedSearchCV(
            xgb_base,
            param_dist,
            n_iter=50,  # Try 50 random combinations
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        # Best model
        best_xgb = random_search.best_estimator_
        
        # Evaluate
        train_score = accuracy_score(y_train, best_xgb.predict(X_train))
        val_score = accuracy_score(y_val, best_xgb.predict(X_val))
        
        print(f"\n✅ Best XGBoost Parameters: {random_search.best_params_}")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Validation Accuracy: {val_score:.4f}")
        
        self.best_models['xgboost'] = best_xgb
        self.optimization_results['xgboost'] = {
            'best_params': random_search.best_params_,
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        return best_xgb
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM"""
        print("\n⚡ Optimizing LightGBM...")
        print("⏱️ This may take 10-15 minutes...")
        
        param_dist = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70, 100],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
        
        random_search = RandomizedSearchCV(
            lgbm_base,
            param_dist,
            n_iter=40,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        best_lgbm = random_search.best_estimator_
        
        train_score = accuracy_score(y_train, best_lgbm.predict(X_train))
        val_score = accuracy_score(y_val, best_lgbm.predict(X_val))
        
        print(f"\n✅ Best LightGBM Parameters: {random_search.best_params_}")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Validation Accuracy: {val_score:.4f}")
        
        self.best_models['lightgbm'] = best_lgbm
        self.optimization_results['lightgbm'] = {
            'best_params': random_search.best_params_,
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        return best_lgbm
    
    def create_super_ensemble(self, X_train, y_train, X_val, y_val):
        """Create optimized ensemble of best models"""
        print("\n🎯 Creating Super Ensemble...")
        
        # Use optimized models
        rf = self.best_models.get('random_forest')
        xgb = self.best_models.get('xgboost')
        lgbm = self.best_models.get('lightgbm')
        
        # Create weighted voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb),
                ('lgbm', lgbm)
            ],
            voting='soft',  # Use probability predictions
            weights=[1, 2, 1.5]  # Give XGBoost slightly more weight
        )
        
        print("   Training super ensemble...")
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        train_score = accuracy_score(y_train, ensemble.predict(X_train))
        val_score = accuracy_score(y_val, ensemble.predict(X_val))
        
        print(f"\n✅ Super Ensemble Performance:")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Validation Accuracy: {val_score:.4f}")
        
        self.best_models['super_ensemble'] = ensemble
        self.optimization_results['super_ensemble'] = {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        return ensemble
    
    def final_evaluation(self, X_test, y_test):
        """Comprehensive evaluation on test set"""
        print("\n" + "="*70)
        print("📊 FINAL TEST SET EVALUATION")
        print("="*70)
        
        results = {}
        
        for name, model in self.best_models.items():
            print(f"\n🔍 Evaluating {name.upper()}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1 Score: {f1:.4f}")
            
            # Classification report
            print("\n   Classification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']))
            
            # Confusion matrix
            print("\n   Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {best_model_name.upper()}")
        print(f"   Final Test Accuracy: {best_accuracy:.4f}")
        print("="*70)
        
        return results, best_model_name
    
    def save_optimized_models(self):
        """Save all optimized models"""
        print("\n💾 Saving optimized models...")
        
        import os
        os.makedirs('models/optimized', exist_ok=True)
        
        for name, model in self.best_models.items():
            filename = f'models/optimized/{name}_optimized.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✅ Saved {name}")
        
        # Save optimization results
        with open('models/optimized/optimization_results.pkl', 'wb') as f:
            pickle.dump(self.optimization_results, f)
        
        print("\n✅ All optimized models saved!")
    
    def run_full_optimization(self):
        """Run complete optimization pipeline"""
        print("="*70)
        print("🚀 STARTING FULL MODEL OPTIMIZATION")
        print("="*70)
        print("\n⏱️ Total estimated time: 40-60 minutes")
        print("   (Grab a coffee! ☕)\n")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Optimize individual models
        self.optimize_random_forest(X_train, y_train, X_val, y_val)
        self.optimize_xgboost(X_train, y_train, X_val, y_val)
        self.optimize_lightgbm(X_train, y_train, X_val, y_val)
        
        # Create super ensemble
        self.create_super_ensemble(X_train, y_train, X_val, y_val)
        
        # Final evaluation
        results, best_model = self.final_evaluation(X_test, y_test)
        
        # Save models
        self.save_optimized_models()
        
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE!")
        print("="*70)
        
        return results, best_model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Run full optimization
    results, best_model = optimizer.run_full_optimization()
    
    print("\n🎉 Ready for Phase 6: Web Interface Development!")