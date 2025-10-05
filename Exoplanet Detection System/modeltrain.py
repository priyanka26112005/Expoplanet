"""
STEP 4: Train Multiple Machine Learning Models
Train and compare different models for exoplanet detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb  # <- needed for callbacks like log_evaluation, early_stopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json
import warnings
import os
warnings.filterwarnings('ignore')


class ExoplanetModelTrainer:
    """
    Train multiple models and create an ensemble for exoplanet classification
    """

    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.models = {}
        self.ensemble = None
        self.best_model = None
        self.best_model_name = None
        self.training_history = {}
        self.results = {}

    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")

        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
        y_test = np.load('data/y_test.npy')

        print(f"✅ Data loaded successfully!")
        print(f"   Train: {X_train.shape}")
        print(f"   Val:   {X_val.shape}")
        print(f"   Test:  {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest classifier"""
        print("\n" + "=" * 70)
        print("🌲 TRAINING RANDOM FOREST")
        print("=" * 70)

        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=0
        )

        print("Training...")
        rf_model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = rf_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        self.models['random_forest'] = rf_model

        print(f"✅ Random Forest trained!")
        print(f"   Validation Accuracy: {accuracy:.4f}")

        # Feature importance if X_train is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            feature_imp = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n📊 Top 10 Important Features:")
            for i, row in feature_imp.head(10).iterrows():
                print(f"   {row['feature']:30s}: {row['importance']:.4f}")

        return rf_model

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost classifier"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING XGBOOST")
        print("=" * 70)

        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0
        )

        print("Training...")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = xgb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        self.models['xgboost'] = xgb_model

        print(f"✅ XGBoost trained!")
        print(f"   Validation Accuracy: {accuracy:.4f}")

        return xgb_model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM classifier"""
        print("\n" + "=" * 70)
        print("💡 TRAINING LIGHTGBM")
        print("=" * 70)

        lgbm_model = LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            class_weight='balanced'
        )

        print("Training...")

        # Use lightgbm module callbacks (not model instance methods)
        callbacks = []
        try:
            # log evaluation every 100 rounds (0 would log every iteration and could be noisy)
            callbacks.append(lgb.log_evaluation(period=100))
            # early stopping after 50 rounds
            callbacks.append(lgb.early_stopping(stopping_rounds=50))
        except Exception:
            # If callbacks unsupported in older versions, continue without them
            callbacks = None

        fit_kwargs = {
            'X': X_train,
            'y': y_train,
            'eval_set': [(X_val, y_val)],
        }

        # scikit-learn wrapper uses .fit(X, y, eval_set=..., callbacks=...)
        if callbacks is not None:
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
        else:
            # fallback
            lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)]
            )

        y_pred = lgbm_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        self.models['lightgbm'] = lgbm_model

        print(f"✅ LightGBM trained!")
        print(f"   Validation Accuracy: {accuracy:.4f}")

        return lgbm_model

    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting classifier"""
        print("\n" + "=" * 70)
        print("📈 TRAINING GRADIENT BOOSTING")
        print("=" * 70)

        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        print("Training...")
        gb_model.fit(X_train, y_train)

        y_pred = gb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        self.models['gradient_boosting'] = gb_model

        print(f"✅ Gradient Boosting trained!")
        print(f"   Validation Accuracy: {accuracy:.4f}")

        return gb_model

    def create_neural_network(self, input_dim):
        """Create a deep learning model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train deep learning model"""
        print("\n" + "=" * 70)
        print("🧠 TRAINING NEURAL NETWORK")
        print("=" * 70)

        nn_model = self.create_neural_network(X_train.shape[1])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )

        print("Training...")
        history = nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        self.models['neural_network'] = nn_model
        self.training_history['neural_network'] = history.history

        # Get validation accuracy
        y_pred_proba = nn_model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_val, y_pred)

        print(f"✅ Neural Network trained!")
        print(f"   Best Validation Accuracy: {max(history.history.get('val_accuracy', [0])):.4f}")
        print(f"   Final Validation Accuracy: {accuracy:.4f}")
        print(f"   Epochs trained: {len(history.history['loss'])}")

        return nn_model

    def create_ensemble(self, X_train, y_train):
        """Create voting ensemble from trained models"""
        print("\n" + "=" * 70)
        print("🤝 CREATING ENSEMBLE MODEL")
        print("=" * 70)

        # Ensure required models exist
        required = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        for r in required:
            if r not in self.models:
                raise ValueError(f"Required model '{r}' not trained; cannot build ensemble.")

        estimators = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('lgbm', self.models['lightgbm']),
            ('gb', self.models['gradient_boosting'])
        ]

        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )

        print("Training ensemble...")
        # Fit ensemble on training data
        self.ensemble.fit(X_train, y_train)

        print(f"✅ Ensemble created with {len(estimators)} models")
        print(f"   Models: Random Forest, XGBoost, LightGBM, Gradient Boosting")

        return self.ensemble

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models and compare performance"""
        print("\n" + "=" * 70)
        print("📊 EVALUATING ALL MODELS ON TEST SET")
        print("=" * 70)

        results = {}

        # Load label mapping for display if exists
        inv_label_map = None
        try:
            with open('models/preprocessing_config.json', 'r') as f:
                config = json.load(f)
                label_map = config.get('label_mapping', {})
                inv_label_map = {int(v): k for k, v in label_map.items()}
        except Exception:
            # fallback to numeric labels if mapping missing
            inv_label_map = {i: str(i) for i in range(self.n_classes)}

        for name, model in self.models.items():
            print(f"\n{'=' * 70}")
            print(f"{name.upper().replace('_', ' ')}")
            print(f"{'=' * 70}")

            if name == 'neural_network':
                y_pred_proba = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(X_test)
                # some sklearn models may not implement predict_proba consistently; handle safely
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except Exception:
                    # try one-hot from predictions
                    y_pred_proba = np.eye(self.n_classes)[y_pred]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Try to calculate AUC (for multiclass)
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0

            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }

            print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            if auc > 0:
                print(f"AUC:       {auc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)

            # Classification Report
            target_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]
            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred,
                target_names=target_names,
                zero_division=0
            ))

        # Evaluate ensemble
        if self.ensemble:
            print(f"\n{'=' * 70}")
            print("ENSEMBLE (VOTING)")
            print(f"{'=' * 70}")

            y_pred_ensemble = self.ensemble.predict(X_test)
            try:
                y_pred_proba_ensemble = self.ensemble.predict_proba(X_test)
            except Exception:
                y_pred_proba_ensemble = np.eye(self.n_classes)[y_pred_ensemble]

            accuracy = accuracy_score(y_test, y_pred_ensemble)
            precision = precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)

            try:
                auc = roc_auc_score(y_test, y_pred_proba_ensemble, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0

            results['ensemble'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }

            print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            if auc > 0:
                print(f"AUC:       {auc:.4f}")

            cm = confusion_matrix(y_test, y_pred_ensemble)
            print("\nConfusion Matrix:")
            print(cm)

            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred_ensemble,
                target_names=[inv_label_map[i] for i in sorted(inv_label_map.keys())],
                zero_division=0
            ))

        # Find best model
        print("\n" + "=" * 70)
        print("🏆 MODEL COMPARISON")
        print("=" * 70)

        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('f1_score', ascending=False)

        print("\n" + results_df.to_string())

        best_model_name = results_df.index[0]
        self.best_model_name = best_model_name
        self.best_model = self.models.get(best_model_name) or self.ensemble
        self.results = results

        print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
        print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
        print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")

        return results

    def plot_results(self):
        """Visualize model comparison"""
        print("\n📊 Generating visualizations...")

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        models = list(self.results.keys())
        # ensure colors list matches length (repeat if needed)
        default_colors = ['#4ade80', '#fbbf24', '#60a5fa', '#f472b6', '#a78bfa']
        colors = (default_colors * ((len(models) // len(default_colors)) + 1))[:len(models)]

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]

            values = [self.results[m][metric] for m in models]

            bars = ax.bar(models, values, color=colors)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_ylim([0, 1.0])
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        os.makedirs('.', exist_ok=True)
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: model_comparison.png")

        plt.close()

    def save_models(self, output_dir='models'):
        """Save all trained models"""
        print(f"\n💾 Saving models to '{output_dir}/'...")
        os.makedirs(output_dir, exist_ok=True)

        # Save sklearn models
        for name, model in self.models.items():
            if name != 'neural_network':
                joblib.dump(model, f'{output_dir}/{name}.pkl')
                print(f"   ✅ Saved {name}.pkl")

        # Save neural network
        if 'neural_network' in self.models:
            # Keras model .save will create the file
            self.models['neural_network'].save(f'{output_dir}/neural_network.h5')
            print(f"   ✅ Saved neural_network.h5")

        # Save ensemble
        if self.ensemble:
            joblib.dump(self.ensemble, f'{output_dir}/ensemble.pkl')
            print(f"   ✅ Saved ensemble.pkl")

        # Save best model reference and results
        best_model_info = {
            'best_model_name': self.best_model_name,
            'results': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in self.results.items()}
        }

        with open(f'{output_dir}/model_results.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)

        print(f"   ✅ Saved model_results.json")

    def train_all(self, X_train, y_train, X_val, y_val):
        """Train all models"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING ALL MODELS")
        print("=" * 70)

        # Train individual models
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_gradient_boosting(X_train, y_train, X_val, y_val)
        self.train_neural_network(X_train, y_train, X_val, y_val)

        # Create ensemble
        self.create_ensemble(X_train, y_train)

        print("\n" + "=" * 70)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("NASA EXOPLANET DETECTION - MODEL TRAINING")
    print("=" * 70)

    # Initialize trainer
    trainer = ExoplanetModelTrainer(n_classes=3)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()

    # Train all models
    trainer.train_all(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    results = trainer.evaluate_all_models(X_test, y_test)

    # Plot results
    trainer.plot_results()

    # Save models
    trainer.save_models('models')

    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n🎉 All models trained, evaluated, and saved!")
    if trainer.best_model_name:
        print(f"🏆 Best model: {trainer.best_model_name.upper()}")
    else:
        print("🏆 Best model: (not determined)")
    print("\n📁 Saved files (in 'models/'):")
    print("   - random_forest.pkl")
    print("   - xgboost.pkl")
    print("   - lightgbm.pkl")
    print("   - gradient_boosting.pkl")
    print("   - neural_network.h5")
    print("   - ensemble.pkl")
    print("   - model_results.json")
    print("   - model_comparison.png")
    print("\n🚀 Next step: Build web interface!")
