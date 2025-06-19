"""
Fixed Neural Network for Agricultural Yield Prediction

Bu model NaN loss problemini çözer:
1. Robust scaling strategy
2. Conservative learning rate
3. Simplified architecture  
4. Better data validation
5. Progressive training approach
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU available and configured")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

def ensure_directories():
    """Create necessary directories"""
    directories = ['reports/figures', 'models', 'data/processed']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_and_validate_data():
    """Load and validate data with extensive checks"""
    logging.info("Loading and validating data...")
    
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    logging.info(f"Original - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Remove rows with missing yield (most critical)
    initial_train = len(train_df)
    initial_test = len(test_df)
    
    train_df = train_df.dropna(subset=['yield'])
    test_df = test_df.dropna(subset=['yield'])
    
    logging.info(f"Removed rows without yield - Train: {initial_train - len(train_df)}, Test: {initial_test - len(test_df)}")
    
    # Validate yield values
    train_df = train_df[train_df['yield'] > 0]
    test_df = test_df[test_df['yield'] > 0]
    
    # Remove extreme outliers (keep 95% of data)
    train_q1, train_q3 = train_df['yield'].quantile([0.025, 0.975])
    test_q1, test_q3 = test_df['yield'].quantile([0.025, 0.975])
    
    train_df = train_df[(train_df['yield'] >= train_q1) & (train_df['yield'] <= train_q3)]
    test_df = test_df[(test_df['yield'] >= test_q1) & (test_df['yield'] <= test_q3)]
    
    logging.info(f"After validation - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Check target statistics
    logging.info(f"Train yield: {train_df['yield'].mean():.2f} ± {train_df['yield'].std():.2f}")
    logging.info(f"Test yield: {test_df['yield'].mean():.2f} ± {test_df['yield'].std():.2f}")
    
    return train_df, test_df

def select_and_validate_features(train_df, top_k=15):
    """Select and validate features with extensive NaN checking"""
    logging.info("Selecting and validating features...")
    
    # Get seasonal features first (they're proven to work)
    seasonal_features = []
    for stage in ['early_', 'mid_', 'late_']:
        for metric in ['NDVI', 'SAVI', 'NDRE', 'temp_avg', 'precipitation', 'water_stress']:
            col_name = f"{stage}{metric}"
            if col_name in train_df.columns:
                seasonal_features.append(col_name)
    
    logging.info(f"Found {len(seasonal_features)} seasonal features")
    
    # Calculate correlations only for seasonal features
    if seasonal_features:
        correlations = train_df[seasonal_features + ['yield']].corr()['yield'].abs()
        correlations = correlations.drop('yield').sort_values(ascending=False)
        
        selected_features = correlations.head(top_k).index.tolist()
    else:
        # Fallback to any numeric features
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['county_id', 'year', 'month_num', 'yield']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        correlations = train_df[feature_cols + ['yield']].corr()['yield'].abs()
        correlations = correlations.drop('yield').sort_values(ascending=False)
        selected_features = correlations.head(top_k).index.tolist()
    
    # Validate features for NaN
    valid_features = []
    for feature in selected_features:
        train_nan_pct = train_df[feature].isna().mean()
        if train_nan_pct < 0.1:  # Less than 10% missing
            valid_features.append(feature)
            logging.info(f"  ✅ {feature:25s}: corr={correlations[feature]:.4f}, NaN={train_nan_pct:.1%}")
        else:
            logging.info(f"  ❌ {feature:25s}: too many NaN ({train_nan_pct:.1%})")
    
    logging.info(f"Selected {len(valid_features)} valid features")
    return valid_features

def prepare_clean_data(train_df, test_df, features):
    """Prepare clean data matrices with extensive validation"""
    logging.info("Preparing clean data matrices...")
    
    # Extract feature matrices
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df['yield'].values
    y_test = test_df['yield'].values
    
    logging.info(f"Initial shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Fill any remaining NaN with median (conservative approach)
    for feature in features:
        if X_train[feature].isna().any():
            median_val = X_train[feature].median()
            X_train[feature] = X_train[feature].fillna(median_val)
            X_test[feature] = X_test[feature].fillna(median_val)
            logging.info(f"Filled NaN in {feature} with median: {median_val:.4f}")
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    
    # Final validation - check for any remaining NaN or infinite values
    train_nan_count = np.isnan(X_train).sum()
    test_nan_count = np.isnan(X_test).sum()
    train_inf_count = np.isinf(X_train).sum()
    test_inf_count = np.isinf(X_test).sum()
    
    if train_nan_count > 0 or test_nan_count > 0:
        logging.error(f"Still have NaN values! Train: {train_nan_count}, Test: {test_nan_count}")
        return None, None, None, None
    
    if train_inf_count > 0 or test_inf_count > 0:
        logging.error(f"Have infinite values! Train: {train_inf_count}, Test: {test_inf_count}")
        return None, None, None, None
    
    # Target validation
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        logging.error("Target values contain NaN!")
        return None, None, None, None
    
    logging.info("✅ Data validation passed - no NaN or infinite values")
    logging.info(f"Feature ranges: X_train [{X_train.min():.4f}, {X_train.max():.4f}]")
    logging.info(f"Target ranges: y_train [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    return X_train, X_test, y_train, y_test

def safe_scaling(X_train, X_test, y_train, y_test):
    """Apply safe, conservative scaling"""
    logging.info("Applying safe scaling...")
    
    # Use MinMaxScaler for features (more conservative than StandardScaler)
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Simple standardization for targets
    target_mean = y_train.mean()
    target_std = y_train.std()
    
    y_train_scaled = (y_train - target_mean) / target_std
    y_test_scaled = (y_test - target_mean) / target_std
    
    # Final validation after scaling
    if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
        logging.error("NaN values after feature scaling!")
        return None, None, None, None, None, None, None
    
    if np.isnan(y_train_scaled).any() or np.isnan(y_test_scaled).any():
        logging.error("NaN values after target scaling!")
        return None, None, None, None, None, None, None
    
    logging.info("✅ Scaling completed successfully")
    logging.info(f"Scaled features: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    logging.info(f"Scaled targets: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_mean, target_std

def create_conservative_model(input_dim):
    """Create conservative model architecture"""
    logging.info(f"Creating conservative model with {input_dim} features...")
    
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    # Very conservative learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Much lower learning rate
        loss='mse',  # Simple MSE instead of Huber
        metrics=['mae']
    )
    
    logging.info(f"Conservative model created with {model.count_params()} parameters")
    logging.info("Model features:")
    logging.info("  - Small architecture to prevent overfitting")
    logging.info("  - Very low learning rate (0.0001)")
    logging.info("  - Simple MSE loss")
    logging.info("  - Conservative dropout")
    
    return model

def train_with_validation(model, X_train, y_train, X_val, y_val):
    """Train with extensive validation"""
    logging.info("Training with validation...")
    
    # Check inputs before training
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        logging.error("NaN in training data!")
        return None, None
    
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        logging.error("NaN in validation data!")
        return None, None
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint('models/fixed_nn.keras', monitor='val_loss', save_best_only=True)
    ]
    
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        logging.info("Training completed successfully")
        return model, history
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return None, None

def evaluate_fixed_model(model, X_test, y_test, target_mean, target_std):
    """Evaluate with inverse scaling"""
    logging.info("Evaluating fixed model...")
    
    try:
        # Make predictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Inverse scale
        y_test_orig = y_test * target_std + target_mean
        y_pred_orig = y_pred_scaled.flatten() * target_std + target_mean
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        logging.info(f"🎯 Fixed Model Performance:")
        logging.info(f"  R² Score: {r2:.4f}")
        logging.info(f"  RMSE: {rmse:.4f} bushels/acre")
        logging.info(f"  MAE: {mae:.4f} bushels/acre")
        
        # Simple plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
        plt.plot([y_test_orig.min(), y_test_orig.max()], 
                [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
        plt.xlabel('Actual Yield (bushels/acre)')
        plt.ylabel('Predicted Yield (bushels/acre)')
        plt.title(f'Fixed NN Predictions (R² = {r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        residuals = y_test_orig - y_pred_orig
        plt.scatter(y_pred_orig, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Yield')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/fixed_nn_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        return None

def main():
    """Main training function with extensive error handling"""
    try:
        logging.info("🔧 Starting FIXED neural network training...")
        
        ensure_directories()
        
        # Load and validate data
        train_df, test_df = load_and_validate_data()
        if train_df is None or test_df is None:
            logging.error("Data loading failed!")
            return None
        
        # Select features
        features = select_and_validate_features(train_df, top_k=15)
        if not features:
            logging.error("No valid features found!")
            return None
        
        # Prepare clean data
        X_train, X_test, y_train, y_test = prepare_clean_data(train_df, test_df, features)
        if X_train is None:
            logging.error("Data preparation failed!")
            return None
        
        # Safe scaling
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_mean, target_std = safe_scaling(
            X_train, X_test, y_train, y_test
        )
        if X_train_scaled is None:
            logging.error("Scaling failed!")
            return None
        
        # Split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
        )
        
        logging.info(f"Final data shapes:")
        logging.info(f"  Train: {X_train_final.shape}")
        logging.info(f"  Val: {X_val.shape}")
        logging.info(f"  Test: {X_test_scaled.shape}")
        
        # Create conservative model
        model = create_conservative_model(X_train_final.shape[1])
        
        # Train with validation
        model, history = train_with_validation(model, X_train_final, y_train_final, X_val, y_val)
        if model is None:
            logging.error("Training failed!")
            return None
        
        # Evaluate
        results = evaluate_fixed_model(model, X_test_scaled, y_test_scaled, target_mean, target_std)
        if results is None:
            logging.error("Evaluation failed!")
            return None
        
        # Save everything
        model.save('models/fixed_nn_model.keras')
        joblib.dump(feature_scaler, 'models/fixed_nn_feature_scaler.joblib')
        joblib.dump({'mean': target_mean, 'std': target_std}, 'models/fixed_nn_target_stats.joblib')
        joblib.dump(features, 'models/fixed_nn_features.joblib')
        
        logging.info("🎉 Fixed neural network training completed successfully!")
        logging.info(f"🏆 Final R² Score: {results['r2']:.4f}")
        
        if results['r2'] > 0:
            logging.info("✅ SUCCESS: Achieved positive R²!")
        else:
            logging.info(f"⚠️  R² still negative, but training completed without NaN issues")
        
        return results
        
    except Exception as e:
        logging.error(f"❌ Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 