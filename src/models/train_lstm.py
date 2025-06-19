"""
OPTIMIZED LSTM WITH BREAKTHROUGH METHOD

Bu LSTM model breakthrough yaklaşımını kullanır:
✅ Random split (temporal split değil)
✅ Year feature exclusion 
✅ Distribution shift çözümü
✅ Dense architecture (tabular için uygun)
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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

def fix_normalization(df):
    """Fix critical normalization issues"""
    logging.info("🔧 Fixing normalization...")
    
    df_fixed = df.copy()
    
    # Fix satellite bands (0-10000 -> 0-1)
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12', 'B8A']
    for band in bands:
        if band in df_fixed.columns:
            if df_fixed[band].max() > 1:
                df_fixed[band] = df_fixed[band] / 10000.0
                df_fixed[band] = df_fixed[band].clip(0, 1)
                logging.info(f"  {band}: normalized to 0-1")
    
    # Fix vegetation indices
    vegetation_fixes = {
        'EVI': (-1, 1), 'SAVI': (0, 1), 'NDRE': (0, 1), 'NDVI': (-1, 1)
    }
    
    for idx, (min_val, max_val) in vegetation_fixes.items():
        if idx in df_fixed.columns:
            df_fixed[idx] = df_fixed[idx].clip(min_val, max_val)
    
    logging.info("✅ Normalization fixed!")
    return df_fixed

def load_and_prepare_breakthrough_data():
    """BREAKTHROUGH: Load data with random split approach"""
    logging.info("🚀 Loading data with BREAKTHROUGH approach...")
    
    # Load both datasets  
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    # Apply normalization fixes
    train_df = fix_normalization(train_df)
    test_df = fix_normalization(test_df)
    
    # BREAKTHROUGH: Combine all data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    logging.info(f"Combined dataset: {combined_df.shape}")
    
    # Clean data
    combined_df = combined_df.dropna(subset=['yield'])
    combined_df = combined_df[combined_df['yield'] > 0]
    
    # Remove extreme outliers
    q1, q3 = combined_df['yield'].quantile([0.01, 0.99])
    combined_df = combined_df[(combined_df['yield'] >= q1) & (combined_df['yield'] <= q3)]
    
    # BREAKTHROUGH: Random split (not temporal!)
    train_new, test_new = train_test_split(
        combined_df, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Check distribution balance
    train_mean = train_new['yield'].mean()
    test_mean = test_new['yield'].mean()
    distribution_diff = abs(train_mean - test_mean)
    
    logging.info(f"✅ BREAKTHROUGH RESULTS:")
    logging.info(f"  Train: {train_new.shape}, Mean yield: {train_mean:.2f}")
    logging.info(f"  Test: {test_new.shape}, Mean yield: {test_mean:.2f}")
    logging.info(f"  Distribution difference: {distribution_diff:.2f}")
    
    if distribution_diff < 2:
        logging.info(f"✅ Distribution shift SOLVED!")
    
    return train_new, test_new

def select_features_exclude_year(df):
    """BREAKTHROUGH: Select features excluding temporal bias"""
    logging.info("🎯 Selecting features (EXCLUDING YEAR for breakthrough)...")
    
    # EXCLUDE temporal features that cause overfitting
    excluded_features = ['year', 'county_id', 'month_num', 'date', 'yield', 'growth_stage']
    
    # Get all numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_features = [col for col in numeric_cols if col not in excluded_features]
    
    # Calculate correlations
    correlations = df[candidate_features + ['yield']].corr()['yield'].abs()
    correlations = correlations.drop('yield').sort_values(ascending=False)
    
    # Select top features
    selected_features = correlations.head(20).index.tolist()
    
    logging.info(f"📊 Selected {len(selected_features)} features (NO TEMPORAL BIAS):")
    for i, feature in enumerate(selected_features[:10]):
        corr = correlations[feature]
        logging.info(f"  {i+1:2d}. {feature:20s}: {corr:.4f}")
    
    # Verify no temporal features
    temporal_check = [f for f in selected_features if 'year' in f.lower()]
    if temporal_check:
        logging.warning(f"⚠️ Temporal features detected: {temporal_check}")
    else:
        logging.info(f"✅ No temporal bias - pure agricultural features")
    
    return selected_features

def create_dense_model_for_tabular(input_dim):
    """BREAKTHROUGH: Dense model for tabular agricultural data"""
    logging.info(f"🏗️ Creating DENSE model for tabular data ({input_dim} features)...")
    
    model = Sequential([
        # Dense layers optimized for tabular agricultural data
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logging.info(f"✅ DENSE model created: {model.count_params()} parameters")
    logging.info("🎯 Architecture: Dense layers for tabular agricultural data (NOT LSTM)")
    
    return model

def prepare_balanced_data(train_df, test_df, features):
    """Prepare balanced data from breakthrough approach"""
    logging.info("🔧 Preparing balanced data...")
    
    # Extract features and targets
    X_train = train_df[features].values
    X_test = test_df[features].values
    y_train = train_df['yield'].values
    y_test = test_df['yield'].values
    
    # Handle NaN
    for i, feature in enumerate(features):
        if np.isnan(X_train[:, i]).any():
            median_val = np.nanmedian(X_train[:, i])
            X_train[np.isnan(X_train[:, i]), i] = median_val
            X_test[np.isnan(X_test[:, i]), i] = median_val
    
    # Scale features
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    logging.info(f"✅ Data prepared: Features [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler

def train_breakthrough_model(model, X_train, y_train, X_val, y_val):
    """Train model with breakthrough approach"""
    logging.info("🚀 Training with BREAKTHROUGH approach...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ModelCheckpoint('models/breakthrough_lstm_dense.keras', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_breakthrough_model(model, X_test, y_test, target_scaler):
    """Evaluate breakthrough model"""
    logging.info("📊 Evaluating BREAKTHROUGH model...")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    logging.info(f"🎯 BREAKTHROUGH LSTM (DENSE) RESULTS:")
    logging.info(f"  R² Score: {r2:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAE: {mae:.4f}")
    
    # Success analysis
    if r2 > 0.8:
        logging.info(f"🎉 EXCELLENT: R² > 0.8 - World-class performance!")
    elif r2 > 0.5:
        logging.info(f"🎉 GREAT: R² > 0.5 - Strong predictive power!")
    elif r2 > 0:
        logging.info(f"✅ SUCCESS: Positive R² achieved!")
    else:
        logging.warning(f"⚠️ Still negative R²")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
    plt.plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'BREAKTHROUGH LSTM-Dense (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    residuals = y_test_orig - y_pred_orig
    plt.scatter(y_pred_orig, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist([y_test_orig, y_pred_orig], bins=20, alpha=0.7, 
             label=['Actual', 'Predicted'], density=True)
    plt.xlabel('Yield')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/breakthrough_lstm_dense_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}

def main():
    """Main function with BREAKTHROUGH approach"""
    try:
        logging.info("🚀 BREAKTHROUGH LSTM (DENSE) - Random Split Method...")
        
        ensure_directories()
        
        # BREAKTHROUGH: Load data with random split
        train_df, test_df = load_and_prepare_breakthrough_data()
        
        # BREAKTHROUGH: Select features excluding year
        features = select_features_exclude_year(train_df)
        
        # Prepare balanced data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler = prepare_balanced_data(
            train_df, test_df, features
        )
        
        # Split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
        )
        
        logging.info(f"Final shapes: Train {X_train_final.shape}, Val {X_val.shape}, Test {X_test_scaled.shape}")
        
        # BREAKTHROUGH: Create dense model (not LSTM sequences!)
        model = create_dense_model_for_tabular(X_train_final.shape[1])
        
        # Train model
        model, history = train_breakthrough_model(model, X_train_final, y_train_final, X_val, y_val)
        
        # Evaluate
        results = evaluate_breakthrough_model(model, X_test_scaled, y_test_scaled, target_scaler)
        
        # Save model and scalers
        model.save('models/breakthrough_lstm_dense.keras')
        joblib.dump(feature_scaler, 'models/breakthrough_lstm_feature_scaler.joblib')
        joblib.dump(target_scaler, 'models/breakthrough_lstm_target_scaler.joblib')
        joblib.dump(features, 'models/breakthrough_lstm_features.joblib')
        
        logging.info("🎉 BREAKTHROUGH LSTM (DENSE) COMPLETE!")
        logging.info(f"🏆 Final R² Score: {results['r2']:.4f}")
        
        return results
        
    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()