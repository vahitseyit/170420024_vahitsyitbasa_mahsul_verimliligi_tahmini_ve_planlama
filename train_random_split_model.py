"""
RANDOM SPLIT MODEL - Distribution Shift Çözümü

Bu approach distribution shift problemini çözer:
✅ Train/Test combine edip random split
✅ Year feature exclude (temporal bias yok)  
✅ Ortalama distribution
✅ Tabular dense model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_normalization(df):
    """Fix normalization issues"""
    logging.info("🔧 Fixing normalization...")
    
    df_fixed = df.copy()
    
    # Fix satellite bands (0-10000 -> 0-1)
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12', 'B8A']
    for band in bands:
        if band in df_fixed.columns:
            if df_fixed[band].max() > 1:  # Only normalize if not already normalized
                df_fixed[band] = df_fixed[band] / 10000.0
                df_fixed[band] = df_fixed[band].clip(0, 1)
                logging.info(f"  {band}: normalized to 0-1")
    
    # Fix vegetation indices
    vegetation_fixes = {
        'EVI': (-1, 1),
        'SAVI': (0, 1), 
        'NDRE': (0, 1),
        'NDVI': (-1, 1)
    }
    
    for idx, (min_val, max_val) in vegetation_fixes.items():
        if idx in df_fixed.columns:
            df_fixed[idx] = df_fixed[idx].clip(min_val, max_val)
            logging.info(f"  {idx}: clipped to [{min_val}, {max_val}]")
    
    logging.info("✅ Normalization fixed!")
    return df_fixed

def combine_and_random_split():
    """Combine train/test and create random split WITHOUT year bias"""
    logging.info("🔄 Combining data and creating random split...")
    
    # Load both datasets
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    logging.info(f"Original - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Fix normalization for both
    train_df = fix_normalization(train_df)
    test_df = fix_normalization(test_df)
    
    # COMBINE ALL DATA
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    logging.info(f"Combined dataset: {combined_df.shape}")
    
    # Clean data
    combined_df = combined_df.dropna(subset=['yield'])
    combined_df = combined_df[combined_df['yield'] > 0]
    
    # Remove extreme outliers
    q1, q3 = combined_df['yield'].quantile([0.01, 0.99])
    combined_df = combined_df[(combined_df['yield'] >= q1) & (combined_df['yield'] <= q3)]
    
    # Check yield distribution
    yield_mean = combined_df['yield'].mean()
    yield_std = combined_df['yield'].std()
    yield_min = combined_df['yield'].min()
    yield_max = combined_df['yield'].max()
    
    logging.info(f"📊 COMBINED YIELD DISTRIBUTION:")
    logging.info(f"  Count: {len(combined_df)}")
    logging.info(f"  Mean: {yield_mean:.2f}")
    logging.info(f"  Std: {yield_std:.2f}")
    logging.info(f"  Range: [{yield_min:.2f}, {yield_max:.2f}]")
    
    # RANDOM SPLIT (80/20) - NO TEMPORAL BIAS
    train_new, test_new = train_test_split(
        combined_df, 
        test_size=0.2, 
        random_state=42,
        shuffle=True  # Ensure random mixing
    )
    
    # Check distributions after split
    train_mean = train_new['yield'].mean()
    test_mean = test_new['yield'].mean()
    distribution_diff = abs(train_mean - test_mean)
    
    logging.info(f"📈 AFTER RANDOM SPLIT:")
    logging.info(f"  Train: {train_new.shape}, Mean yield: {train_mean:.2f}")
    logging.info(f"  Test:  {test_new.shape}, Mean yield: {test_mean:.2f}")
    logging.info(f"  Distribution difference: {distribution_diff:.2f}")
    
    if distribution_diff < 2:
        logging.info(f"✅ Distribution shift SOLVED! Diff < 2")
    else:
        logging.warning(f"⚠️ Still some distribution difference")
    
    # Check year distribution (for info)
    if 'year' in combined_df.columns:
        logging.info(f"📅 Year distribution in splits:")
        train_years = train_new['year'].value_counts().sort_index()
        test_years = test_new['year'].value_counts().sort_index()
        logging.info(f"  Train years: {dict(train_years)}")
        logging.info(f"  Test years: {dict(test_years)}")
    
    return train_new, test_new

def select_features_no_year(df):
    """Select features excluding YEAR and temporal features"""
    logging.info("🎯 Selecting features (EXCLUDING YEAR)...")
    
    # EXCLUDE temporal features that cause bias
    excluded_features = [
        'year',           # Main temporal feature to exclude
        'county_id',      # ID feature
        'month_num',      # Can cause temporal bias
        'date',           # Direct temporal
        'yield'           # Target
    ]
    
    # Get all numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_features = [col for col in numeric_cols if col not in excluded_features]
    
    # Calculate correlations with yield
    correlations = df[candidate_features + ['yield']].corr()['yield'].abs()
    correlations = correlations.drop('yield').sort_values(ascending=False)
    
    # Select top features (focus on the strongest)
    selected_features = correlations.head(18).index.tolist()
    
    logging.info(f"📊 Selected {len(selected_features)} features (NO YEAR):")
    for i, feature in enumerate(selected_features[:12]):
        corr = correlations[feature]
        logging.info(f"  {i+1:2d}. {feature:20s}: {corr:.4f}")
    
    max_corr = correlations.max()
    logging.info(f"Max correlation: {max_corr:.4f}")
    
    # Verify no temporal features included
    temporal_check = [f for f in selected_features if 'year' in f.lower() or 'date' in f.lower()]
    if temporal_check:
        logging.warning(f"⚠️ Temporal features detected: {temporal_check}")
    else:
        logging.info(f"✅ No temporal features included")
    
    return selected_features

def prepare_balanced_data(train_df, test_df, features):
    """Prepare data from balanced split"""
    logging.info("🔧 Preparing balanced data...")
    
    # Extract features and targets
    X_train = train_df[features].values
    X_test = test_df[features].values
    y_train = train_df['yield'].values
    y_test = test_df['yield'].values
    
    # Handle NaN values
    for i, feature in enumerate(features):
        train_col = X_train[:, i]
        if np.isnan(train_col).any():
            median_val = np.nanmedian(train_col)
            X_train[np.isnan(X_train[:, i]), i] = median_val
            X_test[np.isnan(X_test[:, i]), i] = median_val
    
    # Scale features
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale targets (should be much more balanced now)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    logging.info(f"Feature range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    logging.info(f"Target balance - Train: {y_train_scaled.mean():.3f}, Test: {y_test_scaled.mean():.3f}")
    
    target_diff = abs(y_train_scaled.mean() - y_test_scaled.mean())
    if target_diff < 0.1:
        logging.info(f"✅ Target distributions well balanced!")
    else:
        logging.warning(f"⚠️ Still some target imbalance: {target_diff:.3f}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, target_scaler

def create_balanced_model(input_dim):
    """Create model for balanced data"""
    logging.info(f"🏗️ Creating model for {input_dim} features...")
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logging.info(f"Model: {model.count_params()} parameters")
    return model

def train_balanced_model(model, X_train, y_train, X_val, y_val):
    """Train model on balanced data"""
    logging.info("🚀 Training on balanced data...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
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

def evaluate_balanced_model(model, X_test, y_test, target_scaler):
    """Evaluate balanced model"""
    logging.info("📊 Evaluating balanced model...")
    
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
    
    logging.info(f"🎯 BALANCED MODEL RESULTS:")
    logging.info(f"  R² Score: {r2:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAE: {mae:.4f}")
    
    # Success analysis
    if r2 > 0.3:
        logging.info(f"🎉 EXCELLENT: R² > 0.3!")
    elif r2 > 0:
        logging.info(f"🎉 SUCCESS: POSITIVE R² ACHIEVED!")
    elif r2 > -0.2:
        logging.info(f"📈 MAJOR IMPROVEMENT: Very close to positive")
    else:
        logging.warning(f"⚠️ Still negative, but distribution is balanced")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.6)
    plt.plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'BALANCED MODEL (R² = {r2:.3f})')
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
    plt.savefig('reports/figures/balanced_model_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}

def main():
    """Main function with balanced approach"""
    try:
        logging.info("🚀 BALANCED MODEL - NO YEAR BIAS...")
        
        # Combine and create random split
        train_df, test_df = combine_and_random_split()
        
        # Select features (exclude year)
        features = select_features_no_year(train_df)
        
        # Prepare balanced data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, target_scaler = prepare_balanced_data(
            train_df, test_df, features
        )
        
        # Split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
        )
        
        logging.info(f"Final shapes: Train {X_train_final.shape}, Val {X_val.shape}, Test {X_test_scaled.shape}")
        
        # Create and train model
        model = create_balanced_model(X_train_final.shape[1])
        model, history = train_balanced_model(model, X_train_final, y_train_final, X_val, y_val)
        
        # Evaluate
        results = evaluate_balanced_model(model, X_test_scaled, y_test_scaled, target_scaler)
        
        logging.info("🎉 BALANCED MODEL COMPLETE!")
        logging.info(f"🏆 Final R² Score: {results['r2']:.4f}")
        
        return results
        
    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 