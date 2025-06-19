"""
Optimized Neural Network for Agricultural Yield Prediction

Bu model LSTM problemlerini çözer:
1. Dense NN tabular data için daha uygun
2. Seasonal features direkt kullanılır (sequences yok)
3. Target distribution shift düzeltilir
4. Advanced scaling strategy
5. Feature selection optimize edilir
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

def load_and_prepare_data():
    """Load and prepare data addressing distribution shift"""
    logging.info("Loading and preparing data...")
    
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    logging.info(f"Original - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Remove rows with missing yield
    train_df = train_df.dropna(subset=['yield'])
    test_df = test_df.dropna(subset=['yield'])
    
    # Address target distribution shift
    train_mean = train_df['yield'].mean()
    test_mean = test_df['yield'].mean()
    shift = test_mean - train_mean
    
    logging.info(f"Target distribution shift: {shift:.2f}")
    logging.info(f"Train yield: {train_mean:.2f} ± {train_df['yield'].std():.2f}")
    logging.info(f"Test yield: {test_mean:.2f} ± {test_df['yield'].std():.2f}")
    
    # Filter extreme outliers
    train_q1, train_q3 = train_df['yield'].quantile([0.05, 0.95])
    test_q1, test_q3 = test_df['yield'].quantile([0.05, 0.95])
    
    train_df = train_df[(train_df['yield'] >= train_q1) & (train_df['yield'] <= train_q3)]
    test_df = test_df[(test_df['yield'] >= test_q1) & (test_df['yield'] <= test_q3)]
    
    logging.info(f"After filtering - Train: {train_df.shape}, Test: {test_df.shape}")
    
    return train_df, test_df

def select_top_features(train_df, top_k=20):
    """Select top features based on correlation analysis"""
    logging.info("Selecting top predictive features...")
    
    # Get all numeric columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['county_id', 'year', 'month_num', 'yield']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations with yield
    correlations = train_df[feature_cols + ['yield']].corr()['yield'].abs()
    correlations = correlations.drop('yield').sort_values(ascending=False)
    
    # Select top features
    selected_features = correlations.head(top_k).index.tolist()
    
    logging.info(f"Selected {len(selected_features)} top features:")
    for i, (feature, corr) in enumerate(correlations.head(top_k).items()):
        logging.info(f"  {i+1:2d}. {feature:25s}: {corr:.4f}")
    
    return selected_features

def create_optimized_model(input_dim, learning_rate=0.001):
    """Create optimized dense neural network"""
    logging.info(f"Creating optimized dense NN with {input_dim} features...")
    
    model = Sequential([
        # Input layer with dropout
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers with progressive size reduction
        Dense(64, activation='relu'),
        BatchNormalization(), 
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(16, activation='relu'),
        
        # Output layer
        Dense(1)
    ])
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='huber',  # Robust to outliers
        metrics=['mae', 'mse']
    )
    
    logging.info(f"Model created with {model.count_params()} parameters")
    logging.info("Architecture optimizations:")
    logging.info("  - Dense layers for tabular data")
    logging.info("  - BatchNormalization for stable training")
    logging.info("  - Progressive dropout for regularization")
    logging.info("  - Huber loss for robustness")
    
    return model

def advanced_feature_scaling(X_train, X_test, strategy='quantile'):
    """Apply advanced scaling strategy"""
    logging.info(f"Applying {strategy} scaling...")
    
    if strategy == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    elif strategy == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the optimized model"""
    logging.info("Training optimized neural network...")
    
    # Advanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_optimized_nn.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    logging.info("Training completed")
    return model, history

def evaluate_model(model, X_test, y_test, feature_scaler=None, target_scaler=None):
    """Evaluate model with comprehensive metrics"""
    logging.info("Evaluating optimized model...")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform if scalers provided
    if target_scaler:
        y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
    bias = np.mean(y_pred_orig - y_test_orig)
    
    logging.info(f"🎯 Model Performance:")
    logging.info(f"  R² Score: {r2:.4f}")
    logging.info(f"  RMSE: {rmse:.4f} bushels/acre")
    logging.info(f"  MAE: {mae:.4f} bushels/acre")
    logging.info(f"  MAPE: {mape:.2f}%")
    logging.info(f"  Bias: {bias:.4f} bushels/acre")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test_orig, y_pred_orig, alpha=0.6, s=30)
    axes[0, 0].plot([y_test_orig.min(), y_test_orig.max()], 
                   [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Yield (bushels/acre)')
    axes[0, 0].set_ylabel('Predicted Yield (bushels/acre)')
    axes[0, 0].set_title(f'Optimized NN Predictions (R² = {r2:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_test_orig - y_pred_orig
    axes[0, 1].scatter(y_pred_orig, residuals, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Yield (bushels/acre)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Prediction Error (bushels/acre)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics
    axes[1, 1].axis('off')
    metrics_text = f"""
    Performance Metrics:
    
    R² Score: {r2:.4f}
    RMSE: {rmse:.2f} bushels/acre
    MAE: {mae:.2f} bushels/acre  
    MAPE: {mape:.2f}%
    Bias: {bias:.2f} bushels/acre
    
    Sample Size: {len(y_test_orig)}
    Mean Actual: {y_test_orig.mean():.2f}
    Mean Predicted: {y_pred_orig.mean():.2f}
    """
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('reports/figures/optimized_nn_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape, 'bias': bias}

def main():
    """Main training function"""
    try:
        logging.info("🚀 Starting optimized neural network training...")
        
        # Create directories
        ensure_directories()
        
        # Load and prepare data
        train_df, test_df = load_and_prepare_data()
        
        # Select top features
        selected_features = select_top_features(train_df, top_k=20)
        
        # Prepare features and targets
        X_train = train_df[selected_features].values
        y_train = train_df['yield'].values
        X_test = test_df[selected_features].values
        y_test = test_df['yield'].values
        
        logging.info(f"Feature matrix shapes: Train {X_train.shape}, Test {X_test.shape}")
        
        # Advanced feature scaling
        X_train_scaled, X_test_scaled, feature_scaler = advanced_feature_scaling(
            X_train, X_test, strategy='quantile'
        )
        
        # Scale targets for better training stability
        target_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
        )
        
        logging.info(f"Final shapes - Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test_scaled.shape}")
        
        # Create and train model
        model = create_optimized_model(X_train_final.shape[1], learning_rate=0.002)
        model, history = train_model(model, X_train_final, y_train_final, X_val, y_val)
        
        # Evaluate model
        results = evaluate_model(model, X_test_scaled, y_test_scaled, 
                               feature_scaler, target_scaler)
        
        # Save model and scalers
        model.save('models/optimized_nn_model.keras')
        joblib.dump(feature_scaler, 'models/optimized_nn_feature_scaler.joblib')
        joblib.dump(target_scaler, 'models/optimized_nn_target_scaler.joblib')
        joblib.dump(selected_features, 'models/optimized_nn_features.joblib')
        
        logging.info("🎉 Training completed successfully!")
        logging.info(f"🏆 Final R² Score: {results['r2']:.4f}")
        
        # Compare with baseline
        if results['r2'] > 0:
            logging.info("✅ SUCCESS: Positive R² achieved!")
        else:
            logging.info("⚠️  Still negative R² - needs further investigation")
            
        return results
        
    except Exception as e:
        logging.error(f"❌ Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 