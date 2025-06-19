import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Eğitim ve test verilerini yükle"""
    logger.info("Veri setleri yükleniyor...")
    
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    logger.info(f"Eğitim seti boyutu: {train_df.shape}")
    logger.info(f"Test seti boyutu: {test_df.shape}")
    
    return train_df, test_df

def handle_outliers(df, columns):
    """Aykırı değerleri işle"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df_clean

def preprocess_data(train_df, test_df):
    """Geliştirilmiş veri ön işleme adımları"""
    logging.info("Veri ön işleme başlıyor...")

    # Eksik yield_per_acre değerlerini doldur
    def fill_yield_na(df):
        df['yield_per_acre'] = df.groupby(['county_name', 'year'])['yield_per_acre'].transform(
            lambda x: x.fillna(x.mean())
        )
        df['yield_per_acre'] = df['yield_per_acre'].fillna(df['yield_per_acre'].mean())
        return df

    train_df = fill_yield_na(train_df)
    test_df = fill_yield_na(test_df)

    # Hedef değişkeni ayır
    y_train = train_df['yield_per_acre']
    y_test = test_df['yield_per_acre']
    
    # Kategorik ve sayısal sütunları belirle
    categorical_columns = ['county_name']
    numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'yield_per_acre']
    
    # Sonsuz değerleri NaN ile değiştir
    train_df = train_df.replace([float('inf'), float('-inf')], np.nan)
    test_df = test_df.replace([float('inf'), float('-inf')], np.nan)
    
    # Aykırı değerleri işle
    train_df = handle_outliers(train_df, numeric_columns)
    test_df = handle_outliers(test_df, numeric_columns)
    
    # Eksik değerleri doldur
    for col in numeric_columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Kategorik değişkenleri one-hot encoding ile dönüştür
    train_categorical = pd.get_dummies(train_df[categorical_columns], prefix='county_name')
    test_categorical = pd.get_dummies(test_df[categorical_columns], prefix='county_name')
    
    # Eksik sütunları ekle
    for col in train_categorical.columns:
        if col not in test_categorical.columns:
            test_categorical[col] = 0
    for col in test_categorical.columns:
        if col not in train_categorical.columns:
            train_categorical[col] = 0
    
    # Sütunları aynı sırada olacak şekilde düzenle
    train_categorical = train_categorical.reindex(sorted(train_categorical.columns), axis=1)
    test_categorical = test_categorical.reindex(sorted(test_categorical.columns), axis=1)
    
    # Sayısal değişkenleri ölçeklendir (RobustScaler kullan)
    scaler = RobustScaler()
    train_numeric = pd.DataFrame(
        scaler.fit_transform(train_df[numeric_columns]),
        columns=numeric_columns
    )
    test_numeric = pd.DataFrame(
        scaler.transform(test_df[numeric_columns]),
        columns=numeric_columns
    )
    
    # Özellikleri birleştir
    X_train = pd.concat([train_numeric, train_categorical], axis=1)
    X_test = pd.concat([test_numeric, test_categorical], axis=1)
    
    # Özellik seçimi
    selector = SelectKBest(score_func=f_regression, k=50)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Seçilen özelliklerin isimlerini al
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    logging.info(f"Ön işleme sonrası eğitim seti boyutu: {X_train_selected.shape}")
    logging.info(f"Ön işleme sonrası test seti boyutu: {X_test_selected.shape}")
    
    return X_train_selected, X_test_selected, y_train, y_test, selected_features

def train_model(model_name, X_train, y_train):
    """Belirli bir modeli eğit"""
    logging.info(f"\n{model_name} modeli eğitiliyor...")
    
    if model_name == 'XGBoost':
        model = xgb.XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0,
            random_state=42
        )
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    
    elif model_name == 'LightGBM':
        model = lgb.LGBMRegressor(
            random_state=42
        )
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(
            random_state=42,
            verbose=False
        )
        param_distributions = {
            'iterations': [100, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'depth': [3, 5, 7, 9, 11],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    
    # Hiperparametre optimizasyonu
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    logging.info(f"En iyi parametreler: {random_search.best_params_}")
    
    return best_model

def evaluate_model(model, model_name, X_test, y_test, feature_columns):
    """Modeli değerlendir ve sonuçları görselleştir"""
    logger.info(f"\n{model_name} modeli değerlendiriliyor...")
    
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Metrikler
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {mse:.2f}")
    logger.info(f"Test RMSE: {rmse:.2f}")
    logger.info(f"Test MAE: {mae:.2f}")
    logger.info(f"Test R2: {r2:.2f}")
    
    # Özellik önemlilikleri
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Özellik önemliliklerini görselleştir
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title(f'{model_name} - Top 20 Özellik Önemliliği')
        plt.tight_layout()
        plt.savefig(f'reports/figures/{model_name}_feature_importance.png')
        plt.close()
        
        logger.info("\nTop 10 önemli özellik:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"- {row['feature']}: {row['importance']:.4f}")
    
    # Gerçek vs Tahmin grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Verim')
    plt.ylabel('Tahmin Edilen Verim')
    plt.title(f'{model_name} - Gerçek vs Tahmin Edilen Verim')
    plt.tight_layout()
    plt.savefig(f'reports/figures/{model_name}_actual_vs_predicted.png')
    plt.close()
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def save_model(model, model_name):
    """Modeli kaydet"""
    logger.info(f"\n{model_name} modeli kaydediliyor...")
    
    # Model dizinini oluştur
    Path('models').mkdir(exist_ok=True)
    
    # Modeli kaydet
    model_path = f'models/{model_name}.joblib'
    joblib.dump(model, model_path)
    
    logger.info(f"Model kaydedildi: {model_path}")

def main():
    """Ana fonksiyon"""
    try:
        # Veriyi yükle
        train_df, test_df = load_data()
        
        # Veriyi ön işle
        X_train, X_test, y_train, y_test, feature_columns = preprocess_data(train_df, test_df)
        
        # Modelleri eğit ve değerlendir
        models = ['XGBoost', 'LightGBM', 'CatBoost']
        results = {}
        
        for model_name in models:
            # Modeli eğit
            model = train_model(model_name, X_train, y_train)
            
            # Modeli değerlendir
            metrics = evaluate_model(model, model_name, X_test, y_test, feature_columns)
            results[model_name] = metrics
            
            # Modeli kaydet
            save_model(model, model_name)
        
        # Sonuçları karşılaştır
        logger.info("\nModel Karşılaştırması:")
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"MSE: {metrics['mse']:.2f}")
            logger.info(f"RMSE: {metrics['rmse']:.2f}")
            logger.info(f"MAE: {metrics['mae']:.2f}")
            logger.info(f"R2: {metrics['r2']:.2f}")
        
        logger.info("\nModel eğitimi ve değerlendirmesi tamamlandı!")
        
    except Exception as e:
        logger.error(f"Hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    main() 