import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class YieldPredictor:
    def __init__(self):
        """Verim tahmin modeli başlatıcısı"""
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path):
        """
        Veri setini model eğitimi için hazırlar
        
        Args:
            data_path (str): Veri seti dosya yolu
        """
        # Veriyi oku
        df = pd.read_csv(data_path)
        
        # Zaman serisi verilerini düzenle
        X = []
        y = []
        
        for _, row in df.iterrows():
            # Zaman serisi verilerini numpy dizisine dönüştür
            time_series = np.array(eval(row['time_series']))
            X.append(time_series)
            y.append(row['yield'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Veriyi ölçeklendir
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        # Eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        LSTM tabanlı derin öğrenme modeli oluşturur
        
        Args:
            input_shape (tuple): Giriş verisi şekli
        """
        model = models.Sequential([
            # LSTM katmanları
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            # Yoğun katmanlar
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Verim tahmini için tek çıkış
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Modeli eğitir
        
        Args:
            X_train: Eğitim verileri
            y_train: Eğitim etiketleri
            epochs: Eğitim döngüsü sayısı
            batch_size: Batch boyutu
            validation_split: Doğrulama seti oranı
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Modeli değerlendirir
        
        Args:
            X_test: Test verileri
            y_test: Test etiketleri
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Verim tahmini yapar
        
        Args:
            X: Tahmin için giriş verileri
        """
        # Veriyi ölçeklendir
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        return self.model.predict(X)
    
    def save_model(self, model_path):
        """
        Modeli kaydeder
        
        Args:
            model_path (str): Model kayıt yolu
        """
        self.model.save(model_path)
    
    def load_model(self, model_path):
        """
        Kaydedilmiş modeli yükler
        
        Args:
            model_path (str): Model dosya yolu
        """
        self.model = models.load_model(model_path)

def main():
    # Örnek kullanım
    predictor = YieldPredictor()
    
    # Veriyi hazırla
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        'data/processed/corn_data_2017_2021.csv'
    )
    
    # Model oluştur
    model = predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Modeli eğit
    history = predictor.train(X_train, y_train)
    
    # Modeli değerlendir
    test_loss, test_mae = predictor.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")
    
    # Modeli kaydet
    predictor.save_model('data/models/yield_predictor.h5')

if __name__ == "__main__":
    main() 