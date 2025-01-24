import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Veriyi hazırlama ve ön işleme
def wrangle(filename):
    df = pd.read_csv(filename)

    # Convert to numeric (coerce errors to handle invalid data)
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')

    # Drop rows with NaN values in "Price" after conversion
    df = df.dropna(subset=["Price"])

    # split room columns to separate the bedrooms from the living rooms
    df["Room"] = df["Room"].str.split("+", expand=True)[0]

    # change numerical columns from object to float
    df["Area"] = df["Area"].replace(r'[^\d.]', '', regex=True).astype(float)
    df["Age"] = df["Age"].replace(r'[^\d.]', '', regex=True).astype(float)
    df["Floor"] = df["Floor"].replace(r'[^\d.]', '', regex=True).astype(float)
    df["Room"] = df["Room"].replace(r'[^\d.]', '', regex=True).astype(float)

    # drop outliers
    low, high = df["Area"].quantile([0.1, 0.9])
    mask_area = df["Area"].between(low, high)
    low, high = df["Price"].quantile([0.1, 0.9])
    mask_price = df["Price"].between(low, high)
    low, high = df["Age"].quantile([0.1, 0.9])
    mask_age = df["Age"].between(low, high)
    df = df[mask_area & mask_price & mask_age]

    return df

# Veriyi yükleme ve işleme
df = wrangle("C:/Users/sila/Desktop/konut/archive/istanbul_satilik_evler_2023.csv")

# Özellikler ve hedefi ayırma
X = df[["Room", "Area", "Age", "Floor"]]
y = df["Price"]

# Veriyi normalleştirme
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM için veriyi yeniden şekillendirme (3D array: [samples, timesteps, features])
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# LSTM Modeli Oluşturma
model = Sequential()

# LSTM katmanı
model.add(LSTM(units=64, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))

# Çıktı katmanı
model.add(Dense(1))

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, validation_data=(X_test_reshaped, y_test))

# Eğitim sürecini görselleştirme
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Modeli kaydetme (model.h5)
model.save('model.h5')
print("Model başarıyla kaydedildi: model.h5")

# Test verisi ile tahmin yapma
y_pred = model.predict(X_test_reshaped)

# Y_pred'i eski ölçeğine geri döndürme
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Sonuçları karşılaştırma
for i in range(10):
    print(f"Gerçek: {y_test_rescaled[i][0]}, Tahmin: {y_pred_rescaled[i][0]}")

