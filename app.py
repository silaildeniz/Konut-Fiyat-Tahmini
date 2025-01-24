import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import joblib  # scaler'ı kaydetmek için joblib kullanacağız

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

# Modeli Eğitme ve Scaler'ı Kaydetme
def train_model():
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
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, validation_data=(X_test_reshaped, y_test))

    # Modeli kaydetme (model.h5)
    model.save('model.h5')
    print("Model başarıyla kaydedildi: model.h5")

    # Scaler'ı kaydetme
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("Scaler'lar başarıyla kaydedildi.")

# Tkinter GUI
def create_gui():
    ilceler = [
        "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bahçelievler", "Bakırköy",
        "Bağcılar", "Başakşehir", "Beykoz", "Beylikdüzü", "Beyoğlu", "Beşiktaş",
        "Büyükçekmece", "Esenler", "Esenyurt", "Eyüpsultan", "Fatih", "Gaziosmanpaşa",
        "Güngören", "Kadıköy", "Kartal", "Kağıthane", "Küçükçekmece", "Maltepe",
        "Pendik", "Sancaktepe", "Sarıyer", "Silivri", "Sultanbeyli", "Sultangazi",
        "Tuzla", "Zeytinburnu", "Çekmeköy", "Ümraniye", "Üsküdar", "Şile", "Şişli"
    ]

    # Tkinter GUI
    root = tk.Tk()
    root.title("Ev Fiyatı Tahmin Arayüzü")
    root.geometry("450x500")

    # Başlık
    title_label = tk.Label(root, text="Ev Fiyatı Tahmin Arayüzü", font=("Arial", 18, "bold"), pady=20)
    title_label.pack(fill="x")

    # Etiketler ve girişler
    frame_inputs = tk.Frame(root)
    frame_inputs.pack(pady=10)

    label_room = tk.Label(frame_inputs, text="Oda Sayısı:", font=("Arial", 12))
    label_room.grid(row=0, column=0, padx=20, pady=5, sticky="w")
    entry_room = tk.Entry(frame_inputs, font=("Arial", 12), width=20)
    entry_room.grid(row=0, column=1, padx=20, pady=5)

    label_area = tk.Label(frame_inputs, text="M² (Alan):", font=("Arial", 12))
    label_area.grid(row=1, column=0, padx=20, pady=5, sticky="w")
    entry_area = tk.Entry(frame_inputs, font=("Arial", 12), width=20)
    entry_area.grid(row=1, column=1, padx=20, pady=5)

    label_age = tk.Label(frame_inputs, text="Yaş (Bina Yaşı):", font=("Arial", 12))
    label_age.grid(row=2, column=0, padx=20, pady=5, sticky="w")
    entry_age = tk.Entry(frame_inputs, font=("Arial", 12), width=20)
    entry_age.grid(row=2, column=1, padx=20, pady=5)

    label_floor = tk.Label(frame_inputs, text="Kat Sayısı:", font=("Arial", 12))
    label_floor.grid(row=3, column=0, padx=20, pady=5, sticky="w")
    entry_floor = tk.Entry(frame_inputs, font=("Arial", 12), width=20)
    entry_floor.grid(row=3, column=1, padx=20, pady=5)

    # İlçe seçimi için combobox
    label_location = tk.Label(frame_inputs, text="İlçe Seçin:", font=("Arial", 12))
    label_location.grid(row=4, column=0, padx=20, pady=5, sticky="w")

    location_combobox = ttk.Combobox(frame_inputs, font=("Arial", 12), width=18, values=ilceler)
    location_combobox.grid(row=4, column=1, padx=20, pady=5)

    # Tahmin Butonu
    def predict_price():
        try:
            # Kullanıcıdan gelen veriler
            room = float(entry_room.get())
            area = float(entry_area.get())
            age = float(entry_age.get())
            floor = float(entry_floor.get())
            location = location_combobox.get()  # Seçilen ilçe

            # Modeli Yükleme
            model = load_model('model.h5')

            # Scaler'ları Yükleme
            scaler_X = joblib.load('scaler_X.pkl')
            scaler_y = joblib.load('scaler_y.pkl')

            # Verileri normalleştirme
            X_input = np.array([[room, area, age, floor]])
            X_scaled = scaler_X.transform(X_input)

            # Veriyi LSTM modeline uygun şekle getirme
            X_input_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

            # Tahmin yapma
            price_scaled = model.predict(X_input_reshaped)

            # Tahmin edilen fiyatı eski ölçeğine döndürme
            price_rescaled = scaler_y.inverse_transform(price_scaled)

            # Fiyatı görüntüleme
            messagebox.showinfo("Tahmin Sonucu", f"Tahmin Edilen Fiyat: {price_rescaled[0][0]:.2f} TL")

        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")

    predict_button = tk.Button(root, text="Tahmin Et", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white",
                               command=predict_price)
    predict_button.pack(pady=20)

    root.mainloop()

# Modeli eğit
train_model()

# Arayüzü başlat
create_gui()
