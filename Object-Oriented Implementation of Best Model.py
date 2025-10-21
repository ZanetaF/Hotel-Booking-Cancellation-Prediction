#!/usr/bin/env python
# coding: utf-8

# # <span style="color:green;">UTS Model Deployment</span>

# ## Zaneta Fransiske - 2702312146
# ### Dataset B (Hotel) - Case 2

# #### Import Library

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import skew
import pickle
import os


# <span style="color: red; font-size: 22px;">Data Handler</span>

# Kelas `DataHandler` bertujuan untuk mengelola dan memproses data.

# #### Metode Kelas:
# 
# ##### 1. `load_data()`
# 
# - Digunakan untuk memuat dataset dari file CSV yang ada di path yang telah ditentukan pada `self.filepath`.
# - Jika terjadi kesalahan saat memuat data (misalnya file tidak ditemukan), maka akan menampilkan pesan kesalahan.
# 
# ##### 2. `preprocess_data()`
# 
# - **Menangani Missing Value**: 
#   - Mengisi nilai yang hilang pada kolom kategorikal dengan modus. 
#   - Untuk kolom numerik seperti `required_car_parking_space` dan `avg_price_per_room`, nilai yang hilang akan diisi dengan 0 atau median kolom tersebut.
#   
# - **Menangani Outlier**: 
#   - Menghapus data yang tidak valid atau berlebihan pada kolom `no_of_children` (membatasi jumlah anak maksimal 3). 
#   - Kemudian, dilakukan pembersihan lebih lanjut dari outlier menggunakan metode IQR (Interquartile Range).
#   
# - **Encoding Kategorikal**: 
#   - Kolom-kolom kategorikal (`type_of_meal_plan`, `room_type_reserved`, `market_segment_type`) akan diubah menjadi angka menggunakan `LabelEncoder`.
#   
# - **Normalisasi Fitur Numerik**: 
#   - Semua fitur numerik akan diskalakan menggunakan `StandardScaler` agar memiliki distribusi dengan mean 0 dan standar deviasi 1.
#   
# - **Membagi Data**: 
#   - Data kemudian dibagi menjadi data training dan testing menggunakan `train_test_split`.
# 
# ##### 3. `remove_outliers()`
# 
# - Menggunakan metode IQR untuk menghapus outlier pada kolom-kolom yang ditentukan. Outlier dihitung dengan mencari nilai yang berada di luar rentang [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR], di mana Q1 adalah kuartil pertama, dan Q3 adalah kuartil ketiga.
# 
# ##### 4. `get_data()`
# 
# - Mengembalikan data yang sudah diproses dalam bentuk `X_train`, `X_test`, `y_train`, `y_test`, dan `self.feature_names`.
# 

# In[2]:


class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.cols_to_clean = [
            'lead_time', 
            'avg_price_per_room',
            'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled'
        ]
        self.categorical_columns = [
            'type_of_meal_plan', 
            'room_type_reserved', 
            'market_segment_type'
        ]
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        
    def preprocess_data(self):
        df = self.df.copy()
        
        # Handle Missing Value
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
            
        df["required_car_parking_space"].fillna(0, inplace=True)
        df["avg_price_per_room"].fillna(df["avg_price_per_room"].median(), inplace=True)
        
        # Handle Outliers
        df = df[df['no_of_children'] <= 3]
        self.df_cleaned = self.remove_outliers(df, self.cols_to_clean)
        
        # Encoding
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.df_cleaned[col] = le.fit_transform(self.df_cleaned[col])
            self.label_encoders[col] = le
            
        self.df_cleaned['booking_status'] = self.df['booking_status'].apply(
            lambda x: 1 if x == 'Canceled' else 0
        )
        
        # Scaling
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        scaled_features = self.scaler.fit_transform(self.df_cleaned[num_cols])
        self.df_cleaned[num_cols] = scaled_features
        
        # Split Data
        X = self.df_cleaned.drop(columns=['Booking_ID', 'booking_status'])
        y = self.df_cleaned['booking_status']
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
    def remove_outliers(self, df, columns):
        df_copy = df.copy()
        for col in columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        return df_copy
    
    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names


# <span style="color: red; font-size: 22px;">Model Handler</span>
# 
# Kelas `ModelHandler` digunakan untuk menangani seluruh proses terkait model machine learning, mulai dari pelatihan, evaluasi, hingga visualisasi hasil.

# #### Metode Kelas:
# 
# ##### 1. `train(self, X_train, y_train)`
#   - Melatih model menggunakan data pelatihan yang diberikan (`X_train` dan `y_train`).
#   - Menggunakan metode `fit()` dari `RandomForestClassifier` untuk melatih model dengan data pelatihan yang diberikan.
# 
# ##### 2. `evaluate(self, X_test, y_test)`
#   - Mengevaluasi kinerja model pada data testing (`X_test` dan `y_test`).
#   - Menggunakan metode `predict()` untuk memprediksi hasil pada data uji.
#   - Kemudian, menghitung dan mencetak hasil evaluasi dalam bentuk:
#     - **Akurasi** (`accuracy_score`)
#     - **ROC AUC** (`roc_auc_score`)
#     - **Laporan Klasifikasi** (`classification_report`)
# 
# ##### 3. `plot_confusion_matrix(self, X_test, y_test)`
#   - Menampilkan Confusion Matrix.
# 
# ##### 4. `plot_feature_importance(self, feature_names)`
#   - Menampilkan visualisasi feature importance yang digunakan dalam model.
# 

# In[3]:


class ModelHandler:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.best_model_name = "Random Forest"  # For plot title
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        print("Random Forest Accuracy: {:.2f}\n".format(accuracy_score(y_test, y_pred)))
        print("Random Forest ROC AUC Score: {:.2f}\n".format(roc_auc_score(y_test, y_pred)))
        print("Random Forest Classification Report:")
        print(classification_report(y_test, y_pred))
        
    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    def plot_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances * 100  # to percentage
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title(f'Feature Importance - {self.best_model_name}')
        plt.xlabel('Importance (%)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()


# <span style="color: red; font-size: 19px;"> Mengolah Data, Melatih Model, dan Mengevaluasi Hasil Prediksi</span>

# In[4]:


if __name__ == '__main__':
    data_handler = DataHandler('C:/COOLYEAH/SEM 4/Model Deployment/UTS/Dataset_B_hotel.csv')
    data_handler.load_data()
    data_handler.preprocess_data()
    X_train, X_test, y_train, y_test, feature_names = data_handler.get_data()

    model_handler = ModelHandler()
    model_handler.train(X_train, y_train)
    model_handler.evaluate(X_test, y_test)


# In[5]:


model_handler.plot_confusion_matrix(X_test, y_test)


# In[6]:


model_handler.plot_feature_importance(feature_names)


# #### Interpretasi
# 
# Dari hasil evaluasi yang telah dilakukan pada **model berbasis Object-Oriented Programming (OOP)** yang saya buat, dapat disimpulkan bahwa performa model yang dihasilkan sama seperti model machine learning pada case 1. 
# 
# ###### 1. Preprocessing Data
# - Pada kedua model (OOP dan model machine learning case 1), tahap preprocessing data dilakukan dengan cara yang sangat mirip. Keduanya menangani nilai yang hilang, membersihkan data dari outliers, melakukan encoding untuk variabel kategorikal, dan menstandarisasi data numerik. Hal ini memastikan bahwa input data yang digunakan untuk pelatihan model memiliki kualitas yang baik.
# 
# ###### 2. Pembagian Data
# - Pembagian data menjadi training dan testing dilakukan dengan menggunakan teknik yang sama pada kedua model. Data dipecah dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian menggunakan `train_test_split`, yang memastikan distribusi data yang seimbang antara set pelatihan dan pengujian.
# 
# ##### 3. Pelatihan Model
# - Model yang digunakan adalah **Random Forest**, yang diterapkan secara konsisten di kedua pendekatan (model OOP dan case 1). Model dilatih menggunakan data yang telah diproses dan dievaluasi dengan cara yang sama, yaitu menghitung metrik seperti **accuracy**, **ROC AUC**, **precision**, **recall**, dan **f1-score**.
# 
# ###### 4. Evaluasi Model
# - Hasil evaluasi model menggunakan metrik utama, seperti **accuracy**, **ROC AUC**, dan **classification report**, menunjukkan bahwa performa model OOP sama baiknya dengan model case 1. Hasil prediksi dan metrik evaluasi juga mirip, menunjukkan bahwa implementasi model dalam pendekatan OOP tetap mempertahankan akurasi dan kualitas yang sama.
# 
# ###### 5. Visualisasi Hasil
# - Baik pada model OOP maupun model awal case 1, visualisasi **confusion matrix** dan **feature importance** memberikan wawasan yang serupa mengenai bagaimana model bekerja dan fitur mana yang paling berpengaruh terhadap prediksi. Visualisasi ini membantu untuk lebih memahami bagaimana model memproses data dan mengidentifikasi area yang perlu diperbaiki.
# 
# ##### 6. Konsistensi Output
# - Model OOP yang saya buat dapat menghasilkan **output** yang konsisten dengan model machine learning awal. Ini membuktikan bahwa implementasi model menggunakan pendekatan OOP tidak mengubah cara kerja atau hasil dari model machine learning yang sebenarnya.
# 
# 
# ### Kesimpulan
# Dengan demikian, dapat disimpulkan bahwa meskipun pendekatan yang digunakan berbeda (OOP vs. model awal case 1), hasil evaluasi dan performa model menunjukkan kesamaan yang signifikan. Model OOP yang saya buat sudah berhasil memanfaatkan konsep Object-Oriented Programming tanpa mengorbankan kualitas dan akurasi yang telah dicapai oleh model machine learning awal.Dengan OOP memungkinkan pengorganisasian kode yang lebih modular dan terstruktur
# 

# <span style="color: red; font-size: 19px;">Video Link</span>

# **Youtube**: https://youtu.be/dBQmqQquscw
# 
# **Google Drive**: https://drive.google.com/file/d/1Y4z4LGf1nERX2GUCobnmvWJHYZxb6f1A/view?usp=sharing
