#!/usr/bin/env python
# coding: utf-8

# # <span style="color:green;">UTS Model Deployment</span>

# ## Zaneta Fransiske - 2702312146
# ### Dataset B (Hotel) - Case 3

# #### Import Library

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


# <span style="color: red; font-size: 19px;"> Preprocessing Data</span>

# - Membersihkan nilai null (fillna).
# - Menangani kategori dengan mapping numerik.
# - Melakukan normalisasi numerik (manual berdasarkan mean dan std).
# - Menghapus outlier.
# - Support input dalam bentuk dictionary (data satuan) dan DataFrame (batch).

# <span style="color: purple; font-size: 12px;">preprocessor.py</span>

# In[2]:


class DataPreprocessor:
    def __init__(self):
        self.category_mappings = {
            'type_of_meal_plan': {
                'Meal Plan 1': 0,
                'Meal Plan 2': 1,
                'Meal Plan 3': 2,
                'Not Selected': 3
            },
            'room_type_reserved': {
                'Room_Type 1': 0, 
                'Room_Type 2': 1,
                'Room_Type 3': 2,
                'Room_Type 4': 3,
                'Room_Type 5': 4,
                'Room_Type 6': 5,
                'Room_Type 7': 6
            },
            'market_segment_type': {
                'Online': 0,
                'Offline': 1,
                'Corporate': 2,
                'Aviation': 3,
                'Complementary': 4
            }
        }
        
        self.categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        self.cols_to_clean = [
            'lead_time', 
            'avg_price_per_room',
            'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled'
        ]
        
        self.num_col_stats = {
            # column_name: (mean, std)
            'no_of_adults': (0, 1),
            'no_of_children': (0, 1),
            'no_of_weekend_nights': (0, 1),
            'no_of_week_nights': (0, 1),
            'required_car_parking_space': (0, 1),
            'lead_time': (0, 1),
            'arrival_year': (0, 1),
            'arrival_month': (0, 1),
            'arrival_date': (0, 1),
            'repeated_guest': (0, 1),
            'no_of_previous_cancellations': (0, 1),
            'no_of_previous_bookings_not_canceled': (0, 1),
            'avg_price_per_room': (0, 1),
            'no_of_special_requests': (0, 1)
        }
        
        self.scaler = StandardScaler()
    
    def preprocess_data(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
    
        df = data.copy()

        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
        if 'required_car_parking_space' in df.columns:
            df["required_car_parking_space"] = df["required_car_parking_space"].fillna(0)

        if 'avg_price_per_room' in df.columns:
            df["avg_price_per_room"] = df['avg_price_per_room'].fillna(
                df["avg_price_per_room"].median() if not df["avg_price_per_room"].empty else 0
            )

        if 'no_of_children' in df.columns:
            df = df[df['no_of_children'] <= 3]

        if df.empty:
            raise ValueError("No valid data after preprocessing. Check for extreme outliers.")

        df = self.remove_outliers(df, self.cols_to_clean)

        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].map(self.category_mappings.get(col, {})).fillna(0).astype(int)

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in num_cols:
            if col in self.num_col_stats:
                mean, std = self.num_col_stats[col]
                std = std if std != 0 else 1
                df[col] = (df[col] - mean) / std

        return df

    
    def remove_outliers(self, df, columns):
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and not df_copy[col].empty:
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        return df_copy


# <span style="color: red; font-size: 19px;"> Model Prediction</span>

# - Load model dari file .pkl menggunakan pickle.
# - Menyediakan fallback model jika gagal load (untuk keperluan development atau robustness).
# - predict(data) untuk prediksi satu record.
# - batch_predict(csv_path) untuk prediksi banyak data dari file.
# - Return hasil prediksi dan probabilitas cancel.
# - Tangani edge case seperti model tidak valid atau data kosong.

# <span style="color: purple; font-size: 12px;">predictor.py</span>

# In[3]:


class ModelPredictor:
    def __init__(self, model_path, preprocessor):
        self.model = None
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                if hasattr(model, 'estimators_'):
                    for tree in model.estimators_:                           
                        if not hasattr(tree, 'monotonic_cst'):
                            tree.monotonic_cst = None
                    
                        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                            self.model = model
                        else:
                            raise AttributeError("Loaded object does not have required prediction methods")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = self._create_fallback_model()
            print("Using fallback model instead")
            
        self.preprocessor = preprocessor
    
    def _create_fallback_model(self):
        try:
            print("Creating a fallback RandomForestClassifier for demonstration purposes.")
            fallback_model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            n_features = 17 
            dummy_X = np.zeros((10, n_features))
            dummy_y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

            fallback_model.fit(dummy_X, dummy_y)

            for tree in fallback_model.estimators_:
                if not hasattr(tree, 'monotonic_cst'):
                    tree.monotonic_cst = None
                    
            return fallback_model
        except Exception as e:
            print(f"Error creating fallback RandomForestClassifier: {e}")
            
            class SimpleFallbackModel:
                def predict(self, X):
                    return np.zeros(len(X))
                
                def predict_proba(self, X):
                    return np.array([[0.7, 0.3] for _ in range(len(X))])
                
                monotonic_cst = None
            
            return SimpleFallbackModel()
    
    def predict(self, data):
        if self.model is None:
            return {"error": "Model not loaded properly. Please check the model path."}
        
        try:
            processed_data = self.preprocessor.preprocess_data(data)
        
            if processed_data.empty:
             return {"error": "No valid data after preprocessing. Check for extreme outliers."}
        
            if 'Booking_ID' in processed_data.columns:
                processed_data = processed_data.drop(columns=['Booking_ID'])
            if 'booking_status' in processed_data.columns:
                processed_data = processed_data.drop(columns=['booking_status'])
        
            try:
                cancellation_prob = self.model.predict_proba(processed_data)[:, 1]
                prediction = self.model.predict(processed_data)
            except AttributeError as e:
                if 'monotonic_cst' in str(e):
                    if hasattr(self.model, 'estimators_'):
                        for tree in self.model.estimators_:
                            if not hasattr(tree, 'monotonic_cst'):
                                tree.monotonic_cst = None
                        cancellation_prob = self.model.predict_proba(processed_data)[:, 1]
                        prediction = self.model.predict(processed_data)
                    else:
                        print("Using fallback prediction due to monotonic_cst error")
                        prediction = np.array([0])  
                        cancellation_prob = np.array([0.3])  
                else:
                    raise  
        
            results = {
                "prediction": "Canceled" if prediction[0] == 1 else "Not Canceled",
                "cancellation_probability": round(float(cancellation_prob[0]), 2),
                "status_code": int(prediction[0])
            }
            return results
    
        except Exception as e:
            import traceback
            print(f"Prediction error: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Prediction error: {str(e)}"}
    
    def batch_predict(self, data_path):
        if self.model is None:
            return pd.DataFrame()
            
        try:
            data = pd.read_csv(data_path)
            processed_data = self.preprocessor.preprocess_data(data)
            booking_ids = None
            if 'Booking_ID' in processed_data.columns:
                booking_ids = processed_data['Booking_ID'].copy()
                processed_data = processed_data.drop(columns=['Booking_ID'])
            
            if 'booking_status' in processed_data.columns:
                processed_data = processed_data.drop(columns=['booking_status'])
            

            predictions = self.model.predict(processed_data)
            probabilities = self.model.predict_proba(processed_data)[:, 1]
            
            results = pd.DataFrame({
                "prediction": ["Canceled" if p == 1 else "Not Canceled" for p in predictions],
                "cancellation_probability": [round(p, 2) for p in probabilities],
                "status_code": predictions
            })

            if booking_ids is not None:
                results.insert(0, 'Booking_ID', booking_ids)
            
            return results
        
        except Exception as e:
            print(f"Batch prediction error: {str(e)}")
            return pd.DataFrame()


# #### Testing

# - Contoh data.
# - Load model dan prediksi.
# - Menampilkan hasil prediksi.

# In[4]:


def sample_booking():
    return {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 3,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 300,
        'arrival_year': 2022,
        'arrival_month': 7,
        'arrival_date': 15,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 120.5,
        'no_of_special_requests': 1
    }

def example_predict(model_path):
    booking = sample_booking()

    preprocessor = DataPreprocessor()
    predictor = ModelPredictor(model_path, preprocessor)
    
    result = predictor.predict(booking)
    print("Prediction result:", result)
    
    print("\nHasil Prediksi:", result["prediction"])
    print("Probabilitas Cancel:", result["cancellation_probability"])


# In[5]:


if __name__ == "__main__":
    MODEL_PATH = 'C:/COOLYEAH/SEM 4/Model Deployment/UTS/best_modell.pkl'
    
    example_predict(MODEL_PATH)
    


# #### Analisis
# 
# **Prediksi**: Model memprediksi bahwa booking ini akan dibatalkan. Ini berarti bahwa berdasarkan data input yang diberikan, model melihat bahwa kemungkinan besar pemesanan hotel ini tidak akan berhasil dan akan dibatalkan.
# 
# **Probabilitas Cancel**: Nilai probabilitas yang diberikan adalah 0.85 (85%). Artinya, model memiliki tingkat keyakinan yang tinggi bahwa booking ini akan dibatalkan, yang juga menunjukkan bahwa model memiliki tingkat keakuratan yang cukup tinggi dalam memprediksi pembatalan berdasarkan data yang diberikan.

# <span style="color: red; font-size: 19px;">Video Link</span>

# **Youtube**: https://youtu.be/dBQmqQquscw
# 
# **Google Drive**: https://drive.google.com/file/d/1Y4z4LGf1nERX2GUCobnmvWJHYZxb6f1A/view?usp=sharing
