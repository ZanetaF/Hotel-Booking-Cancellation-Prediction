import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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