import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from preprocessor import DataPreprocessor
from predictor import ModelPredictor
from sklearn.ensemble import RandomForestClassifier
import gdown

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    layout="wide"
)
def download_model_from_drive(file_id, output='best_model.pkl'):
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
file_id = '1aU8FtElbZQLK4b1_nE_gU3WjzyjmpXk8'
download_model_from_drive(file_id)

MODEL_PATH = 'best_modell.pkl'
download_model_from_drive(file_id, MODEL_PATH)

preprocessor = DataPreprocessor()
predictor = ModelPredictor(MODEL_PATH, preprocessor)

def main():
    st.title("Hotel Booking Cancellation Predictor")
    st.write("""
    This application predicts whether a hotel booking will be cancelled or not based on various booking details.
    Fill in the booking information below and click on 'Predict' to see the results.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Batch Prediction", "Test Cases"])
    
    with tab1:
        st.header("Booking Prediction")
        
        with st.form(key="booking_form"):
            col1, col2 = st.columns(2)
            
            with col2:
                no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
                no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
                no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, max_value=7, value=1)
                no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, max_value=30, value=3)
                type_of_meal_plan = st.selectbox(
                    "Type of Meal Plan",
                    options=["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
                    index=0
                )
                required_car_parking_space = st.checkbox("Required Car Parking Space")
                room_type_reserved = st.selectbox(
                    "Room Type Reserved",
                    options=["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"],
                    index=0
                )
            
            with col1:
                lead_time = st.slider("Lead Time (days)", min_value=0, max_value=365, value=90)
                arrival_year = st.number_input("Arrival Year", min_value=2020, max_value=2030, value=2023)
                arrival_month = st.slider("Arrival Month", min_value=1, max_value=12, value=7)
                arrival_date = st.slider("Arrival Date", min_value=1, max_value=31, value=15)
                market_segment_type = st.selectbox(
                    "Market Segment Type",
                    options=["Online", "Offline", "Corporate", "Aviation", "Complementary"],
                    index=0
                )
                repeated_guest = st.checkbox("Repeated Guest")
                no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, max_value=10, value=0)
                no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings (Not Canceled)", min_value=0, max_value=10, value=0)
                avg_price_per_room = st.number_input("Average Price Per Room", min_value=0.0, max_value=1000.0, value=120.5)
                no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=1)
            
            submit_button = st.form_submit_button("Predict")
        
        if submit_button:
            booking_data = {
                'no_of_adults': no_of_adults,
                'no_of_children': no_of_children,
                'no_of_weekend_nights': no_of_weekend_nights,
                'no_of_week_nights': no_of_week_nights,
                'type_of_meal_plan': type_of_meal_plan,
                'required_car_parking_space': int(required_car_parking_space),
                'room_type_reserved': room_type_reserved,
                'lead_time': lead_time,
                'arrival_year': arrival_year,
                'arrival_month': arrival_month,
                'arrival_date': arrival_date,
                'market_segment_type': market_segment_type,
                'repeated_guest': int(repeated_guest),
                'no_of_previous_cancellations': no_of_previous_cancellations,
                'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
                'avg_price_per_room': avg_price_per_room,
                'no_of_special_requests': no_of_special_requests
            }
            
            with st.spinner("Making prediction..."):
                result = predictor.predict(booking_data)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                if result["prediction"] == "Canceled":
                    st.error(f"⚠️ Booking is predicted to be **CANCELLED** with {result['cancellation_probability']*100:.2f}% probability.")
                else:
                    st.success(f"✅ Booking is predicted to be **NOT CANCELLED** with {(1-result['cancellation_probability'])*100:.2f}% probability.")
                
                with st.expander("View detailed prediction result"):
                    st.json(result)
    
    with tab2:
        st.header("Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV file with booking data", type=["csv"])
        
        if uploaded_file is not None:
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                df = pd.read_csv("temp_upload.csv")
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                required_columns = ['no_of_adults', 'no_of_children', 'type_of_meal_plan', 
                                    'room_type_reserved', 'lead_time', 'market_segment_type']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    if st.button("Process Batch"):
                        with st.spinner("Processing batch predictions..."):
                            results = predictor.batch_predict("temp_upload.csv")
                        
                        if results.empty:
                            st.error("Error processing the file. Please check the format and try again.")
                        else:
                            st.success("Batch processing completed!")
                            
                            st.write("Prediction Results:")
                            st.dataframe(results)
                            
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="batch_prediction_results.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                
            if os.path.exists("temp_upload.csv"):
                os.remove("temp_upload.csv")
    
    with tab3:
        st.header("Test Cases")
        
        st.subheader("Test Case 1: Cancel")
        st.write("""
        This is a booking with characteristics that make it likely to be cancelled:
        - Long lead time (180 days)
        - High room price
        - Not a repeated guest
        - Online booking
        """)
        
        test_case_1 = {
            'no_of_adults': 2,
            'no_of_children': 1,
            'no_of_weekend_nights': 2,
            'no_of_week_nights': 1,
            'type_of_meal_plan': 'Meal Plan 2',
            'required_car_parking_space': 0,
            'room_type_reserved': 'Room_Type 2',
            'lead_time': 180,
            'arrival_year': 2023,
            'arrival_month': 8,
            'arrival_date': 15,
            'market_segment_type': 'Online',
            'repeated_guest': 0,
            'no_of_previous_cancellations': 1,
            'no_of_previous_bookings_not_canceled': 0,
            'avg_price_per_room': 150.0,
            'no_of_special_requests': 0
        }
        
        with st.expander("View test case details"):
            st.json(test_case_1)
        
        if st.button("Run Test Case 1"):
            with st.spinner("Making prediction..."):
                result = predictor.predict(test_case_1)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                if result["prediction"] == "Canceled":
                    st.error(f"⚠️ Test Case 1: Booking is predicted to be **CANCELLED** with {result['cancellation_probability']*100:.1f}% probability.")
                else:
                    st.success(f"✅ Test Case 1: Booking is predicted to be **NOT CANCELLED** with {(1-result['cancellation_probability'])*100:.1f}% probability.")
                
                st.json(result)
        
        st.subheader("Test Case 2: Not Cancel")
        st.write("""
        This is a booking with characteristics that make it unlikely to be cancelled:
        - Short lead time (7 days)
        - Repeated guest
        - Corporate booking
        - Multiple special requests
        """)
        
        test_case_2 = {
            'no_of_adults': 1,
            'no_of_children': 0,
            'no_of_weekend_nights': 0,
            'no_of_week_nights': 3,
            'type_of_meal_plan': 'Meal Plan 1',
            'required_car_parking_space': 1,
            'room_type_reserved': 'Room_Type 1',
            'lead_time': 7,
            'arrival_year': 2023,
            'arrival_month': 6,
            'arrival_date': 10,
            'market_segment_type': 'Corporate',
            'repeated_guest': 1,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 2,
            'avg_price_per_room': 95.5,
            'no_of_special_requests': 3
        }
        
        with st.expander("View test case details"):
            st.json(test_case_2)
        
        if st.button("Run Test Case 2"):
            with st.spinner("Making prediction..."):
                result = predictor.predict(test_case_2)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                if result["prediction"] == "Canceled":
                    st.error(f"⚠️ Test Case 2: Booking is predicted to be **CANCELLED** with {result['cancellation_probability']*100:.1f}% probability.")
                else:
                    st.success(f"✅ Test Case 2: Booking is predicted to be **NOT CANCELLED** with {(1-result['cancellation_probability'])*100:.1f}% probability.")
                
                st.json(result)

if __name__ == "__main__":
    main()