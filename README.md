# 🏨 Hotel Booking Cancellation Predictor

![Python](https://skillicons.dev/icons?i=python) ![Scikit-learn](https://skillicons.dev/icons?i=scikitlearn) ![XGBoost](https://skillicons.dev/icons?i=xgboost) ![Pandas](https://skillicons.dev/icons?i=pandas) ![Streamlit](https://skillicons.dev/icons?i=streamlit)

**Course:** Model Deployment
**Dataset:** Hotel Booking Dataset  
**Year:** 2025  

## About This Project
This project is a Machine Learning-based web application that predicts whether a hotel booking will be canceled or confirmed based on customer and booking details.
The system is built using Streamlit for the frontend and a trained model for backend prediction.

## Dataset Features
The dataset includes the following features:
- `Booking_ID`: Unique identifier for each booking  
- `no_of_adults`: Number of adults  
- `no_of_children`: Number of children  
- `no_of_weekend_nights`: Number of weekend nights  
- `no_of_week_nights`: Number of week nights  
- `type_of_meal_plan`: Meal plan type  
- `required_car_parking_space`: Parking requirement (0=No, 1=Yes)  
- `room_type_reserved`: Encrypted room type  
- `lead_time`: Days between booking and arrival  
- `arrival_year`, `arrival_month`, `arrival_date`: Arrival date details  
- `market_segment_type`: Market segment  
- `repeated_guest`: Flag if guest has booked before  
- `no_of_previous_cancellations`: Previous cancellations  
- `no_of_previous_bookings_not_canceled`: Previous confirmed bookings  
- `avg_price_per_room`: Average price per room (Euros)  
- `no_of_special_requests`: Total special requests  
- `booking_status` (Target): Cancellation flag (0=Not canceled, 1=Canceled)


## Project Workflow

### 1. Exploratory Data Analysis (EDA) & Preprocessing
- Explored missing values, data distribution, and outliers  
- Handled categorical encoding, scaling, and feature selection  
- Split data into train/test sets  

### 2. Model Training & Comparison
- Trained **Random Forest** and **XGBoost** classifiers  
- Evaluated models using accuracy, F1-score, and ROC-AUC  
- Selected the best performing model and saved it using **pickle**  

### 3. OOP Implementation
- Refactored the training pipeline into classes and methods  
- Encapsulated preprocessing, model training, and evaluation in OOP design  

### 4. Inference / Prediction Script
- Created script to make predictions using the saved model  
- Provided input interface for new booking data  

### 5. Streamlit Deployment
- Built interactive Streamlit web app for model predictions  
- Included **2 test cases** for demonstration  
- Users can input booking details and get cancellation probability  


## Technologies Used
- Python  
- Pandas & NumPy  
- scikit-learn & XGBoost  
- Matplotlib & Seaborn  
- Streamlit  
- Pickle (for model saving)  

🔗 **Live Demo:** [Hotel Booking Predictor](https://hotelbookingpredictor.streamlit.app/)
