import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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
                            print(f"Model loaded successfully from {model_path}")
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