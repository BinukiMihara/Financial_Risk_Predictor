# Financial_Risk_Prediction
An attempt to build a machine learning model capable of predicting the rate of someone facing financial risk being low, medium or high based on their financial and social attributes.
Check the application - https://financial-risk.streamlit.app/

Key features of the prediction system:

 • Data Cleaning & Preprocessing: Treated the missing values, check for outliers, took care of class imbalances, normalized and transformed data, and converted categorical variables into numeric format. 

 • Feature Scaling and selection: Applied MaxAbsScaler to ensure that the numerical features were standardized for the models. Using Mutual Information (MI) scores, the most informative features were selected. 

 • Model Training and Evaluation: The models XGBoost, CatBoost, LightGBM, and RandomForest were taken into consideration. The scaled and resampled dataset was used to train each model. After training, models were assessed using confusion matrices, accuracy, and classification reports. For hyperparameter tuning used RandomizedSearchCV and GridSearchCV.

 • Model Saving and Deployment: Because of its high training accuracy and test accuracy following hyperparameter adjustment, XGBoost was chosen as the top-performing model. Finally, the app was deployed as a Streamlit application. 

