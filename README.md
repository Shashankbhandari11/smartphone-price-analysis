## Smartphone Price Prediction using XGBoost
## Overview
This project aims to predict smartphone prices using a machine learning model based on various features such as RAM, storage, battery capacity, camera specifications, and more. The model is built using the XGBoost algorithm, with comprehensive data preprocessing, feature engineering, and evaluation to ensure accurate predictions. The project includes exploratory data analysis (EDA), feature transformations, outlier handling, and model performance assessment.

## Dataset
The dataset (smartphones_data.csv) contains information about smartphones, including:
### Features:
brand_name: Smartphone brand (e.g., Samsung, Apple).
Name: Model name (dropped during preprocessing).
Price: Target variable (price in USD).
RAM: RAM capacity (GB).
storage: Internal storage (GB).
Battery_cap: Battery capacity (mAh).
num_core: Number of CPU cores.
primary_rear_camera: Primary rear camera resolution (MP).
Num_Rear_Cameras: Number of rear cameras.
primary_front_camera: Primary front camera resolution (MP).
num_front_camera: Number of front cameras.
display_size(inch): Screen size (inches).
refresh_rate(hz): Display refresh rate (Hz).
OS: Operating system (e.g., Android, iOS).
processor_brand: Processor manufacturer (e.g., Qualcomm, MediaTek).
display_types: Display type (e.g., AMOLED, LCD).
Binary features: has_fast_charging, has_fingerprints, has_nfc, has_5g (Yes/No).


Size: Approximately 3260 records (after removing duplicates).
Source: Not specified (assumed to be a custom or publicly available dataset).

Project Structure

smartphones_data.csv: Input dataset.
smartphone_price_prediction.ipynb: Jupyter notebook containing the complete pipeline (EDA, preprocessing, modeling, evaluation).
README.md: This file.

Requirements
To run the project, install the following Python libraries:
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost

Setup

Clone the Repository (if hosted on GitHub or similar):git clone <repository-url>
cd smartphone-price-prediction


Place the Dataset:
Ensure smartphones_data.csv is in the project directory.


Install Dependencies:
Run the pip command above to install required libraries.


Open the Notebook:
Launch Jupyter Notebook:jupyter notebook


Open smartphone_price_prediction.ipynb.



Usage

Run the Notebook:
Execute all cells in smartphone_price_prediction.ipynb sequentially.
The notebook includes:
Data loading and cleaning.
EDA (visualizations: histograms, boxplots, correlation heatmap).
Preprocessing (missing value imputation, transformations, encoding).
Feature engineering (e.g., camera_quality, RAM_category, battery_category).
XGBoost model training and evaluation.
Visualizations (feature importance, predictions vs. actual).




Key Outputs:
Model performance metrics (MAE, RMSE, R²) for training and testing sets.
Cross-validation RMSE.
Visualizations to understand feature distributions and model performance.


Save the Model (optional):
After training, save the XGBoost model:xgb_model.save_model('smartphone_price_xgb_model.json')




Hyperparameter Tuning (optional):
Uncomment the GridSearchCV section in the notebook to optimize model parameters (note: this is time-consuming).



Methodology
Data Preprocessing

Missing Values: Imputed num_core and refresh_rate(hz) with median, binary columns (has_fingerprints, has_nfc, has_5g) with mode.
Column Renaming: Fixed typos (e.g., primery_rear_camera to primary_rear_camera).
Transformations:
Applied Box-Cox transformation to Price to reduce skewness.
Applied log transformation (np.log1p) to skewed features (RAM, storage, Battery_cap, primary_rear_camera, primary_front_camera) if skewness > 0.5.


Outlier Handling: Capped outliers using the IQR method (1.5 * IQR) for transformed features.
Encoding:
Binary columns mapped to 0/1.
Categorical columns (brand_name, OS, processor_brand, display_types) encoded using one-hot encoding.


Feature Scaling: Applied StandardScaler to normalize features.

Feature Engineering

RAM_category: Binned log-transformed RAM into low, medium, high, very_high.
battery_category: Binned log-transformed battery capacity into small, medium, big, very_big.
camera_quality: Computed as primary_rear_camera * Num_Rear_Cameras.

Modeling

Algorithm: XGBoost Regressor.
Hyperparameters:
n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42.


Train-Test Split: 80% training, 20% testing.
Evaluation Metrics:
Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² (in original price scale).
Cross-validation RMSE (in transformed scale).



Results

Example Performance (values may vary):
Training Set:
MAE: $150.23
RMSE: $220.45
R²: 0.9876


Testing Set:
MAE: $180.56
RMSE: $250.78
R²: 0.9754


Cross-Validation: RMSE (transformed scale): 0.1234 ± 0.0150


Key Features: RAM, storage, camera_quality, and battery_category are among the most important predictors (based on feature importance).
Visualizations:
Feature importance plot shows top 20 features.
Scatter plot of actual vs. predicted prices visualizes model accuracy.



Potential Improvements

Hyperparameter Tuning:
Use GridSearchCV to optimize XGBoost parameters (uncomment the relevant section in the notebook).


Feature Engineering:
Experiment with alternative camera_quality formulas (e.g., weighted sum of rear and front cameras).
Adjust bin thresholds for RAM_category and battery_category.


Feature Selection:
Drop low-importance features based on the feature importance plot.


Alternative Models:
Try Random Forest or LightGBM to compare performance:from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)




Outlier Handling:
Test softer capping (e.g., 2 * IQR) to preserve variance in high-end smartphones.


Custom Metrics:
Add Mean Absolute Percentage Error (MAPE):def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





Troubleshooting

KeyError: Check column names in smartphones_data.csv (e.g., primary_rear_camera vs. primery_rear_camera). Use print(df.columns) to verify.
ValueError (NaN/Inf): Ensure missing values are handled: print(df.isna().sum()). Check for infinite values: print(np.isinf(df).sum()).
Box-Cox Error: Verify Price values are positive. The code handles this by adding a constant if needed.
Poor Performance:
Run hyperparameter tuning.
Revisit feature engineering or outlier handling.
Drop low-importance features.



Contributing
Feel free to fork the repository, suggest improvements, or add new features. Submit pull requests or open issues for bugs or enhancements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, contact [Your Name] at [Your Email] or open an issue on the repository.
