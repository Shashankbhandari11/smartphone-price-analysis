# Smartphone Price Prediction using XGBoost

##  Overview
This project aims to predict smartphone prices using a machine learning model based on various features such as RAM, storage, battery capacity, camera specifications, and more. The model is built using the **XGBoost** algorithm, with comprehensive data preprocessing, feature engineering, and evaluation to ensure accurate predictions. It includes exploratory data analysis (EDA), feature transformations, outlier handling, and model performance assessment.

---

##  Dataset
The dataset (`smartphones_data.csv`) contains information about smartphones, including:

### Features:
- `brand_name`: Smartphone brand (e.g., Samsung, Apple)
- `Name`: Model name  
- `Price`: Target variable (price in USD)  
- `RAM`: RAM capacity (GB)  
- `storage`: Internal storage (GB)  
- `Battery_cap`: Battery capacity (mAh)  
- `num_core`: Number of CPU cores  
- `primary_rear_camera`: Primary rear camera resolution (MP)  
- `Num_Rear_Cameras`: Number of rear cameras  
- `primary_front_camera`: Primary front camera resolution (MP)  
- `num_front_camera`: Number of front cameras  
- `display_size(inch)`: Screen size (inches)  
- `refresh_rate(hz)`: Display refresh rate (Hz)  
- `OS`: Operating system (e.g., Android, iOS)  
- `processor_brand`: Processor manufacturer (e.g., Qualcomm, MediaTek)  

---

##  Project Structure

- `smartphones_data.csv`: Input dataset  
- `smartphone_price_prediction.ipynb`: Jupyter Notebook with the full ML pipeline (EDA, preprocessing, modeling, evaluation)  
- `README.md`: Project documentation  

---

##  Requirements

Install required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
```

## Clone the Repository:
```bash
git clone https://github.com/Shashankbhandari11/smartphone-price-analysis.git
cd smartphone-price-analysis
```

---
## Run the Notebook:
Open and run all cells in bash```
smartphone_price_prediction.ipynb

---
##  Dataset
Data loading and cleaning

EDA (visualizations: histograms, boxplots, correlation heatmap)

Preprocessing (missing value handling, transformations, encoding)

Feature engineering (e.g., camera_quality, RAM_category, battery_category)

XGBoost model training and evaluation

Visualizations (feature importance, actual vs predicted prices)

## Key Outputs
- Model performance metrics:

- MAE (Mean Absolute Error)

- RMSE (Root Mean Square Error)

- R² Score



## Contributing
Feel free to fork the repository, suggest improvements, or add new features.
Submit pull requests or open issues for bugs or enhancements.

## License
This project is licensed under the MIT License.
