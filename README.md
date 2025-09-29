# 📈 NiftyTradingModel  

A machine learning–powered trading model that predicts buy/sell/do-nothing signals from OHLCV+OI (Open, High, Low, Close, Volume, Open Interest) market data.  
This project uses **feature engineering + XGBoost** to generate trading predictions, wrapped in a simple **Streamlit app** for interactive use.  

🚀 Features

📊 Interactive Streamlit App — deployed on Streamlit Cloud.

⚡ Multi-class Classification Model predicting 5 trading actions.

🔍 Exploratory Data Analysis (EDA) with feature engineering.

🧠 Model Training with hyperparameter tuning.

🧪 Performance Evaluation with statistical tests.

📑 Export Reports for insights.

🏗️ Tech Stack

Frontend & Deployment: Streamlit

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Modeling: Scikit-learn

Statistical Analysis: SciPy, statsmodels


--- 

## 🏗️ Project Structure
```bash
NiftyTradingModel/
├── Visualizations #all the plotting and designs
├── app.py # Streamlit app
├── feature_engineer.py # Feature engineering + preprocessing
├── xgb_model.pkl # Trained model
├── requirements.txt # Dependencies
└── README.md # Documentation
```
---
📊 Model Results
🔹 Confusion Matrix

🔹 Classification Report
Class	Precision	Recall	F1-Score	Support
Buy+	0.55	0.47	0.51	33,342
Buy	0.49	0.42	0.45	33,376
Do Nothing	0.64	0.86	0.73	33,571
Sell+	0.46	0.47	0.47	33,809
Sell	0.53	0.49	0.51	33,414
Accuracy			0.54	167,512
Macro Avg	0.53	0.54	0.53	167,512
Weighted Avg	0.53	0.54	0.53	167,512
🔹 Statistical Tests

Chi-square Test (Confusion Matrix): χ² = 133,807.85, df = 16, p < 0.00001

Cramér’s V: 0.447 (moderate association)

Cohen’s Kappa: 0.428 (moderate agreement beyond chance)

---

## ⚙️ Installation (Local Setup)  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/heubert-69/NiftyTradingModel.git
cd NiftyTradingModel
```
2️⃣ Create virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
🚀 Usage
1️⃣ Run the Streamlit app locally
```bash
streamlit run app.py
```
2️⃣ Upload your CSV
CSV must have columns:

```arduino
datetime, open, high, low, close, volume, oi
```
3️⃣ View Predictions
Model outputs:

0 → Buy+

1 → Buy

2 → Do Nothing

3 → Sell

4 → Sell+

Predictions are displayed in a table and can be downloaded as predictions.csv.

📂 Example Input
CSV Example:

```csv
datetime,open,high,low,close,volume,oi
2024-01-01 09:15:00,21750,21800,21700,21780,15000,12000
2024-01-01 09:20:00,21780,21820,21740,21760,18000,12200
```
🛠️ Development
Feature Engineering
Located in feature_engineer.py:

engineer_features_and_labels(df) → generates features + labels

preprocess_for_sklearn(X) → converts datetime/object to numeric + handles NaNs

Training
Model was trained using engineered features and exported via:

```python
joblib.dump(model, "xgb_model.pkl")
```

📦 Requirements
See requirements.txt:

pandas

numpy

scikit-learn

xgboost

streamlit

matplotlib

imbalanced-learn

🌐 Deployment
The app is live here: https://niftytradingmodel-etxjms2gsmi5ruid6yrhmf.streamlit.app/
👉 Streamlit Cloud Demo

To redeploy:

Push updates to your GitHub repo.

Streamlit Cloud will auto-deploy changes.

📜 License
This project is licensed under the MIT License.

🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss.

👨‍💻 Author
Voltsy (Heubert-69)

GitHub: @heubert-69

