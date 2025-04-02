# Titanic Survival Prediction

This project predicts the survival of Titanic passengers using the **Decision Tree algorithm** with a **pipeline** for preprocessing and model training.

## 📌 Features

- Uses **DecisionTreeClassifier** from `sklearn`
- Implements **Pipeline** for efficient preprocessing and training
- Handles missing values, encodes categorical features, scales numerical features, and selects best features
- Evaluates model performance with accuracy score and confusion matrix
- Deploys model using **Render** for easy access via API

## 📂 Project Structure

```
├── data
│   ├── train.csv  # Training dataset
│   ├── test.csv   # Test dataset
├── titanic_pipeline.ipynb  # Jupyter Notebook with the pipeline
├── model.pkl  # Saved Decision Tree model
├── app.py  # Flask app for deployment
├── requirements.txt  # Dependencies
├── render.yaml  # Render deployment configuration
├── README.md  # Project documentation
```

## 🛠️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to train and evaluate the model:
   ```bash
   jupyter notebook titanic_pipeline.ipynb
   ```

## 🔍 Data Preprocessing

The pipeline includes:

- **Handling Missing Values**:
  - Filling missing values for `Age` using median.
  - Filling missing values for `Embarked` using the most frequent value.
- **Encoding Categorical Features**: One-hot encoding for `Sex` and `Embarked`.
- **Feature Scaling**: Normalizing numerical features using MinMaxScaler.
- **Feature Selection**: Using `SelectKBest` with the chi-square test to select the top 8 features.
- **Model Training**: Applying `DecisionTreeClassifier` with optimized parameters.

## 🚀 Model Training

The pipeline is implemented using:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2

# Define transformations
trf1 = ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])
], remainder='passthrough')

trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [1, 6])
], remainder='passthrough')

trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0, 10, None))
])

trf4 = SelectKBest(score_func=chi2, k=8)
trf5 = DecisionTreeClassifier()

# Create full pipeline
model_pipeline = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', trf4),
    ('trf5', trf5)
])
```

## 🌐 Deployment with Render

1. Install **Flask** to create a simple API:
   ```bash
   pip install flask
   ```
2. Create `app.py` to serve predictions:
   ```python
   from flask import Flask, request, jsonify
   import pickle
   import numpy as np

   app = Flask(__name__)

   model = pickle.load(open('model.pkl', 'rb'))

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.get_json()
       prediction = model.predict(np.array(data['features']).reshape(1, -1))
       return jsonify({'prediction': int(prediction[0])})

   if __name__ == '__main__':
       app.run(debug=True)
   ```
3. Create a `render.yaml` file:
   ```yaml
   services:
     - type: web
       name: titanic-survival-api
       env: python
       buildCommand: "pip install -r requirements.txt"
       startCommand: "python app.py"
       plan: free
   ```
4. Deploy on **Render**:
   - Push code to GitHub
   - Connect repository to Render
   - Render automatically builds and deploys the application

5. **Access the Deployed Model**:
   The model is deployed at:
   🔗 [Titanic Survival API](https://titanic-survival-prediction-mb5x.onrender.com/)

## 📊 Evaluation

- **Accuracy Score**
- **Confusion Matrix**
- **Feature Importance Analysis**

## 🏆 Results

After training the model on the Titanic dataset, we achieved an accuracy of **64%** on the validation set. Further tuning can improve performance.

## 🤝 Contributing

Pull requests are welcome! If you'd like to improve this project, feel free to fork and submit PRs.

## 📜 License

This project is open-source under the **MIT License**.

---

Happy Coding! 🚀

