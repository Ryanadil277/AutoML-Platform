AutoML Platform

An Interactive Automated Machine Learning Platform for Classification and Regression Tasks.

Overview


<video controls src="2026-04-23 23-08-12.mp4" title="
"></video>
AutoML Platform is a web-based application designed to simplify the machine learning workflow by automating data preprocessing, exploratory data analysis, model training, evaluation, and hyperparameter tuning.

The platform allows users to upload their datasets, select a target variable, choose a machine learning task, train different models, compare their performance, and save the best-performing model.

Key Features

1. Data Upload

- Upload datasets in CSV format.
- Preview uploaded data.
- Explore dataset structure and information.

2. Data Preprocessing

- Handle missing values.
- Prepare features for machine learning.
- Encode categorical variables.
- Prepare data for model training.

3. Exploratory Data Analysis (EDA)

- Correlation analysis.
- Data distribution visualization.
- Feature exploration.

4. Machine Learning Tasks

The platform supports two main machine learning tasks:

- Classification: Predict categorical outcomes.
- Regression: Predict continuous numerical values.

5. Model Selection and Training

- Select the target variable.
- Choose individual machine learning models.
- Train multiple models.
- Compare model performance.

6. Model Evaluation

Evaluate trained models using appropriate performance metrics.

Classification Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Regression Metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

7. Hyperparameter Tuning

Optimize model performance through hyperparameter tuning to identify better model configurations.

8. Model Saving and Export

Save and download the best-performing trained model for future use.

Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

Project Structure

AutoML-Platform/
│
├── app.py
├── app_old.py
├── model_saving.py
├── tuning.py
├── requirements.txt
├── testing_data_final.csv
│
└── README.md

Note: Update this structure to match the actual files in the repository.

Installation and Setup

1. Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the Project Directory

cd AutoML-Platform

3. Create a Virtual Environment

python -m venv .venv

4. Activate the Virtual Environment

Windows:

.venv\Scripts\activate

5. Install Dependencies

pip install -r requirements.txt

6. Run the Application

streamlit run app.py

Project Goal

The main goal of this project is to make machine learning more accessible by providing an interactive platform that automates essential ML processes and reduces the need for extensive manual coding.

Future Improvements

- Support additional machine learning algorithms.
- Advanced feature selection.
- More hyperparameter optimization techniques.
- Enhanced model comparison and visualization.
- Support for additional dataset formats.

---

AutoML Platform – Simplifying Machine Learning Through Automation.