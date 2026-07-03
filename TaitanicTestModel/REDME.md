# 🚢 Titanic Survival Prediction using Logistic Regression

## Overview

This project implements a **Logistic Regression** model to predict whether a passenger survived the Titanic disaster based on various passenger attributes. The project demonstrates a complete machine learning workflow, including data loading, preprocessing, visualization, model training, evaluation, and model persistence using **Joblib**.

The implementation is built using **Python**, **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, and **Seaborn**.

---

# 📌 Features

* Load Titanic dataset from CSV
* Display dataset information and statistics
* Handle missing values
* Remove unnecessary columns
* Encode categorical variables
* Train a Logistic Regression model
* Evaluate model performance
* Display confusion matrix
* Visualize survival distribution
* Save the trained model using Joblib
* Load the saved model for prediction

---

# 🛠 Technologies Used

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

# 📂 Project Structure

```text
Titanic-Survival-Prediction/
│
├── MarvellousTitanicDataset.csv
├── TitanicLogisticRegression.py
├── marvelloustitanic.pkl
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

The dataset contains passenger information from the Titanic disaster.

### Input Features

* Passenger Class (Pclass)
* Sex
* Age
* SibSp (Number of siblings/spouses aboard)
* Parch (Number of parents/children aboard)
* Fare
* Embarked

### Target Variable

* **Survived**

  * 0 → Did Not Survive
  * 1 → Survived

---

# ⚙️ Machine Learning Workflow

1. Load the dataset
2. Explore the dataset
3. Remove unnecessary columns
4. Handle missing values
5. Encode categorical variables
6. Split the dataset into training and testing sets
7. Train the Logistic Regression model
8. Save the trained model
9. Load the saved model
10. Predict survival
11. Evaluate the model
12. Visualize the results

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Removing unnecessary columns

  * PassengerId
  * Name
  * Cabin
  * zero
* Handling missing values

  * Age → Median
  * Fare → Median
  * Embarked → Mode
* Encoding categorical variables
* Converting boolean values to integers

---

# 🤖 Model

Algorithm used:

**Logistic Regression**

```python
model = LogisticRegression(
    max_iter=1000
)
```

---

# 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy Score
* Confusion Matrix

---

# 💾 Model Persistence

The trained model is saved using **Joblib**.

Save model:

```python
joblib.dump(model, "marvelloustitanic.pkl")
```

Load model:

```python
loaded_model = joblib.load("marvelloustitanic.pkl")
```

---

# 📊 Data Visualization

The project visualizes:

* Passenger Survival Count
* Survival Distribution using Seaborn Countplot

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Titanic-Survival-Prediction.git
```

Navigate to the project folder:

```bash
cd Titanic-Survival-Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Or install using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Place the dataset file (`MarvellousTitanicDataset.csv`) in the project directory.

Run the project:

```bash
python TitanicLogisticRegression.py
```

---

# 📋 Sample Output

```text
======================================================================
Step 1 : Loading the Dataset
======================================================================

Initial Dataset Loaded Successfully

Shape of Dataset : (891, 12)

Missing Values Handled Successfully

Model Trained Successfully

Accuracy : 81.6%

Confusion Matrix

[[91 14]
 [19 55]]

Model Saved Successfully

Model Loaded Successfully
```

---

# 📚 Learning Outcomes

This project demonstrates:

* Data preprocessing
* Handling missing values
* Exploratory Data Analysis (EDA)
* Feature engineering
* Logistic Regression
* Model training and testing
* Model serialization with Joblib
* Data visualization
* Machine Learning workflow

---

# 🎯 Future Enhancements

* Add ROC Curve and AUC Score
* Display Precision, Recall, and F1-Score
* Hyperparameter tuning using GridSearchCV
* Feature importance analysis
* Cross-validation
* Build a Streamlit web application
* Deploy using Flask or Django
* Add real-time passenger survival prediction

---

# 📦 Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

# 👨‍💻 Author

**Yash Chandrakant Gaikwad**

Electronics & Telecommunication Engineering Student

### Skills

* Python
* Machine Learning
* Data Analysis
* Scikit-learn
* Django
* MySQL
* AI/ML

---

# 📄 License

This project is open-source and available for educational and learning purposes.

---

# 🙏 Acknowledgements

* Scikit-learn Documentation
* Pandas Documentation
* NumPy Documentation
* Matplotlib Documentation
* Seaborn Documentation
* Kaggle Titanic Dataset

