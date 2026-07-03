# 🌸 Iris Flower Classification using Decision Tree

A Machine Learning project that classifies Iris flowers into three different species using the **Decision Tree Classification** algorithm. This project demonstrates the complete machine learning workflow, including data analysis, visualization, model training, prediction, and performance evaluation.

---

## 📌 Project Overview

The Iris dataset is one of the most popular datasets in machine learning. It contains measurements of iris flowers from three species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

Using these measurements, a **Decision Tree Classifier** is trained to predict the species of an iris flower.

---

## 🚀 Features

* Load and analyze the Iris dataset
* Perform Exploratory Data Analysis (EDA)
* Visualize the dataset using scatter plots
* Split data into training and testing sets
* Train a Decision Tree Classifier
* Predict flower species
* Evaluate model performance
* Display Accuracy Score
* Generate Confusion Matrix
* Generate Classification Report

---

## 🛠️ Technologies Used

* Python 3
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── iris.csv
├── IrisDecisionTree.py
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The project uses the **Iris Dataset**, which contains:

* **150 Samples**
* **4 Features**

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* **3 Classes**

  * Setosa
  * Versicolor
  * Virginica

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Iris-Flower-Classification.git
```

Navigate to the project folder:

```bash
cd Iris-Flower-Classification
```

Install the required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Or install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Make sure the `iris.csv` dataset is present in the project directory.

Run the project:

```bash
python IrisDecisionTree.py
```

---

## 📈 Machine Learning Workflow

1. Load the dataset
2. Analyze the dataset
3. Visualize the data
4. Split data into training and testing sets
5. Train the Decision Tree model
6. Make predictions
7. Evaluate the model
8. Display the confusion matrix

---

## 📊 Model Configuration

* Algorithm: Decision Tree Classifier
* Criterion: Gini Index
* Maximum Depth: 3
* Random State: 42

---

## 📋 Evaluation Metrics

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

---

## 📌 Sample Output

```text
----------------------------------------
Step 1 : Data Analysis
----------------------------------------

Shape of Dataset : (150, 5)

Missing Values:
0

Class Distribution:

setosa        50
versicolor    50
virginica     50

----------------------------------------
Step 4 : Train the Model
----------------------------------------

DecisionTreeClassifier(max_depth=3)

Model Training Completed

Accuracy : 100.00 %

Confusion Matrix

[[10 0 0]
 [0 9 0]
 [0 0 11]]
```

---

## 🎯 Future Enhancements

* Visualize the Decision Tree
* Hyperparameter tuning using GridSearchCV
* Compare with other classification algorithms
* Cross-validation
* Feature importance analysis
* Build a Streamlit or Flask web application
* Deploy the model to the cloud

---

## 📚 Learning Outcomes

This project helps understand:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization
* Decision Tree Classification
* Train-Test Split
* Model evaluation techniques
* Confusion Matrix interpretation
* Classification metrics in Scikit-learn

---

## 📦 Requirements

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 👨‍💻 Author

**Yash Chandrakant Gaikwad**

Electronics & Telecommunication Engineering Student

### Skills

* Python
* Machine Learning
* Data Analysis
* Scikit-learn
* Pandas
* Django
* MySQL
* AI/ML

---

## 📄 License

This project is open-source and available for educational and learning purposes.

---

## Acknowledgements

* Scikit-learn
* Pandas
* Matplotlib
* Seaborn
* UCI Machine Learning Repository

