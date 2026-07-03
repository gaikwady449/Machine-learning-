# 🏏🎾 Ball Classification using Decision Tree

## Overview

This project demonstrates a simple **Decision Tree Classification** model using **Scikit-learn** to classify an object as either a **Tennis Ball** or a **Cricket Ball** based on its physical characteristics.

The model is trained on a small manually created dataset using two input features:

* Weight (grams)
* Surface Texture (Smooth or Rough)

This beginner-friendly project introduces the fundamentals of supervised machine learning using the Decision Tree algorithm.

---

## Features

* Manual dataset creation
* Feature encoding for categorical values
* Decision Tree model training
* Ball type prediction
* Simple command-line output
* Easy-to-understand implementation

---

## Technologies Used

* Python 3
* Scikit-learn

---

## Project Structure

```text
Ball-Classification/
│
├── BallClassification.py
├── README.md
└── requirements.txt
```

---

## Dataset

### Input Features

| Feature | Description                |
| ------- | -------------------------- |
| Weight  | Weight of the ball (grams) |
| Surface | 0 = Smooth, 1 = Rough      |

### Target Labels

| Label | Ball Type    |
| ----- | ------------ |
| 1     | Tennis Ball  |
| 2     | Cricket Ball |

### Sample Training Data

| Weight | Surface | Ball         |
| -----: | :-----: | ------------ |
|     35 |  Rough  | Tennis Ball  |
|     47 |  Rough  | Tennis Ball  |
|     90 |  Smooth | Cricket Ball |
|     48 |  Rough  | Tennis Ball  |
|     96 |  Smooth | Cricket Ball |

---

## Machine Learning Workflow

1. Create the training dataset
2. Encode categorical values
3. Train the Decision Tree Classifier
4. Predict the ball type
5. Display the prediction

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Ball-Classification.git
```

Navigate to the project folder:

```bash
cd Ball-Classification
```

Install the required package:

```bash
pip install scikit-learn
```

---

## How to Run

Run the Python script:

```bash
python BallClassification.py
```

---

## Example Output

```text
Ball Classification Case Study

Split Dataset Into Independent And Dependent Variables

Model Prediction:
Object looks like Tennis Ball

Predicted Label:
[1]
```

---

## Decision Tree Algorithm

The project uses Scikit-learn's `DecisionTreeClassifier` with the default configuration.

```python
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(Xtrain, Ytrain)

prediction = model.predict([[35, 1]])
```

---

## Learning Outcomes

This project helps you understand:

* Supervised Machine Learning
* Decision Tree Classification
* Feature Encoding
* Model Training
* Prediction using Scikit-learn
* Working with small datasets

---

## Future Enhancements

* Use a larger dataset
* Read training data from a CSV file
* Evaluate model accuracy using a test set
* Display a confusion matrix
* Visualize the Decision Tree
* Build a GUI using Tkinter
* Develop a web interface using Flask or Django

---

## Requirements

* Python 3.8 or above
* Scikit-learn

Install dependencies:

```bash
pip install scikit-learn
```

---

## Author

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

## License

This project is open-source and intended for educational and learning purposes.

---

## Acknowledgements

* Scikit-learn Documentation
* Python Documentation
* Marvellous Infosystems for Machine Learning guidance

