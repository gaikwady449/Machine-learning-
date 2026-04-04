import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
Border="-"*40
#####################################################################################################
# Step 1 : load the Dataset
#####################################################################################################
print(Border)
print("step 1 : load the data")
print(Border)


Datasetpath=r"C:\Users\gaikw\OneDrive\Desktop\python\Mashine_learning\iris.csv.csv"

df=pd.read_csv(Datasetpath)

("Step 1 : load the dataset")
print("Dataset gets load succsesfully")
print("Initial entry from dataset:")
print(df.head())

Border="-"*40
#####################################################################################################
# step 2 : Data Analysis(EDA)
#####################################################################################################
print(Border)
print("step 2 : Data Analysis")
print(Border)


print("Shape of dataset:", df.shape)

print("Column names:",list(df.columns))

print("Missing value (per column)")
print(df.isnull().sum())

print("Class Distribusion (Species count)")
print(df["species"].value_counts())

print("Statical report of dataset")
print(df.describe())

Border="-"*40
#####################################################################################################
# step 3 : Decide Independant and Dependant Variable 
#####################################################################################################
print(Border)
print("step 3 : Dicide indipendant And Dependant Variables ")
print(Border)

# X : indipendant Variable / Features
# y : dependant Variable / lable
Feature_Cols=[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X= df[Feature_Cols]
Y= df["species"]

print("X shape:",X.shape)
print("Y shape:",Y.shape)


Border="-"*40
#####################################################################################################
# step 4 : Data Visulaisation 
#####################################################################################################
print(Border)
print("step 4 : Data Visulaisation")
print(Border)

# Scatter plot

plt.figure(figsize=(7,5))
for sp in df["species"].unique():
    temp=df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"],[temp["petal width (cm)"]],label=sp)

plt.title("Iris : petal length vs petal width:")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend()
plt.grid = True
plt.show()

Border="-"*40
#####################################################################################################
# step 5 : Split the Dataset For training and testing 
#####################################################################################################
print(Border)
print("step 5 : split the Dataset For training and testing ")
print(Border)

# Test size = 20% 
# Train size = 80%

X_train,X_test,Y_train,Y_test= train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
print("X_shape:", X.shape)
print("Y_shape :",Y.shape)
print("Data spliting activity done :")
print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("Y_train:",Y_train.shape)
print("Y_test:",Y_test.shape)


Border="-"*40
#####################################################################################################
# step 6 : Build the model
#####################################################################################################
print(Border)
print("step 6 : Build the model")
print(Border)

print("We are going to use DicisiontreeClassifire ")

model=DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

print("Model Succesfully created :",model)


Border="-"*40
#####################################################################################################
# step 7 : Train the model
#####################################################################################################
print(Border)
print("step 6 : Train  the model")
print(Border)

model.fit(X_train,Y_train)

print("Model training completed ")

Border="-"*40
#####################################################################################################
# step 8 : evaluate the model
#####################################################################################################
print(Border)
print("step 8 : evaluate  the model")
print(Border)

Y_pred=model.predict(X_test)

print("model Evaluation(testing) Completed")

print(Y_pred.shape)

print("Expected answers:")
print(Y_test)
print(Y_pred)

Border="-"*40
#####################################################################################################
# step 9 : evaluate the model performance
#####################################################################################################
print(Border)
print("step 9 : evaluate  the model performance ")
print(Border)

accuracy=accuracy_score(Y_test,Y_pred)

print("Accuracy of model is:",accuracy*100)

cm=confusion_matrix(Y_test,Y_pred)
print("Confusion matrix :")
print(cm)

print("Classivifaction Report")
print(classification_report(Y_test,Y_pred))

#####################################################################################################
Border="-"*40
#####################################################################################################
# step 10 : evaluate the model performance
#####################################################################################################
print(Border)
print("step 9 : evaluate  the model performance ")
print(Border)

data=confusion_matrix(confusion_matrix=cm,display_label=model.classes_)
data.plot()

plt.title("Confusion matrix of Iris Dataset")
plt.show()







