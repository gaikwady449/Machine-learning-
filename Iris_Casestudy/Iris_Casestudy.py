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

#------------------------------------------------------
#   Function name : VisulizeModel
#   Description :   It is used to Visulise the Data 
#   Parameters :    Filename 
#   Return :        Graph 
#   Date :          16/03/2026
#   Author :        yash Chandrakant Gaikwad
#--------------------------------------------------------
def VisulizeModel(model, cm):
    print(Border)
    print("step 7 : evaluate  the model performance ")
    print(Border)

    data=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    data.plot()

    plt.title("Confusion matrix of Iris Dataset")
    plt.show()
#--------------------------------------------------------
#   Function name : TestModel
#   Description :   It is used to Evaluate the Model
#   Parameters :    model, X_test,Y_test
#   Return :        None
#   Date :          14/03/2026
#   Author :        yash Chandrakant Gaikwad
#--------------------------------------------------------
def TestModel(X_test,Y_test,model):
    print(Border)
    print("step 5 : evaluate  the model")
    print(Border)

    Y_pred=model.predict(X_test)

    print("model Evaluation(testing) Completed")

    print(Y_pred.shape)

    print("Expected answers:")
    print(Y_test)
    print(Y_pred)


    print(Border)
    print("step 6 : evaluate  the model performance ")
    print(Border)

    accuracy=accuracy_score(Y_test,Y_pred)

    print("Accuracy of model is:",accuracy*100)

    cm=confusion_matrix(Y_test,Y_pred)
    print("Confusion matrix :")
    print(cm)

    print("Classivifaction Report")
    print(classification_report(Y_test,Y_pred))

    VisulizeModel(model,cm)
#--------------------------------------------------------
#   Function name : TrainModel
#   Description :   It does split X, Y, train  data ,testing data
#   Parameters :    df
#   Return :        None
#   Date :          16/03/2026
#   Author :        yash Chandrakant Gaikwad
#--------------------------------------------------------
def TrainModel(df):
    print(Border)
    print("step 4 : Train The Model")
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

# Test size = 20% 
# Train size = 80%

    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)

    print("X_shape:", X.shape)
    print("Y_shape :",Y.shape)
    print("Data spliting activity done :")
    print("X_train:",X_train.shape)
    print("X_test:",X_test.shape)
    print("Y_train:",Y_train.shape)
    print("Y_test:",Y_test.shape)

    print("We are going to use DicisiontreeClassifire ")

    model=DecisionTreeClassifier(
         criterion="gini",
         max_depth=3,
         random_state=42
        )

    print("Model Succesfully created :",model)
    model.fit(X_train,Y_train)

    print("Model training completed ")

    TestModel(X_test,Y_test,model)

#------------------------------------------------------
#   Function name : DataVisulization
#   Description :   It is used to Visulise the Data 
#   Parameters :    Filename 
#   Return :        Graph 
#   Date :          16/03/2026
#   Author :        yash Chandrakant Gaikwad
#--------------------------------------------------------

def DataVisulization(df):
    print(Border)
    print("step 3 : Data Visulaisation")
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
#--------------------------------------------------------
#   Function name : DataAnalysis
#   Description :   It shows Basic Information About Dataset  and
#                   Analysis Of Data
#   Parameters :    Data path of dataset file
#   Return :        None
#   Date :          16/03/2026
#   Author :        Yash Chandrakant Gaikwad
#--------------------------------------------------------
def DataAnalysis(df):
    print(Border)
    print("step 1 : Data Analysis")
    print(Border)


    print("Shape of dataset:", df.shape)

    print("Column names:",list(df.columns))

    print("Missing value (per column)")
    print(df.isnull().sum())

    print("Class Distribusion (Species count)")
    print(df["species"].value_counts())

    print("Statical report of dataset")
    print(df.describe())

#--------------------------------------------------------
#   Function name : irisLogistic
#   Description :   This is main pipeline controller
#                   It loads the dataset, shows raw data
#                   It preprocess the dataset & train the model
#   Parameters :    Data path of dataset file
#   Return :        None
#   Date :          16/03/2026
#   Author :        Yash Chandrakant Gaikwad
#--------------------------------------------------------

def  Irislogicstic(Datapath):
    df=pd.read_csv(Datapath)

    DataAnalysis(df)
    DataVisulization(df)
    TrainModel(df)
    

#--------------------------------------------------------
#   Function name : main
#   Description :   Starting point of the application
#   Parameters :    None
#   Return :        None
#   Date :          16/03/2026
#   Author :       Yash Chandrakant Gaikwad 
#--------------------------------------------------------
def main():
    Irislogicstic("iris.csv.csv")
    


if __name__ =="__main__":
    main()