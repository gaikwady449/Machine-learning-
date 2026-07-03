import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


#----------------------------------------------------------------------------------
# step 1 : load the Dataset 
#----------------------------------------------------------------------------------

df=pd.read_csv("breast_cancer.csv")
print("shape of Dataset:",df.shape)
print("First 5 record :",df.head())


#----------------------------------------------------------------------------------
# step 2 : seprate featureas  and label
#---------------------------------------------------------------------------------- 

X=df.drop("target",axis=1)
Y=df["target"]

print("Shape of X")



#----------------------------------------------------------------------------------
# step 3 : split the Dataset into Training and Testing 
#----------------------------------------------------------------------------------

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)



#----------------------------------------------------------------------------------
# step 4 : Create Base model
#----------------------------------------------------------------------------------

base_model=DecisionTreeClassifier(random_state=42)


#----------------------------------------------------------------------------------
# step 5: create Bagging model
#----------------------------------------------------------------------------------

Bagging_model=BaggingClassifier(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)


#----------------------------------------------------------------------------------
# step 6 : Train Bagging model
#----------------------------------------------------------------------------------

Bagging_model.fit(X_train,Y_train)




#----------------------------------------------------------------------------------
# step 7 : Test Bagging model
#----------------------------------------------------------------------------------


Y_pred=Bagging_model.predict(X_test)



#----------------------------------------------------------------------------------
# step 7 : Evaluate Bagging model
#----------------------------------------------------------------------------------


print("Bagging Accuracy:",accuracy_score(Y_test,Y_pred)*100)

print("confusion matrix:")
print(confusion_matrix(Y_test,Y_pred))