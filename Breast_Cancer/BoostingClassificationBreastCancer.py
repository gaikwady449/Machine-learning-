import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
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
# step 4 : Create Boosting model  Adaboost 
#----------------------------------------------------------------------------------

Boost_model=AdaBoostClassifier(random_state=42,n_estimators=50,learning_rate=1.0)





#----------------------------------------------------------------------------------
# step 5 : Train Bagging model
#----------------------------------------------------------------------------------

Boost_model.fit(X_train,Y_train)




#----------------------------------------------------------------------------------
# step 7 : Test Boosting model
#----------------------------------------------------------------------------------


Y_pred=Boost_model.predict(X_test)



#----------------------------------------------------------------------------------
# step 7 : Evaluate Boosting model
#----------------------------------------------------------------------------------


print("Boostng Accuracy:",accuracy_score(Y_test,Y_pred)*100)

print("confusion matrix:")
print(confusion_matrix(Y_test,Y_pred))