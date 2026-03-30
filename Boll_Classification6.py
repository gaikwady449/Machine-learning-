from sklearn import tree

# Rough = 1
# Smooth = 0

# Cricket  = 0
# Tennis = 1


#--------------------------------------------------------
#   Function name : TrainModel
#   Description :   It does split X, Y, tarinning data ,testing data
#   Parameters :    X,Y
#   Return :        None
#   Date :          14/03/2026
#   Author :        yash Chandrakant Gaikwad
#--------------------------------------------------------
def Train_Model():

    print("split Dataset Into Independant And Dependant Variable")
    # independant Variable For Traning 
    Xtrain=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]]
    # independant Variable For Testing
    Xtest=[[35,1],[95,0]]
    # Dependant Variable For Traning 
    Ytrain=[1,1,2,1,2,1,2,1,1,1,2,1,2]
    # Dependant Variable For Testing 
    Ytest=[1,2]

    modelobj = tree.DecisionTreeClassifier()

    Trendmodel=modelobj.fit(Xtrain,Ytrain)

    Result=Trendmodel.predict([[35,1]])

    print(type(Result))
    if Result == 1:
        print("Object lokks like Tennis Ball")

    elif Result == 0:
        print("object looks like Ckeicket Ball")

    

    print("Model predict an object as :",Result)

#----------------------------------------------------------------------------------------------------------
# Funtion name : Load_Dataset
# Description : This is main pipeline controller
# parameter : Data path of dataset file
             # it load the dataset show the raw the Data
             # it preprocess the dataset train the model
# Return : None
# Date:14/03/26
# Author : yash chandrakant Gaikwad
#----------------------------------------------------------------------------------------------------------

def Load_Dataset():
    print("Ball classification case studies")
    # Indipendant variable
    # orignal encoded Dataset
    X=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    # Dependant Variable
    Y=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]


#----------------------------------------------------------------------------------------------------------
# Funtion name : main
# Description : starting point of the application
# parameter : None
# Return : None
# Date:14/03/26
# Author : yash chandrakant Gaikwad
#-----------------------------------------------------------------------------------------------------------
def main():

    Train_Model()
    Load_Dataset()
if __name__ == "__main__":
    main()