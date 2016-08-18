import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
# from sklearn.svm import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

'''def SVM_Model(Scaled_Input_Data, Output_Data):
    Grid_Dict = {"C": [1e-1,1e0, 1e1],"gamma": np.logspace(-2, 1, 3)}
    cv_Strat = cross_validation.StratifiedKFold(Output_Data, n_folds=3)
    svm_Tuned = GridSearchCV(SVC(kernel='rbf', gamma=0.1, tol = 0.05), cv=cv_Strat, param_grid=Grid_Dict)
    svm_Tuned.fit(Scaled_Input_Data, Output_Data)
    return(svm_Tuned)

def SVM_Predictor(svm_Tuned, Input_test_Data):
    Predicted_SVM = svm_Tuned.predict(Input_test_Data)
    return(Predicted_SVM)
'''
def RF_Model(Input_Data, Output_Data):
    RFModel = RandomForestClassifier()
    RFModel.fit(Input_Data, Output_Data)
    return(RFModel)

def RF_Predictor(RFModel, Input_test_Data, Address_Results_RF):
    Input_test_IDs = Input_test_Data.iloc[:,0]
    Input_test_Data_NoID = Input_test_Data.iloc[:,1:]
    Results_prb = RFModel.predict_proba(Input_test_Data_NoID)
    Results_prbOne = Results_prb[:,1]

    # Submission
    output = pd.DataFrame({'activity_id':Input_test_IDs, 'outcome': Results_prbOne})
    #with open(Address_Results_RF, 'a') as f:
	#output.to_csv(f, header=False, index = False)

    output.to_csv(Address_Results_RF, index = False)
