# Importing necessary packages
import numpy as np
import pandas as pd

from Input_PreProcessor import People_PreProcessor
from Input_PreProcessor import Activity_PreProcessor
# from Models import SVM_Model
# from Models import SVM_Predictor
from Models import RF_Model
from Models import RF_Predictor

from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split

# Reading the dataset 
Address_people = 'C:/Users/nekooeimehr/AppData/Local/Programs/Python/Python35-32/Kaggle RedHat/people.csv'
Address_ActTrain = 'C:/Users/nekooeimehr/AppData/Local/Programs/Python/Python35-32/Kaggle RedHat/act_train.csv'
Address_ActTest = 'C:/Users/nekooeimehr/AppData/Local/Programs/Python/Python35-32/Kaggle RedHat/act_test.csv'
Address_Results_RF = 'C:/Users/nekooeimehr/AppData/Local/Programs/Python/Python35-32/Kaggle RedHat/redhat_RF10.csv'

N = 50000
people_data = pd.read_csv(Address_people)
ActTrain_data = pd.read_csv(Address_ActTrain, nrows = N)
ActTrain_dataX = ActTrain_data.iloc[:,:-1]
ActTrain_dataY = ActTrain_data.iloc[:,-1]
ActTest_data = pd.read_csv(Address_ActTest, nrows = N, skiprows = range(1, 10*N)) # range(1, 2*N)
Act_whole_data = pd.concat([ActTrain_dataX, ActTest_data], ignore_index = True)

# Processing the dataset
people_Processed = People_PreProcessor(people_data)
(ActTrain_Processed, ActTest_Processed) = Activity_PreProcessor(Act_whole_data, N)

# Merging the 2 datasets
MergedTrain_Processed = ActTrain_Processed.merge(people_Processed, how='left', on='people_id')
MergedTest_Processed = ActTest_Processed.merge(people_Processed, how='left', on='people_id')

# MergedTrain_Processed = MergedTrain_Processed2.sample(n = 20000, replace=False)

# Seperating the input variables(Predictors) and the output variable and scaling the input variables
Input_train_Data = MergedTrain_Processed.iloc[:,2:]
# Scaled_train_Data = scale(Input_train_Data)
Output_Data = ActTrain_dataY
Input_test_Data = MergedTest_Processed.iloc[:,1:]
#Scaled_test_Data = scale(Input_test_Data)

'''
#######################################First Model: SVM######################################################
# Building the model
(MeanMSE_SVR, svr_Tuned) = First_Model_SVR(Scaled_train_Data, Output_Data)

# Predicting the test set using the built model
SVR_Results = SVR_Predictor(svr_Tuned, Scaled_test_Data, Address_test_SVR)
'''
#####################################Second Model: Random Forest######################################################
# Building the model
RF_MDL = RF_Model(Input_train_Data, Output_Data)

# Predicting the test set using the built model
KRR_Results = RF_Predictor(RF_MDL, Input_test_Data, Address_Results_RF)

