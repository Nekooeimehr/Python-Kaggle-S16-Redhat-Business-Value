import numpy as np
import pandas as pd
import datetime

from datetime import datetime

class Ctg2Num (object):
    def __init__(self, Input_data):   # Initializing the class
        self.Input_data = Input_data
    def Low_Freq_Merger(self, ColNames, prcnt1 = 25):  # Merges the values with low frequencies and replaces them with a new value
        for cols in ColNames:
            value_counts_Col = self.Input_data[cols].value_counts() 
            Rplc_Indx = value_counts_Col[value_counts_Col <= np.percentile(value_counts_Col, prcnt1)].index
            self.Input_data[cols].replace(Rplc_Indx, 'Type 0', inplace = True)
        return (self.Input_data)
    
    def Dummy_Generator(self, ColNames, dummy_na = False): # Creates dummy variables and removes the original categorical variables
        if dummy_na == False:
            Dummies = pd.get_dummies(self.Input_data[ColNames], drop_first = True)
        else:
            Dummies = pd.get_dummies(self.Input_data[ColNames], drop_first = True, dummy_na = True)
        self.Input_data = self.Input_data.join(Dummies)
        self.Input_data = self.Input_data.drop(ColNames, axis=1) 
        return (self.Input_data)

    def Freq_Replacor(self, ColNames): # Replaces variables with many values with their frequencies and removes the original variables
        for cols in ColNames:
            Freq_Ctg = self.Input_data[cols].value_counts().reset_index(name='count').rename(columns={'index': cols})
            self.Input_data = pd.merge(self.Input_data, Freq_Ctg, on=[cols], how='left')
            self.Input_data = self.Input_data.drop(cols, axis=1)
        return (self.Input_data)
    
def People_PreProcessor(people_data):

    # Handeling the missing values by replacing them with median if continous and by mode if categorical
    people_data['char_38'].fillna(people_data.median()['char_38'], inplace = True)
    assert (people_data['char_38'].shape[0]-people_data['char_38'].dropna().shape[0]) == 0,'The column Char-38 still has missing values'

    people_data.iloc[:,1:-1] = people_data.iloc[:,1:-1].apply(lambda x:x.fillna(x.value_counts().index[0]))
    assert (people_data.iloc[:,1:-1].shape[0]-people_data.iloc[:,1:-1].dropna().shape[0]) == 0,'The dataset still has missing values'
    
    # Converting TRUE FALSE Variables to ints
    people_data.iloc[:,12:] = people_data.iloc[:,12:].astype(int)

    # Deleting the unuseful features like the IDs 
    # people_data = people_data.drop(['group_1','date'], axis=1)

    # Handeling categorical features by converting them to binary dummy variables
    # For Char_3, char_4, the values with low frequencies are merged
    Ctg_People_Obj = Ctg2Num(people_data)
    Ctg_People_Obj.Low_Freq_Merger(['char_3','char_4'])
    
    # Generating Dummy variables and merging to the existing dataset
    Ctg_Ind_dum = ['char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9']
    Ctg_People_Obj.Dummy_Generator(Ctg_Ind_dum)
    
    # Replacing variable group_1 by the frequency of the groups
    people_data_Allnum = Ctg_People_Obj.Freq_Replacor(['group_1'])

    # Transforming some of the variables
        # Variable date were categorized to 4 quearters of the year and then binarized. The year of the variabe date was also binazrized.  
    people_data_Allnum['date'] =  pd.to_datetime(people_data_Allnum['date'])
    Date_Quarter = people_data_Allnum['date'].dt.quarter
    Date_Quarter_Dummies = pd.get_dummies(Date_Quarter,drop_first = True)
    people_data_Allnum = people_data_Allnum.join(Date_Quarter_Dummies)
    Date_year_Dummies = pd.get_dummies(people_data_Allnum['date'].dt.year,drop_first = True)
    people_Processed = people_data_Allnum.join(Date_year_Dummies)
    people_Processed.drop('date', axis=1, inplace = True)

    return(people_Processed)


def Activity_PreProcessor(Act_data, N):

    # Handeling the missing values by replacing them with median if continous and by mode if categorical
    # For char_10, first group by the people_id and then impute within each group. This way activities are imputed depeding on who has done them.   
    Act_data.groupby(by = 'people_id')['char_10'].transform(lambda x: x.fillna(x.value_counts().index[0]) if (x.dropna().shape[0] != 0) else x)
    # The variabes 'date' and 'activity_category' are imputed using their mode.
    ctg_Indx = ['date','activity_category','char_10']
    Act_data[ctg_Indx] = Act_data[ctg_Indx].apply(lambda x:x.fillna(x.value_counts().index[0]))
    assert (Act_data[ctg_Indx].shape[0] - Act_data[ctg_Indx].dropna().shape[0]) == 0,'The dataset still has missing values'

    # Handeling categorical features by converting them to binary dummy variables
    # For Char_1, char_2, char_8, chr_9 the values with low frequencies (In the first quartile) are merged
    Ctg_Act_Obj = Ctg2Num(Act_data)
    Ctg_Act_Obj.Low_Freq_Merger(['char_1','char_2','char_8','char_9'])
        
    # Generating Dummy variables and merging to the existing dataset
    Ctg_Ind_dum = ['char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9']
    Ctg_Act_Obj.Dummy_Generator(Ctg_Ind_dum, dummy_na = True)
    Ctg_Act_Obj.Dummy_Generator(['activity_category'])

    # Replacing variable char_10 by the frequency of the values
    Act_data_Allnum = Ctg_Act_Obj.Freq_Replacor(['char_10'])

    # Transforming some of the variables
        # Variable date were categorized to 4 quearters of the year and then binarized. The year of the variabe date was also binazrized.  
    Act_data_Allnum['date'] =  pd.to_datetime(Act_data_Allnum['date'])
    Date_Quarter = Act_data_Allnum['date'].dt.quarter
    Date_Quarter_Dummies = pd.get_dummies(Date_Quarter,drop_first = True)
    Act_data_Allnum = Act_data_Allnum.join(Date_Quarter_Dummies)
    Date_year_Dummies = pd.get_dummies(Act_data_Allnum['date'].dt.year,drop_first = True)
    Act_Processed = Act_data_Allnum.join(Date_year_Dummies)
    Act_Processed.drop('date', axis=1, inplace = True)
    Act_Train_Processed = Act_Processed.iloc[:N,:]
    Act_Test_Processed = Act_Processed.iloc[N:,:]
    return(Act_Train_Processed, Act_Test_Processed)

