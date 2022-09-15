# -*- coding: utf-8 -*-
"""
@author: yapan liu (su)
"""

import pandas as pd

# pull the information of the dataset which is used to train the model
def datasetInfo(dataset_id, ob_type):
    # id: the id of the dataset from ASHRAE OB database
    # df = pd.read_excel('./Models/all_studies_info_simplified.xlsx') 
    # df.to_parquet('./Models/all_studies_info_simplified.parquet', engine='pyarrow')
    
    # read from the parquet file
    df = pd.read_parquet('./Models/all_studies_info_simplified.parquet', engine='pyarrow')
    
    # select the data based on the dataset id and the behavior type
    df_select = df[(df['Study ID'] == dataset_id) & (df['Behavior Type'] == ob_type)].copy()
    df_select.set_index('Study ID', inplace=True)
    
    return df_select 

# crop the white region of the image in linux: "convert -trim input.jpg output.jpg"

# pull the model contributor information
def modelContributor(behavior_type, model_name):
    df = pd.read_csv("./Models/Model_Info.csv")
    df.replace(r'\n', '; ', regex=True, inplace=True)
    contributor_columns = ["Contributor", "Affiliation","Dataset", "Related_Publication"]
    model_info_columns = ["Behavior_Type","Model_Name","Model_Type","Model_Details","Model_Input","Model_Output","Model_Training","Model_Testing"]
    
    # model contributor information
    df_contributor = df.loc[(df['Behavior_Type'] == behavior_type) & (df['Model_Name'] == model_name), contributor_columns].copy()
    df_contributor = df_contributor.T
    df_contributor.columns = ['Information']
    
    # model information
    df_model = df.loc[(df['Behavior_Type'] == behavior_type) & (df['Model_Name'] == model_name), model_info_columns].copy()
    df_model = df_model.T
    df_model.columns = ['Information']
    
    # replace _ with space in the columns and index names
    df_model.replace('_', ' ', regex=True, inplace=True)
    df_model.index = df_model.index.str.replace('_', ' ', regex=True)
    
    df_contributor.replace('_', ' ', regex=True, inplace=True)
    df_contributor.index = df_contributor.index.str.replace('_', ' ', regex=True)
    
    return df_model, df_contributor
    
    

if __name__ == "__main__":
    
    # dataset_id = 26
    # ob_type = "Window_Status"
    # datasetInfo(dataset_id, ob_type)
    
    # data_path = './Models/Window_Status/SVM_E3D/'
    df_model, df_contributor = modelContributor(behavior_type='Window_Status', model_name='SVM_E3D')
    print(df_model)
    print(df_contributor)
    print("")

