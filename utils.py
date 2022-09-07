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
def modelContributor(data_path):
    df = pd.read_csv(f'{data_path }'+ '/model_info.txt', sep=':', header=None)
    
    df.set_index(0, drop=True, inplace=True)
    df.columns = ['Information']
    
    return df
    
    

if __name__ == "__main__":
    
    dataset_id = 26
    ob_type = "Window_Status"
    datasetInfo(id, ob_type)
    
    data_path = './Models/Window_Status/SVM_E3D/'
    modelContributor(data_path)

