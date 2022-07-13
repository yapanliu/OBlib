import enum
import json 
import logging
import zipfile
import pandas as pd
from obplatform import Connector, logger
import pickle 

def load_ashrae_data():
    # load picked data from the pickle file
    with open("./OBlib/tmp.txt", "rb") as fp:
        df_main_table = pickle.load(fp)
    # Type: 0-[in-situ]; 1-[survey]; 2-[mixed]
    df_main_table = df_main_table.loc[df_main_table['Study_Type']==0]
    df_main_table.reset_index(drop=True, inplace=True)    
    behaviors = df_main_table['Behavior_Type'].unique()

    print("##############################################################")
    print("Available Behavior Types!")
    for idx, name in enumerate(behaviors):
        print(idx, name)
    print("##############################################################")

    while True:
        try:
            behavior_num = int(input(f"Please enter the behavior number (0-{len(behaviors)-1}): "))
        except ValueError:
            print("This is an unaccepted response, enter a valid value")
            continue
        if behavior_num < 0 or behavior_num > len(behaviors)-1:
            print("This is out of range, enter a valid value")
            continue
        else:
            break
    print("##############################################################")
    behavior_select = behaviors[behavior_num]
    print(f"You selected: Number '{behavior_num}', Behavior '{behavior_select}'")
    
    df_selected = df_main_table[df_main_table['Behavior_Type'] == behavior_select]
    df_selected.reset_index(drop=True, inplace=True)
    print(f"Available Studies with selected behavior: {behavior_select}!")
    results = df_selected.to_json(orient='index')
    parsed = json.loads(results)
    print(json.dumps(parsed, indent=4))  
    
    # available studies based on the selected behavior
    study_ids = df_selected['Study_ID'].unique().tolist()

    # ask user to select a study to download 
    while True:
        try:
            id = int(input(f"Please enter the study ID to download: "))
        except ValueError:
            print("This is an unaccepted response, enter a valid value")
            continue
        if id not in study_ids:
            print("This is not in the list, enter a valid value")
            continue
        else:
            break
    # create a connector of OBPlatform
    connector = Connector()
    # Download Plug Load + Occupant Presence behaviors from study 22, 11, and 2.
    connector.download_export(
        "data.zip",
        [behavior_select],
        [id],
        show_progress_bar=True,  # False to disable progrees bar
    )
    # breakpoint()
    zf = zipfile.ZipFile("data.zip")
    df = pd.read_csv(zf.open(f"{behavior_select}_Study{id}.csv"))
    print("##############################################################")
    print("Head of the dataframe:")
    print(df.head())


if __name__ == "__main__":
    
    load_ashrae_data()