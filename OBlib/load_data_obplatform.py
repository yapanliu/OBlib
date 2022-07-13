import enum
import logging
import zipfile
import pandas as pd
from obplatform import Connector, logger
import pickle 

# save data to pickle file 
df_main_table = pd.read_excel('./OBlib/All_behavior_studies_copy_20210614.xlsx')
print(df_main_table.columns)

with open("./OBlib/tmp.txt", "wb") as fp:
    pickle.dump(df_main_table, fp)

print(df_main_table['Behavior_Type'].unique())

# load picked data from the pickle file
with open("./OBlib/tmp.txt", "rb") as fp:
    df_main_table = pickle.load(fp)

print("##############################################################")
print("Available Behavior Types!")
for idx, name in enumerate(df_main_table['Behavior_Type'].unique()):
    print(idx, name)
print("##############################################################")


connector = Connector()
# breakpoint()
# List all behaviors available in the database
# print(connector.list_behaviors())
[print(x) for x in connector.list_behaviors()]

# Print progress information
# Comment out the following line to hide progress information
logger.setLevel(logging.INFO)

# Download Plug Load + Occupant Presence behaviors from study 22, 11, and 2.
connector.download_export(
    "data.zip",
    ["Lighting_Status"],
    ["9"],
    show_progress_bar=True,  # False to disable progrees bar
)

# behavior_type = "Plug_Load"
behavior_type = "Lighting_Status"
study_id = "9"

zf = zipfile.ZipFile("data.zip")
df = pd.read_csv(zf.open(f"{behavior_type}_Study{study_id}.csv"))
print(df.head())

# List all behaviors available in study 1, 2, 3, and 4
json_study_behaviors = connector.list_behaviors_in_studies(studies=["9", "22"])
print(json_study_behaviors)

# List all studies available in the database, filtered by behavior types,
# countries, cities, {building type, room_type} combinations. [required]
json_studies = connector.list_studies(
    behaviors=["Occupancy_Measurement", "Plug_Load"],
    countries=["USA", "UK"],
    cities=["Palo Alto", "Coventry", "San Antonio"],
    buildings=[
        {
            "building_type": "Educational",
            "room_type": "Classroom",
        },
        {
            "building_type": "Educational",
            "room_type": "Office",
        },
        {
            "building_type": "Residential",
            "room_type": "Single-Family House",
        },
    ],
)
print(json_studies)





