# OBlib Streamlit Documentation
Install the required packages listed in the requirements.txt together with Python 3.8.
(For the current models download 'Study_26_Study26.csv' from the [shared Google Drive](https://drive.google.com/file/d/1pBptbqEkn5WkWeRh4gNowM7AB1QNfBbt/view?usp=sharing) and copy it into `/data/study_26/` with the already provided dictionary.)

To run streamlit locally set the path to the folder containing ‘oblib_app.py’ and execute “streamlit run oblib_app.py”.
Online deployment canbe access here (linked with staging branch): [https://yapanliu-oblib-streamlit-app-staging-4j4hur.streamlit.app/](https://yapanliu-oblib-streamlit-app-staging-4j4hur.streamlit.app/)

## Implementation of new models

In the models-directory a template is provided to implement and test your models. Create a new subfolder in the corresponding data-type folder and add the template (model.py) together with your pretrained model (for example via pickle, tensorflow, etc.) as shown below (new folders and files marked with *)

* models
    * datatype (e.g. Shading)
        * *name_of_your_model
            * *model.py
			* *model_file.pkl


The new model folder is automatically included in the dropdown menu of the app and doesn’t need to be added separately.

## Current state of preprocessing

The preprocessing of the already included models has currently been rolled back to the old state where training and test data is calculated in the same function:
`return x_train, y_train, x_test, y_test, test_time)`

Normally the data split should happen in the respective function train/test resulting in a type-independent preprocessing function that only does preprocessing:
`return x, y, time`

In case datatype-dependant test datasets are agreed on the loading and splitting can be outsourced from the test functions into a combined script.

## Requirements for new models

Which function and return needs to be kept and what can be freely changed:
* dataset(): name of dataset for display button and path to used dataset
* load_trained(): load trained model and return it
* test(): test supplied model with given test data and return prediction, true values and time scale for plotting

…

## ASHRAE datasets for testing / API

The AHSRAE API implemented by Yapan Liu is currently located in another branch of this project and will be merged with this branch. This will be needed to automatically download datasets locally and reduce file size if larger amounts of data are added later but not necessarily used.

