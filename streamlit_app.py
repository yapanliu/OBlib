# -*- coding: utf-8 -*-
"""
@author:    andre orth (iai)
            yapan liu (syracuse university)

July 2022

OBlib streamilit app
"""

# runs under conda oblib environment

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
# from tensorflow.keras.models import load_model
from pathlib import Path
import importlib
from evaluation import AbsoluteMetrices
from utils import datasetInfo, modelContributor

# add sidebar
def sideBar():
    logo_path = './assets/images/IEA-EBC-Logo-removebg-preview.png'

    # Add sidebar to the app
    st.sidebar.markdown("## OBLib - Occupant Behavior Library")

    st.sidebar.markdown("---")  # Horizontal line
    st.sidebar.markdown("### About")
    st.sidebar.markdown("The 'OBLib-Occupant Behavior library' is a product of dedicated research and collaboration by the [BEST Lab](http://bestlab.syr.edu/) at Syracuse University, generously supported by IEA EBC Annex 79. To advance occupant behavior modeling, we have teamed up with international researchers to benchmark data-driven models on building occupant behaviors. Our preliminary efforts entailed an in-depth analysis of available datasets, pinpointing those apt for diverse modeling objectives. As an open-source offering, this library showcases a spectrum of machine learning models tailored to specific datasets from the [ASHRAE Global Occupant Behavior Database](https://ashraeobdatabase.com). Furthermore, we've benchmarked each model to compare its performance for building control applications.")
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Spacer
    st.sidebar.markdown("[GitHub Repository](https://github.com/yapanliu/OBlib) | [Contact Us](mailto:yliu88@syr.edu)")
    
    st.sidebar.markdown("---")  # Horizontal line

    # st.sidebar.markdown("---")  # Horizontal line


    # Add a logo
    st.sidebar.image(logo_path, width=200)


# add select box for the behavior type
def selectBehaviorType(models_path):
    
    ob_types = [f.name for f in models_path.iterdir() if f.is_dir()]  # get the folder names under the models path
    # add select option to the available folder names (behavior types)
    ob_types.append('<Select>')
    ob_types.sort()

    # st.markdown("## Select the type of behavior")
    st_ob_type = st.selectbox('Select the type of occupant behavior', ob_types, index=0)

    # select the model type
    if st_ob_type == "<Select>":
        st.markdown("**Please select a type of occupant behavior from the dropdown menu.*")

    else:
        behavior_type_path = models_path / st_ob_type # get the path to the selected behavior type
        selectModelType(behavior_type_path, st_ob_type)


# add select box for the model type
def selectModelType(behavior_type_path, st_ob_type):

    model_types = [f.name for f in behavior_type_path.iterdir() if f.is_dir()]
    
    if len(model_types) == 0:
        st.markdown("**Currently, no model available for this behavior type.*")
        st.markdown("**Updates coming soon!*")
    else:
        model_types.append('<Select>')
        model_types.sort()

        chosen_model = st.selectbox(f'Select the model of - {st_ob_type}', model_types, index=0)
        
        # select the model to be processed
        if chosen_model == "<Select>":
            st.markdown("**Please select a model from the dropdown menu.*")  
        else:
            # present results
            st.write(f"*Selected {st_ob_type.replace('_', ' ')} with {chosen_model.replace('_', ' ')}") 
            model_type_path = behavior_type_path / chosen_model
            
            # testModel(model_type_path, st_ob_type, chosen_model) # pre test and save results
            
            loadModelResults(model_type_path, st_ob_type, chosen_model) # load test results, reduce running time and server load

def loadModelResults(model_type_path, st_ob_type, chosen_model):
    '''
    model_type_path: behavior name + selected model
    st_ob_type: selected behavior name
    chosen_model: selected model name
    '''
    # get the model contributor information and model details
    df_model, df_contributor = modelContributor(behavior_type=st_ob_type, model_name=chosen_model)
    
    
    dataset_name = df_contributor.loc['Dataset', 'Information'] # get the dataset name
    
    st.header('Model Testing Results')
    st.markdown("#### Dataset Information")
    # get the dataset information
    df_dataset= datasetInfo(int(dataset_name.split(' ')[-1]), st_ob_type)
    st.dataframe(df_dataset)
    
    # load the model information
    st.markdown("#### Model Information")
    st.table(df_model)
    
    
    # load test results
    st.markdown("#### Testing Results")
    results_dir = f"./{model_type_path}/test_results/"
    image = Image.open(results_dir + 'fig.png')

    st.image(image, caption='Model Resting Results')
    
    # '''evaluation metrics''' 
    with open(f'{results_dir}/evaluation.pickle', 'rb') as f:
        eval = pickle.load(f)
        
    with open(f'{results_dir}/predictions.pickle', 'rb') as f:
        predictions = pickle.load(f)
    
    # display results
    col1, col2 = st.columns([1, 1])
    
    if 'Confusion Matrix' in eval.index:
        with col1:
            conf = eval['Evaluation']['Confusion Matrix']
            eval_ = eval.drop(index='Confusion Matrix')
            st.markdown("#### Evaluation Results")
            st.table(eval_.style.format("{:.2%}"))
        with col2:
            st.markdown("#### Confusion Matrix")
            st.table(conf)
            # st.markdown("*0-Close, 1-Open*")
    else:
        with col1:
            st.markdown("#### Evaluation Results")
            eval = eval.astype(float).round(2)
            st.dataframe(eval.style.format("{:.2%}"))
            
    
    st.markdown("#### Download Results")
    csv_metrics = convert_df(eval)
    st.download_button(
        "Click to download evaluation",
        csv_metrics,
        f"evaluation-{st_ob_type}-{chosen_model}.csv",
        "text/csv",
        key='download-evaluation-csv'
    )
    predictions.reset_index(drop=True, inplace=True)
    csv_pred = convert_df(predictions)

    st.download_button(
        "Click to download predictions",
        csv_pred,
        f"predictions-{st_ob_type}-{chosen_model}.csv",
        "text/csv",
        key='download-predictions-csv'
    )
    
    # get the model contributor information
    st.markdown("#### Model Information")
    st.table(df_contributor)

    st.markdown("""---""")
# select the model to be processed 
def testModel(model_type_path, st_ob_type, chosen_model):
    '''
    model_type_path: behavior name + selected model
    st_ob_type: selected behavior name
    chosen_model: selected model name
    '''
        
    imported_module = importlib.import_module('.'.join(model_type_path.with_suffix('').parts) + '.model')
    
    model_class = imported_module.Model()

    dataset_name, dataset_path = model_class.dataset()

    df = pd.read_csv(dataset_path)
    model = model_class.load_trained(model_type_path)
    y_pred, y_test, test_time = model_class.test(df, model)

    st.header(f'Model Testing Results for {dataset_name}')
    st.markdown("**Display dataset information here (pull from master table)*")
    scan_plot = st.empty()

    '''evaluation metrics'''
    metrices = AbsoluteMetrices(y_test, y_pred)
    eval_fetch = getattr(metrices, st_ob_type)
    eval = eval_fetch()
    
    col1, col2 = st.columns([1, 1])

    if 'Confusion Matrix' in eval.index:
        with col1:
            conf = eval['Evaluation']['Confusion Matrix']
            eval_ = eval.drop(index='Confusion Matrix')
            st.markdown("#### Evaluation Results")
            st.dataframe(eval_.style.format("{:.2%}"))
        with col2:
            st.markdown("#### Confusion Matrix")
            st.dataframe(conf)
            # st.markdown("*0-Close, 1-Open*")
    else:
        with col1:
            st.markdown("#### Evaluation Results")
            st.dataframe(eval.style.format("{:.2%}"))

    plot_df = pd.DataFrame(
        {'Date_Time': test_time.Date_Time, 'Test': y_test.to_numpy().reshape(1, -1)[0],
            'Prediction': y_pred})
    fig = px.line(pd.melt(plot_df, id_vars=['Date_Time'], value_vars=plot_df.columns[1:]),
                    x='Date_Time', y='value', color='variable')

    scan_plot.plotly_chart(fig)

    csv_metrics = convert_df(eval)
    st.download_button(
        "Click to download evaluation",
        csv_metrics,
        f"evaluation-{st_ob_type}-{chosen_model}.csv",
        "text/csv",
        key='download-evaluation-csv'
    )
    csv_pred = convert_df(plot_df)

    st.download_button(
        "Click to download predictions",
        csv_pred,
        f"predictions-{st_ob_type}-{chosen_model}.csv",
        "text/csv",
        key='download-predictions-csv'
    )
    
    # save test results
    results_dir = f"./{model_type_path}/test_results/"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fig.write_image(results_dir + "fig.png")
    
    # print(plot_df)
    # print(eval)
    
    with open(f'{results_dir}/evaluation.pickle', 'wb') as f:
        pickle.dump(eval, f)
        
    with open(f'{results_dir}/predictions.pickle', 'wb') as f:
        pickle.dump(plot_df, f)
    
    # plot_df.to_csv(results_dir + "predictions.csv", index=False)
    # eval.to_csv(results_dir + "evaluation.csv", index=True)
    

def main():
    
    # render the sidebar
    sideBar()
    
    # set title of the page
    st.title('OBLib-Occupant Behavior library')
    st.markdown('<font style="color:blue"> *Currently only the  "Occupant Number" and "Window Status" have valid models.<br>More models coming soon!* </font>', unsafe_allow_html=True)
    
    models_path = Path('Models')  # path to the models folder
    
    selectBehaviorType(models_path)


@st.cache
def upload_data(upload):
    if upload.name.endswith('txt'):
        raw_data = np.load()        # todo: finish both imports + error handling
    elif upload.name.endswith('csv'):
        raw_data = pd.read_csv()
    return raw_data


@st.cache
def load_basedata():
    # todo: base_data = (load ASHRAE with API or include in database // check if downloaded otherwise prompt)
    base_data = ()
    return base_data


@st.cache
def update_scan_plot(input_df, prediction, labels, basescans):
    # todo: check further caching options for better performance with larger test sets
    return


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


def logout():
    st.session_state["pwd"] = ""


if __name__ == '__main__':
    main()
