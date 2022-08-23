# -*- coding: utf-8 -*-
"""
@author: andre orth (iai)

July 2022

OBlib streamilit app
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
# from tensorflow.keras.models import load_model
from pathlib import Path
import importlib

def main():
    st.title('OBlib - Occupant Behavior library')

    st.sidebar.text_input('Input Password:', key='pwd', type='password')
    st.sidebar.button('Logout', on_click=logout)

    if st.session_state['pwd'] == 'test':

        models_path = Path('models')
        data_types = [f.name for f in models_path.iterdir() if f.is_dir()]
        data_types.append('<Select>')
        data_types.sort()

        st_data_type = st.selectbox('Choose type of data', pd.DataFrame(data_types))
        if st_data_type == "<Select>":
            st.write("Please select a type of data from the dropdown menu")

        else:
            data_type_path = models_path / st_data_type
            model_types = [f.name for f in data_type_path.iterdir() if f.is_dir()]
            model_types.append('<Select>')
            model_types.sort()

            chosen_model = st.selectbox('Choose corresponding model', model_types)

            if chosen_model == "<Select>":
                st.write("Please select a model from the dropdown menu")
            else:
                st.header(f"{st_data_type} with {chosen_model}")
                model_type_path = data_type_path / chosen_model
                imported_module = importlib.import_module('.'.join(model_type_path.with_suffix('').parts) + '.model')
                model_class = imported_module.Model()

                dataset_name, dataset_path = model_class.dataset()

                col1, col2 = st.columns([1, 1])
                with col1:
                    b1 = st.button('Upload', key='1')
                with col2:
                    b2 = st.button(dataset_name, key='2')
                uploaded_file = None

                if b1:
                    uploaded_file = st.file_uploader("Upload building measurement data", type=['csv', 'txt'])
                    # ToDo: preprop/check for correct formatting
                    df = []
                    if uploaded_file is None:
                        st.write("Please upload a file or choose a model for calculation with ASHRAE database.")
                elif b2:
                    df = pd.read_csv(dataset_path)
                    model = model_class.load_trained(model_type_path)
                    y_pred, y_test, test_time = model_class.test(df, model)

                    st.header('Results')
                    scan_plot = st.empty()

                    from evaluation import AbsoluteMetrices
                    metrices = AbsoluteMetrices(y_test, y_pred)
                    eval_fetch = getattr(metrices, st_data_type)
                    eval = eval_fetch()

                    if 'Confusion Matrix' in eval.index:
                        conf = eval['Evaluation']['Confusion Matrix']
                        eval_ = eval.drop(index='Confusion Matrix')
                        st.dataframe(eval_.style.format("{:.2%}"))
                        st.write("Confusion Matrix")
                        st.dataframe(conf)
                    else:
                        st.dataframe(eval.style.format("{:.2%}"))

                    plot_df = pd.DataFrame(
                        {'Date_Time': test_time.Date_Time, 'Test': y_test.to_numpy().reshape(1, -1)[0],
                         'Prediction': y_pred})
                    fig = px.line(pd.melt(plot_df, id_vars=['Date_Time'], value_vars=plot_df.columns[1:]),
                                  x='Date_Time', y='value', color='variable')

                    scan_plot.plotly_chart(fig)

                    csv_metrics = convert_df(eval)
                    st.download_button(
                        "Press to download evaluation",
                        csv_metrics,
                        f"evaluation-{st_data_type}-{chosen_model}.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    csv_pred = convert_df(plot_df)

                    st.download_button(
                        "Press to download predictions",
                        csv_pred,
                        f"predictions-{st_data_type}-{chosen_model}.csv",
                        "text/csv",
                        key='download-csv'
                    )


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
