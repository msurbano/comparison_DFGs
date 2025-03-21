import streamlit as st
import numpy as np
import pandas as pd
import pm4py

SAMPLE_DATA = "https://raw.githubusercontent.com/msurbano/VISCoPro/refs/heads/main/Event%20log%20example.csv"



def cargar_datos(mensaje_container, sample_data):

    with st.sidebar.expander("Upload your event log"):
        uploaded_file = st.file_uploader('Select a file (XES, CSV, or XLSX extension).')
        if st.button('Load sample data', type="primary"):
            sample_data = pd.read_csv(SAMPLE_DATA, sep=",",
            on_bad_lines='skip')



    if len(sample_data)>0:
        df = sample_data

        if df is not None:
            if pd.api.types.is_datetime64_any_dtype(df['time:timestamp']):
                st.write("1) La columna timestamp es de tipo datetime")
            else:
                # st.write('no es fecha')
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
                # st.write('1) timestamp convertido a tipo datetime')


            if not pd.api.types.is_string_dtype(df['case:concept:name']):
                df['case:concept:name'] = df['case:concept:name'].astype(str)

            st.session_state.original = df

            return df
            

    elif uploaded_file:
        mensaje_container.write("Loading...")
        if uploaded_file.name.endswith('.xes'):
            try:
                # Crear un archivo temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as temp_file:
                    temp_file.write(uploaded_file.read())  # Escribir el contenido del archivo subido
                    temp_file_path = temp_file.name  # Obtener la ruta del archivo temporal
                
                # Leer el archivo XES desde la ruta temporal
                log = pm4py.read_xes(temp_file_path)
                df = pm4py.convert_to_dataframe(log)
            except Exception as e:
                st.error(f"Error processing XES file: {str(e)}")
        elif uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                # Leer archivo Excel con pandas
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error processing Excel file: {str(e)}")
        else:
            st.error("Unsupported file format. Please upload a CSV or XES file")



        if df is not None:
            if pd.api.types.is_datetime64_any_dtype(df['time:timestamp']):
                st.write("1) La columna timestamp es de tipo datetime")
            else:
                # st.write('no es fecha')
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
                # st.write('1) timestamp convertido a tipo datetime')


            if not pd.api.types.is_string_dtype(df['case:concept:name']):
                df['case:concept:name'] = df['case:concept:name'].astype(str)
        
            st.session_state.original = df 

            

            return df

