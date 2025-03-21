import streamlit as st

def dfg_options(dataframe):
    atributos = dataframe.columns.tolist()
    atributos2 = atributos.copy()

    if 'concept:name' in atributos2:
        atributos2.remove('concept:name')
        atributos2.insert(0, 'concept:name')


    columnas_a_eliminar = ['case:concept:name', 'time:timestamp']

    for col in columnas_a_eliminar:
        atributos2.remove(col)
        atributos.remove(col)

            
    col1, col2, col3, col4 = st.columns(4)

            
    nodes = col1.selectbox(
                'Nodes',
                (atributos2))

    lista_metricas = ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 
                    'Total Repetitions', 'Mean Cycle Time',  'Median Cycle Time', 'StDev Cycle Time', 'Total Cycle Time']
    metric = col2.selectbox(
                    'Metric',
                    (lista_metricas))

            # metric2 = col2.selectbox(
            #         'Additional metric',
            #         (lista_metricas), )

    perc_act = col3.slider('Activity threshold', min_value=0, max_value=100, value=100)

    perc_path = col4.slider('Path threshold', min_value=0, max_value=100, value=100)

    return nodes, metric, perc_act, perc_path
     