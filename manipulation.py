import streamlit as st
import numpy as np
import pandas as pd
import pm4py
import copy
import deprecation
import os
import recommendations
# import cairosvg
from PIL import Image
# import svgwrite
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py.algo.filtering.dfg import dfg_filtering
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.statistics.rework.cases.log import get as cases_rework_get
from pm4py.statistics.start_activities.log.get import get_start_activities
from pm4py.statistics.end_activities.log.get import get_end_activities
import networkx as nx
from pm4py.statistics.rework.cases.log import get as rework_cases
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.end_activities import end_activities_filter
from pm4py.statistics.rework.cases.log import get as rework_cases
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.filtering.log.attributes import attributes_filter
# from pm4py.objects.dfg import dfg_factory
import json
import re
# import datetime
from datetime import date, time, datetime
from pm4py.visualization.dfg.variants.frequency import apply
from pm4py.visualization.dfg.variants import performance
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from PIL import Image
from io import StringIO
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from streamlit import session_state as ss

ids = []

if 'ids' not in st.session_state:
    st.session_state['ids'] = []


def small_text(text):
    return f"<p style='font-size:11px; color:grey; font-style:italic;'>{text}</p>"





def manipulation_options(df, original, i): 

    with st.sidebar.expander(f"Data manipulation action {i+1}", expanded=True):
    
        ft_group = st.selectbox(
            'Filter type',
            ('Attribute', 'Performance', 'Follower', 'Timeframe', 'Rework', 'Endpoints'),
            key=f'ft_group_{i}'
            )   

        filters1 = ("Mandatory", "Forbidden", "Keep Selected")
        filters2 = ("Case performance", 'Path performance')
        filters3 = ("Directly Followed", "Eventually Followed") # , "Keep Selected Fragments")

        if ft_group == 'Attribute':
            filter_modes = filters1
        elif ft_group == 'Performance':
            filter_modes = filters2
        elif ft_group == 'Follower':
            filter_modes = filters3
        else:
            filter_modes = [ft_group]

        if ft_group == 'Attribute':
            ft = st.selectbox('Filter mode', filter_modes, key=f'ft_{i}')
            explanations = {
                'Mandatory': "This filter removes all cases that do not have at least one event with one of the selected values.",
                'Forbidden': "This filter removes all cases that have at least one event with one of the selected values.",
                'Keep Selected': "This filter removes all events that do not have one of the selected values."
            }
            st.markdown(small_text(explanations[ft]), unsafe_allow_html=True)

            at = st.selectbox(
                'Attribute', original.columns,
                key=f'at_{i}'
            )

            valores = ['* All values'] + list(original[at].unique())
            # st.write(valores)
            
            value = st.multiselect('Value', list(valores), key=f'value_{i}')

            g = st.checkbox('Group by', key=f'g_{i}')

            manip = [ft, (at, g), value]

        elif ft_group == 'Performance':

            ft = st.selectbox('Filter mode', filter_modes, key=f'ft_{i}')

            explanations = {
                'Path performance': "This filter selects event log traces that contain a specific transition between two user-selected activities that fall within a user-specified time interval.",
                'Case performance': "This filter selects event log traces whose duration falls within a user-specified time interval."
            }
            st.markdown(small_text(explanations[ft]), unsafe_allow_html=True)

            if ft == "Path performance":
                act1 = st.selectbox(
                    'From', original['concept:name'].unique(),
                    key=f'act1_{i}'
                )

                act2 = st.selectbox(
                    'To', original['concept:name'].unique(),
                    key=f'act2_{i}'
                )

                rango = st.slider("Minutes between them:", 1, 500, key=f'rang_{i}', value=st.session_state["rango"].get(f'rang_{i}', (25, 75)))
                manip = [ft, (act1, act2), rango]

            else: # 'Case performance':
                mode_options = ["Unique interval", "More than one interval"]
                mode = st.selectbox("Mode", mode_options, key=f'mode_{i}')

                if mode == "Unique interval":
                    rango = st.slider("Minutes between them:", 1, 500, key=f'rang_{i}', value= (25, 75))
                else:
                    n = st.number_input('Number of intervals', step=1, min_value=2, key=f'nrange_{i}')
                    rango = []
                    for j in range(n):
                        r1 = st.sidebar.slider(f"Minutes between the interval {j+1}:", 1, 500, key=f'rang2_{j}', value= (25, 75))
                        rango.append(r1)

                manip = [ft, mode, rango]

        elif ft_group == "Endpoints":
            st.markdown(small_text("This filter removes all cases in which the first and/or last events do not have one of the selected values."), unsafe_allow_html=True)

            options = ['* All values'] + list(pm4py.get_start_activities(df))
            act1 = st.multiselect('From', options, key=f'act1_{i}')

            options2 = ['* All values'] + list(pm4py.get_end_activities(df))
            act2 = st.multiselect('To', options2, key=f'act2_{i}')

            manip = [ft_group, act1, act2]

        elif ft_group == 'Rework':
            st.markdown(small_text("This filter selects event log traces where a concrete activity occurs more than once with a specified minimum frequency."), unsafe_allow_html=True)
            act1 = st.selectbox(
                'Activity', df['concept:name'].unique(),
                key=f'act1_{i}')

            value = st.number_input('Minimum frequency', step=1, key=f'value_{i}')

            manip = [ft_group, act1, value]

        

        elif ft_group == 'Timeframe':
            st.markdown(small_text("This filter selects event log traces that belong to a time period using the timestamp attribute."), unsafe_allow_html=True)
            
            r, l = st.columns(2)

            default_date1 = original['time:timestamp'][0].date()
            default_hour1 = datetime.now().time()

            default_date2 = original['time:timestamp'][0].date()
            default_hour2 = datetime.now().time()

            rango1 = r.date_input("From", key=f'date1_{i}', value=default_date1)
            h1 = l.time_input('Time From', key=f'hour1_{i}', value=default_hour1)
            start_datetime = datetime.combine(rango1, h1)

            rango2 = r.date_input("To", key=f'date2_{i}', value=default_date2)
            h2 = l.time_input('Time To', key=f'hour2_{i}', value=default_hour2)
            end_datetime = datetime.combine(rango2, h2)

            manip = [ft_group, (start_datetime, end_datetime)]
        
        else:
            ft = st.selectbox('Filter mode', filter_modes, key=f'ft_{i}')

            explanations = {
                'Directly Followed': "This filter selects event log traces where one activity is followed by another at some point, but not necessarily immediately.",
                'Eventually Followed': "This filter selects event log traces where one activity is eventually followed by another.",
                'Keep Selected Fragments': "This filter retains only a specific sub-process, defined as a part of the process between two specific activities."
            }
            st.markdown(small_text(explanations[ft]), unsafe_allow_html=True)

            nfollow = st.number_input('Number of fragments', step=1, min_value=1)
            
            k = 1
            lista_act = []

            act1 = st.selectbox(
                            'From', (original['concept:name'].unique()), 
                            key='act1_%s' % i)

            while (nfollow > 1 and k < nfollow):

                # st.write('prueba')
                actk = st.selectbox(
                        'To - From', (original['concept:name'].unique()), 
                        key='actk_%s' % k)

                lista_act.append(actk)
                k = k+1

            act2 = st.selectbox(
                        'To', (original['concept:name'].unique()), 
                        index=original['concept:name'].unique().tolist().index(st.session_state["act2"].get('act2_%s' % i, original['concept:name'].unique()[0])), 
                        key='act2_%s' % i)
            
            # st.write(lista_act)
            manip = [ft,(act1,act2),lista_act]

    return manip 










def manipulation_options2(df, original, i, id):

    
    # ids = st.session_state['ids']
    
    
    # Inicializar almacenamiento en sesión
    if 'manipulations' not in st.session_state:
        st.session_state['manipulations'] = {}

    # Recuperar o inicializar valores para el id actual
    if id not in st.session_state['manipulations']:
        st.session_state['manipulations'][id] = {}

    # Acceso al almacenamiento específico del id
    id_state = st.session_state['manipulations'][id]

    
    with st.sidebar.expander(f"Data manipulation action {i+1}", expanded=True):
        options = ['Attribute', 'Performance', 'Follower', 'Timeframe', 'Rework', 'Endpoints']
        # st.write(id_state.get('ft_group', 0))
        # st.write(id_state.get('ft_group', 0) in options)
        # st.write(f"Tipo de ft_group: {type(id_state.get('ft_group', 0))}")
        ft_group = st.selectbox(
            'Filter type',
            (options),
            key=f'ft_group_{i}_{id}', index=options.index(id_state.get('ft_group', 0))
            )   
        id_state['ft_group'] = ft_group

        filters1 = ("Mandatory", "Forbidden", "Keep Selected")
        filters2 = ("Case performance", 'Path performance')
        filters3 = ("Directly Followed", "Eventually Followed", "Keep Selected Fragments")

        if ft_group == 'Attribute':
            filter_modes = filters1
        elif ft_group == 'Performance':
            filter_modes = filters2
        elif ft_group == 'Follower':
            filter_modes = filters3
        else:
            filter_modes = [ft_group]

        if ft_group == 'Attribute':
            ft = st.selectbox('Filter mode', filter_modes, key=f'ft_{i}_{id}', index=filter_modes.index(id_state.get('ft', 0)))
            id_state['ft'] = ft
            explanations = {
                'Mandatory': "This filter removes all cases that do not have at least one event with one of the selected values.",
                'Forbidden': "This filter removes all cases that have at least one event with one of the selected values.",
                'Keep Selected': "This filter removes all events that do not have one of the selected values."
            }
            st.markdown(small_text(explanations[ft]), unsafe_allow_html=True)

            at = st.selectbox(
                'Attribute', original.columns,
                key=f'at_{i}_{id}', index=original.columns.tolist().index(id_state.get('at', 0)))
            id_state['at'] = at
            

            valores = ['* All values'] + list(original[at].unique())
            # st.write(valores)
            
            value = st.multiselect('Value', list(valores), key=f'value_{i}_{id}', default=id_state.get('value', []))
            id_state['value'] = value

            g = st.checkbox('Group by', key=f'g_{i}_{id}', value=id_state.get('g', False))
            id_state['g'] = g

            manip = [ft, (at, g), value]

        elif ft_group == 'Performance':

            ft = st.selectbox('Filter mode', filter_modes, key=f'ft_{i}')

            explanations = {
                'Path performance': "This filter selects event log traces that contain a specific transition between two user-selected activities that fall within a user-specified time interval.",
                'Case performance': "This filter selects event log traces whose duration falls within a user-specified time interval."
            }
            st.markdown(small_text(explanations[ft]), unsafe_allow_html=True)

            if ft == "Path performance":
                act1 = st.selectbox(
                    'From', original['concept:name'].unique(),
                    key=f'act1_{i}'
                )

                act2 = st.selectbox(
                    'To', original['concept:name'].unique(),
                    key=f'act2_{i}'
                )

                rango = st.slider("Minutes between them:", 1, 500, key=f'rang_{i}', value=st.session_state["rango"].get(f'rang_{i}', (25, 75)))
                manip = [ft, (act1, act2), rango]

            else: # 'Case performance':
                mode_options = ["Unique interval", "More than one interval"]
                mode = st.selectbox("Mode", mode_options, key=f'mode_{i}')

                if mode == "Unique interval":
                    rango = st.slider("Minutes between them:", 1, 500, key=f'rang_{i}', value= (25, 75))
                else:
                    n = st.number_input('Number of intervals', step=1, min_value=2, key=f'nrange_{i}')
                    rango = []
                    for j in range(n):
                        r1 = st.sidebar.slider(f"Minutes between the interval {j+1}:", 1, 500, key=f'rang2_{j}', value= (25, 75))
                        rango.append(r1)

                manip = [ft, mode, rango]

        elif ft_group == "Endpoints":
            st.markdown(small_text("This filter removes all cases in which the first and/or last events do not have one of the selected values."), unsafe_allow_html=True)

            options = ['* All values'] + list(pm4py.get_start_activities(df))
            act1 = st.multiselect('From', options, key=f'act1_{i}')

            options2 = ['* All values'] + list(pm4py.get_end_activities(df))
            act2 = st.multiselect('To', options2, key=f'act2_{i}')

            manip = [ft_group, act1, act2]

        elif ft_group == 'Rework':
            st.markdown(small_text("This filter selects event log traces where a concrete activity occurs more than once with a specified minimum frequency."), unsafe_allow_html=True)
            act1 = st.selectbox(
                'Activity', df['concept:name'].unique(),
                key=f'act1_{i}')

            value = st.number_input('Minimum frequency', step=1, key=f'value_{i}')

            manip = [ft_group, act1, value]

        

        elif ft_group == 'Timeframe':
            st.markdown(small_text("This filter selects event log traces that belong to a time period using the timestamp attribute."), unsafe_allow_html=True)
            
            r, l = st.columns(2)

            default_date1 = original['time:timestamp'][0].date()
            default_hour1 = datetime.now().time()

            default_date2 = original['time:timestamp'][0].date()
            default_hour2 = datetime.now().time()

            rango1 = r.date_input("From", key=f'date1_{i}', value=default_date1)
            h1 = l.time_input('Time From', key=f'hour1_{i}', value=default_hour1)
            start_datetime = datetime.combine(rango1, h1)

            rango2 = r.date_input("To", key=f'date2_{i}', value=default_date2)
            h2 = l.time_input('Time To', key=f'hour2_{i}', value=default_hour2)
            end_datetime = datetime.combine(rango2, h2)

            manip = [ft_group, (start_datetime, end_datetime)]
        
        else:
            ft = st.selectbox('Filter mode', filter_modes, key=f'ft_{i}')

            explanations = {
                'Directly Followed': "This filter selects event log traces where one activity is followed by another at some point, but not necessarily immediately.",
                'Eventually Followed': "This filter selects event log traces where one activity is eventually followed by another.",
                'Keep Selected Fragments': "This filter retains only a specific sub-process, defined as a part of the process between two specific activities."
            }
            st.markdown(small_text(explanations[ft]), unsafe_allow_html=True)

            nfollow = st.number_input('Number of fragments', step=1, min_value=1)
            
            k = 1
            lista_act = []

            act1 = st.selectbox(
                            'From', (original['concept:name'].unique()), 
                            key='act1_%s' % i)

            while (nfollow > 1 and k < nfollow):

                # st.write('prueba')
                actk = st.selectbox(
                        'To - From', (original['concept:name'].unique()), 
                        key='actk_%s' % k)

                lista_act.append(actk)
                k = k+1

            act2 = st.selectbox(
                        'To', (original['concept:name'].unique()), 
                        index=original['concept:name'].unique().tolist().index(st.session_state["act2"].get('act2_%s' % i, original['concept:name'].unique()[0])), 
                        key='act2_%s' % i)
            
            # st.write(lista_act)
            manip = [ft,(act1,act2),lista_act]

    ids.append(id)
    st.session_state['ids'] = ids

    return manip



def apply_manipulation(df, original, manip):

    # st.write(manip)

    filtered_dataframe={}
    
    dfs={}

    if not isinstance(df, dict):
        dfs['']=df
    else:
        
        dfs=df 

    # st.write('Atributos manipulacion: ', manip)

    ftype = manip[0]
    v1 = manip[1]
    v2 = manip[2]

    for key, df in dfs.items():

        if (ftype == 'Directly Followed'):
            activityFROM = v1[0] 
            activityTO = v1[1]
            if(v2==[]):
                filt = pm4py.filter_directly_follows_relation(df, 
                    [(activityFROM,activityTO)], activity_key='concept:name', 
                            case_id_key='case:concept:name', timestamp_key='time:timestamp')
                if(len(filt)!=0):
                    if(key==''):
                        filtered_dataframe[str(activityFROM) + " - " + str(activityTO)] = filt
                    else:
                        filtered_dataframe[key + " ; " + str(activityFROM) + " - " + str(activityTO)] = filt
            else:
                v2.insert(0, activityFROM)
                v2.append(activityTO)
                for frag in range(len(v2) - 1):
                    activityFROM = v2[frag]
                    activityTO = v2[frag + 1]
                    filt = pm4py.filter_directly_follows_relation(df, 
                        [(activityFROM, activityTO)], activity_key='concept:name', 
                            case_id_key='case:concept:name', timestamp_key='time:timestamp')

                    if(len(filt)!=0):
                        if(key==''):
                            filtered_dataframe[str(activityFROM) + " - " + str(activityTO)] = filt
                        else:
                            filtered_dataframe[key + " ; " + str(activityFROM) + " - " + str(activityTO)] = filt
                    
        elif(ftype == 'Eventually Followed'):

            activityFROM = v1[0] 
            activityTO = v1[1]
            if(v2==[]):
                filt = pm4py.filter_eventually_follows_relation(df, 
                    [(activityFROM,activityTO)], activity_key='concept:name', 
                            case_id_key='case:concept:name', timestamp_key='time:timestamp')
                if(len(filt)!=0):
                    if(key==''):
                        filtered_dataframe[str(activityFROM) + " - " + str(activityTO)] = filt
                    else:
                        filtered_dataframe[key + " ; " + str(activityFROM) + " - " + str(activityTO)] = filt
            else:
                v2.insert(0, activityFROM)
                v2.append(activityTO)
                for frag in range(len(v2) - 1):
                    activityFROM = v2[frag]
                    activityTO = v2[frag + 1]
                    filt = pm4py.filter_eventually_follows_relation(df, 
                        [(activityFROM, activityTO)], activity_key='concept:name', 
                            case_id_key='case:concept:name', timestamp_key='time:timestamp')

                    if(len(filt)!=0):
                        if(key==''):
                            filtered_dataframe[str(activityFROM) + " - " + str(activityTO)] = filt
                        else:
                            filtered_dataframe[key + " ; " + str(activityFROM) + " - " + str(activityTO)] = filt
                   

                    
        elif ftype == "Keep Selected Fragments":
            # st.write('prueba')
            activityFROM = v1[0] 
            activityTO = v1[1]
            if(v2==[]):
                filt = pm4py.filter_between(df, 
                    activityFROM,activityTO, activity_key='concept:name', 
                            case_id_key='case:concept:name', timestamp_key='time:timestamp')
                # st.write(filt)
                if(len(filt)!=0):
                    if(key==''):
                        filtered_dataframe[str(activityFROM) + " - " + str(activityTO)] = filt
                    else:
                        filtered_dataframe[key + " ; " + str(activityFROM) + " - " + str(activityTO)] = filt
                
            else:
                v2.insert(0, activityFROM)
                v2.append(activityTO)
                for frag in range(len(v2) - 1):
                    activityFROM = v2[frag]
                    activityTO = v2[frag + 1]
                    filt = pm4py.filter_between(df, 
                        activityFROM, activityTO, activity_key='concept:name', 
                            case_id_key='case:concept:name', timestamp_key='time:timestamp')

                    if(len(filt)!=0):
                        if(key==''):
                            filtered_dataframe[str(activityFROM) + " - " + str(activityTO)] = filt
                        else:
                            filtered_dataframe[key + " ; " + str(activityFROM) + " - " + str(activityTO)] = filt
                    
            
      
        elif ftype == 'Mandatory':
            g = v1[1]
            atr = v1[0] 
            # atr = 'City'
            if v2 == ['* All values']: 
                valores = df[atr].unique()
                for v in valores:
                    grupo = pm4py.filter_trace_attribute_values(df, atr, [v])
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[str([v])] = grupo
                        else:
                            filtered_dataframe[key + " - " + str([v])] = grupo
                        
            else:  
                # st.write([v2])  
                if(g==True):
                    # st.write('prueba')
                    for v in v2:
                        grupo = pm4py.filter_trace_attribute_values(df, atr, [v])
                        if(len(grupo)!=0):
                            if(key==""):
                                filtered_dataframe[str([v])] = grupo
                            else:
                                filtered_dataframe[key + " - " + str([v])] = grupo
                            # filtered_dataframe[v] = grupo
                # st.write(v1, v2)
                else:
                    
                    # st.write(df,atr,v2)
                    grupo = pm4py.filter_trace_attribute_values(df, atr, v2)
                    # st.write(grupo)
                    
                    # st.write(grupo)
                    if(len(grupo)!=0):
                        
                        if(key==""):
                            filtered_dataframe[str(v2)] = grupo
                        else:
                            filtered_dataframe[key + " - " + str(v2)] = grupo
                        
        elif ftype == 'Keep Selected':
            g = v1[1]
            atr = v1[0] 
            
            if v2 == ['* All values']:  
                valores = df[atr].unique()
                for v in valores:
                    grupo = df[df[atr]==v]
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[str([v])] = grupo
                        else:
                            filtered_dataframe[key + " - " + str([v])] = grupo
                        # filtered_dataframe[v] = grupo
            else:   
                if(g==True):
                    valores = df[atr].unique()
                    for v in v2:
                        grupo = df[df[atr]==v]
                        if(len(grupo)!=0):
                            # filtered_dataframe[v] = grupo
                            if(key==""):
                                filtered_dataframe[str([v])] = grupo
                            else:
                                filtered_dataframe[key + " - " + str([v])] = grupo
                else:                    
                    grupo = pm4py.filter_event_attribute_values(df, atr, v2,  level='event')
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[str(v2)] = grupo
                        else:
                            filtered_dataframe[key + " - " + str(v2)] = grupo

        elif ftype == 'Forbidden':   
            g = v1[1] 
            atr = v1[0] 
            
            if v2 == ['* All values']: 
                valores = df[atr].unique()
                for v in valores:
                    grupo = pm4py.filter_trace_attribute_values(df, atr, [v], retain=False)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[str([v])] = grupo
                        else:
                            filtered_dataframe[key + " - " + str([v])] = grupo
                        # filtered_dataframe[v] = grupo
            else:       
                if(g==True):
                    for v in v2:
                        grupo = pm4py.filter_trace_attribute_values(df, atr, [v], retain=False)
                        if(len(grupo)!=0):
                            if(key==""):
                                filtered_dataframe[str([v])] = grupo
                            else:
                                filtered_dataframe[key + " - " + str([v])] = grupo
                            # filtered_dataframe[v] = grupo
                else:
                    grupo = pm4py.filter_trace_attribute_values(df, atr, v2, retain=False)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[str(v2)] = grupo
                        else:
                            filtered_dataframe[key + " - " + str(v2)] = grupo

        elif ftype == 'Rework':
            # log = check_log(df)
            grupo = pm4py.filter_activities_rework(df, v1, v2)
            # grupo = pm4py.convert_to_dataframe(filtered_log)
            if(len(grupo)!=0):
                if(key==""):
                    filtered_dataframe[v1 + ': ' + str(v2)] = grupo
                else:
                    filtered_dataframe[key + " - " + v1 + ': ' + str([v2])] = grupo
                
        elif ftype == 'Endpoints':
            # log = check_log(df)
            log_start = pm4py.get_start_activities(df)
            log_end = pm4py.get_end_activities(df)

            if (v1 == ['* All values'] and v2 == []):
                for a in log_start:
                    grupo = pm4py.filter_start_activities(df, [a])
                    # grupo = pm4py.convert_to_dataframe(filtered_log)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[a + ' - ' + 'endpoints'] = grupo
                        else:
                            filtered_dataframe[key + " - " + a + ' - ' + 'endpoints'] = grupo
                        

            elif (v1 == ['* All values'] and v2 == ['* All values']):
                for a in log_start:
                    for e in log_end:
                        filtered_log = pm4py.filter_start_activities(df, [a])
                        grupo = pm4py.filter_end_activities(filtered_log, [e])
                        # grupo = pm4py.convert_to_dataframe(filtered_log)
                        if(len(grupo)!=0):
                            if(key==""):
                                filtered_dataframe[a + ' - ' + e] = grupo
                            else:
                                filtered_dataframe[key + " - " + a + ' - ' + e] = grupo
                            
            
            elif (v1 == ['* All values'] and v2 != []):
                for a in log_start:
                    filtered_log = pm4py.filter_start_activities(df, [a])
                    grupo = pm4py.filter_end_activities(filtered_log, v2)
                    # grupo = pm4py.convert_to_dataframe(filtered_log)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[a + ' - ' + str(v2)] = grupo
                        else:
                            filtered_dataframe[key + ' - ' + a + ' - ' + str(v2)] = grupo

            elif (v1 != [] and v2 == ['* All values']):
                for e in log_end:
                    filtered_log = pm4py.filter_end_activities(df, [e])
                    grupo = pm4py.filter_start_activities(filtered_log, v1)
                    # grupo = pm4py.convert_to_dataframe(filtered_log1)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[str(v1) + ' - ' + e] = grupo
                        else:
                            filtered_dataframe[key + ' - ' + str(v1) + ' - ' + e] = grupo

            elif (v1 == [] and v2 == ['* All values']):
                for e in log_end:
                    grupo = pm4py.filter_end_activities(df, [e])
                    # grupo = pm4py.convert_to_dataframe(filtered_log)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe['startpoints' + ' - ' + e] = grupo
                        else:
                            filtered_dataframe[key + ' - ' + 'startpoints' + ' - ' + e] = grupo
            
            elif (v1 != [] and v2 != []):
                filtered_log = pm4py.filter_start_activities(df, v1)
                grupo = pm4py.filter_end_activities(filtered_log, v2)
                # grupo = pm4py.convert_to_dataframe(filtered_log)
                if(len(grupo)!=0):
                    if(key==""):
                    # st.write(a)
                        filtered_dataframe[str(v1) + ' - ' + str(v2)] = grupo
                    else:
                        filtered_dataframe[key + ' - ' + str(v1) + ' - ' + str(v2)] = grupo

            
            elif (v1 == [] and v2 != []):
                grupo = pm4py.filter_end_activities(df, v2)
                # grupo = pm4py.convert_to_dataframe(filtered_log)
                if(len(grupo)!=0):
                    if(key==""):
                        filtered_dataframe['startpoints -' + str(v2)] = grupo
                    else:
                        filtered_dataframe[key + ' - ' + 'startpoints -' + str(v2)] = grupo

            elif (v1 != [] and v2 == []):
                grupo = pm4py.filter_start_activities(df, v1)
                # grupo = pm4py.convert_to_dataframe(filtered_log)
                if(len(grupo)!=0):
                    if(key==""):
                        filtered_dataframe['selected startpoints -  endpoints'] = grupo
                    else:
                        filtered_dataframe[key + ' - ' + 'selected startpoints -  endpoints'] = grupo

            else:
                if(key==""):
                    filtered_dataframe[str(v1) + ' - endpoints'] = df
                else:
                    filtered_dataframe[key + ' - ' + str(v1) + ' - endpoints'] = df

        elif ftype == 'Path performance':
            # log = check_log(df)
            grupo = pm4py.filter_paths_performance(df, v1, v2[0]*60, v2[1]*60)
            # grupo = pm4py.convert_to_dataframe(filtered_log)
            if(len(grupo)!=0):
                if(key==""):
                    filtered_dataframe[str(v1)] = grupo
                else:
                    filtered_dataframe[key + ' - ' + str(v1)] = grupo
        
        elif ftype == 'Timeframe':

            grupo = pm4py.filter_time_range(df, v2[0] , v2[1], mode=v1)

            # Parace que para events funciona bien pero para contained y intersecting no filtra bien 
            if(len(grupo)!=0):
                if(key==""):
                    filtered_dataframe[str(v2[0]) + ' - ' + str(v2[1])] = grupo
                else:
                    filtered_dataframe[key + ' - ' + str(v2[0]) + ' - ' + str(v2[1])] = grupo

        elif ftype == 'Case performance':
            
            # log = check_log(df)
            if(v1=="Unique interval"):
                grupo = pm4py.filter_case_performance(df, v2[0]*60, v2[1]*60)
                # grupo = pm4py.convert_to_dataframe(filtered_log)
                if(len(grupo)!=0):
                    if(key==""):
                        filtered_dataframe[v2] = grupo
                    else:
                        filtered_dataframe[key + ' - ' + str(v2)] = grupo
            else:
                j=0
                for j in rango:
                    # st.write(j, rango)
                    grupo = pm4py.filter_case_performance(df, j[0]*60, j[1]*60)
                    # grupo = pm4py.convert_to_dataframe(filtered_log)
                    if(len(grupo)!=0):
                        if(key==""):
                            filtered_dataframe[j] = grupo
                        else:
                            filtered_dataframe[key + ' - ' + j] = grupo
                    #     st.write('si hay resultados')
                    # else:
                    #     st.write('No hay resultados')
    # if(st.sidebar.button('Filtrar')):
    #     st.session_state.generate_pressed = False
    # st.write(filtered_dataframe)
    
    return filtered_dataframe 