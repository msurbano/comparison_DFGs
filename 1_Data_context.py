import streamlit as st
import numpy as np
import pandas as pd
import pm4py
import copy
import deprecation
import statisticslog
import os
import recommendations
import specification
from PIL import Image
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
import json
import re
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
from graphviz import Digraph
import metricas
import load_data
import manipulation
import dfg_creation
import dfg_properties
import positions_creation
import visualization


st.set_page_config(page_title="Main page", layout="wide")

pd.set_option("styler.render.max_elements", 2000000)


st.title("VISCoPro :mag_right::chart_with_downwards_trend:")
# st.markdown("""---""")


# --------------------------------------------------------------------------------------



if "generate_pressed" not in st.session_state:
    st.session_state.generate_pressed = False

if "filter_types" not in st.session_state:
        st.session_state["filter_types"] = {}

if "filter_type_group" not in st.session_state:
        st.session_state["filter_type_group"] = {}

if "attribute" not in st.session_state:
        st.session_state["attribute"] = {}

if "values" not in st.session_state:
        st.session_state["values"] = {}

if "act1" not in st.session_state:
        st.session_state["act1"] = {}

if "act2" not in st.session_state:
        st.session_state["act2"] = {}

if "actk" not in st.session_state:
        st.session_state["actk"] = {}

if "rango" not in st.session_state:
    st.session_state["rango"] = {}

if "number_values" not in st.session_state:
    st.session_state["number_values"] = {}

if "range_values" not in st.session_state:
    st.session_state["range_values"] = {}

if "modes" not in st.session_state:
    st.session_state["modes"] = {}

if "nrange" not in st.session_state:
    st.session_state["nrange"] = {}

if "rango2" not in st.session_state:
    st.session_state["rango2"] = {}

if "input_values" not in st.session_state:
    st.session_state["input_values"] = {}

if "group" not in st.session_state:
    st.session_state["group"] = {}

if "nfollow" not in st.session_state:
    st.session_state["nfollow"] = 1

if "lista_act" not in st.session_state:
    st.session_state["lista_act"] = {}

if 'original' not in st.session_state:
    st.session_state.original = pd.DataFrame()

if 'positions' not in st.session_state:
    st.session_state.positions = {}

if 'positions_edges' not in st.session_state:
    st.session_state.positions_edges = {}

if 'viz' not in st.session_state:
    st.session_state.viz = Digraph()

if 'mapeo' not in st.session_state:
    st.session_state.mapeo = {}

if 'sa' not in st.session_state:
    st.session_state.sa = []

if 'ea' not in st.session_state:
    st.session_state.ea = []

if 'unified' not in st.session_state:
    st.session_state.unified = pd.DataFrame()

if 'nodesDFG' not in st.session_state:
    st.session_state.nodesDFG = set()

if 'edgesDFG' not in st.session_state:
    st.session_state.edgesDFG = set()

if 'colores' not in st.session_state:
    st.session_state.colores={}

if 'delete_act' not in st.session_state:
    st.session_state.delete_act=[]

if 'viz_edges' not in st.session_state:
    st.session_state.viz_edges = set()

if 'reference_nodes' not in st.session_state:
    st.session_state.reference_nodes = set()

if 'reference_edges' not in st.session_state:
    st.session_state.reference_edges = set()

if 'reference_sa' not in st.session_state:
    st.session_state.reference_sa = set()

if 'reference_ea' not in st.session_state:
    st.session_state.reference_ea = set()

# ------------------------------------------------------------------------------------------------

mensaje_container = st.empty()
sample_data = []
df = load_data.cargar_datos(mensaje_container, sample_data)
dic_original = {}

#  ------------------------------------------------------------------------------------------------
# Elementos de la interfaz
#  ------------------------------------------------------------------------------------------------
    

if len(st.session_state.original):
    dataframe = st.session_state.original

    if 'inicial' not in st.session_state:
        st.session_state.inicial = dataframe

    

    if dataframe is not None:

        if st.checkbox('Show Event log :page_facing_up:'):
            dataframe


        nodes, metric, perc_act, perc_path = dfg_properties.dfg_options(dataframe)

        

        # tupla = visualization.search_differences()
        # colores = dfg_creation.asignar_colores(tupla[1][1])
        # st.session_state.colores = colores


        # positions={}
        # dic_original['original'] = dataframe
        # dfg_original = dfg_creation.df_to_dfg(dic_original,nodes,metric)
        # dfg_creation.threshold(dfg_original, metric, 100, 100, nodes, positions)

        cont = 0        

        n = st.sidebar.number_input('Number of manipulation actions :pick:', step=1, min_value=0)
        filtered = pd.DataFrame()
        original = dataframe


        # col1, col2, col3, col4 = st.columns(4)
        # z, delete_act = visualization.show_activities(col2, original)

        # if(z=='Filter events by activities'):
        #     original = visualization.filter_events(original, delete_act)
        # #     # filtered = manipulation.apply_manipulation(filtered, original, 
        # #     #     ['Keep Selected', ('concept:name', False), delete_act])
        # #     st.write(original)
        #     delete_act = []

        st.markdown("""---""")
        # col1, col2, col3, col4 = st.columns(4)
        # z, delete_act = visualization.show_activities(col2, original)
        # if(z=='Filter events by activities'):
        #     original = pm4py.filter_event_attribute_values(original, 'concept:name', delete_act,  level='event')
        #     delete_act=[]
        #     st.session_state.inicial = original


        dic_initial = {}
        dic_initial['Initial'] = original

        dfg_initial = positions_creation.df_to_dfg(dic_initial,nodes,'Absolute frequency')
        # positions_creation.nodes_edges(dfg_initial, 'Absolute frequency', 100, 100, nodes)
        positions_creation.threshold(dfg_initial, 'Absolute frequency', 100, 100, nodes)
        

        viz = st.session_state.viz
        # st.write('Initial DFG to fix node positions (DOT engine)')
        # viz



        if(n==0):
            filtered={}
            filtered['Initial'] = dataframe
        else:
            while (cont < n):
                try:
                    manip = manipulation.manipulation_options(dataframe, original, cont)
                    filtered = manipulation.apply_manipulation(dataframe, original, manip)
                    
                except Exception as e:
                    # st.error(f"Error")
                    st.error(e)
                    break

                dataframe = filtered
                cont = cont+1


        # st.markdown("""---""")

# ------------------------------------------------------------------------------------
        # unified = pd.DataFrame()

        # for key, subset in filtered.items():
        #     # st.write(subset)
        #     unified = pd.concat([unified, subset], ignore_index=True)
        # dic_uni= {}
        # dic_uni['Uni'] = unified
        # dfg_uni = positions_creation.df_to_dfg(dic_uni,nodes,'Absolute frequency')
        # positions_creation.threshold(dfg_uni, 'Absolute frequency', 100, 100, nodes)

        # viz = st.session_state.viz
        # st.write('Unified DFG to fix node positions (DOT engine)')
        # viz
# ------------------------------------------------------------------------------------
        
        

        
        

        # if(z=='Filter events by activities'):
        #     filtrado = pm4py.filter_event_attribute_values(original, 'concept:name', delete_act, retain=False, level='event')
        #     dic_initial['Initial'] = filtrado
        #     dfg_initial = positions_creation.df_to_dfg(dic_initial,nodes,'Absolute frequency')
        #     positions_creation.threshold(dfg_initial, 'Absolute frequency', 100, 100, nodes)
        #     viz = st.session_state.viz
        #     delete_act=[]

        col1, col2, col3, col4 = st.columns(4)
        filtered = visualization.zoom_fragment(col1, filtered)
       
        left_column, right_column = st.columns([1, 6])

        if( st.sidebar.button('Generate collection of DFGs', type='primary')):
            st.session_state.generate_pressed = True
        
        
        if st.session_state.generate_pressed :

            tupla = visualization.search_differences(filtered.keys())
            colores = dfg_creation.asignar_colores(tupla[1][1])
            st.session_state.colores = colores

            z, delete_act = visualization.show_activities(col2, original)

            # filtered = visualization.zoom_fragment(filtered)

            if (filtered == {}):
                st.error('No results (no event log subset matches the specified manipulation actions).')
                st.session_state.generate_pressed = False
            else:

                st.markdown("""---""")
                
                dfgs = dfg_creation.df_to_dfg(filtered,nodes,metric)
                # st.write(dfgs.items())
                st.session_state.dataframe = dfgs
                copia_dict = copy.deepcopy(dfgs)

                left_column, right_column = st.columns(2)
                order_options = ['By the search', "Mean case duration", "Median cycle time", "Number of events", "Number of traces", "Number of activities", "Number of variants"]
                order_by = left_column.selectbox("Order by:", order_options, index=5, key='context_order') 
                
                
                stats = dfg_creation.threshold(copia_dict, metric, perc_act, perc_path, nodes, tupla, delete_act)
                # st.write(stats)
                # for g in stats:
                #     g["svg_path"]
                dfg_creation.show_DFGs(stats, order_by, metric)
 
                
                st.markdown("""---""")
                i=0
                with st.expander(" **Pattern recommendation**  :bulb:"):
                    # order_options = ["Mean case duration", "Median cycle time", "Events", "Traces", "Activities", "Variants"]
                    # order_by = st.selectbox("Order by:", order_options, index=0, key='order'+str(i))
                    # if (filtered != {}):
                    recommendations.pattern_recommendations(filtered, nodes, metric, perc_act, perc_path)
                with st.expander(" **Pattern specification** :memo:"):
                    # if (filtered != {}):
                    
                    specification.pattern(original, dfgs, nodes, metric, perc_act, perc_path,i)
                i+=1
                
            



