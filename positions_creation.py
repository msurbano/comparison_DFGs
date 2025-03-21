import streamlit as st
import numpy as np
import pandas as pd
import pm4py
import copy
import deprecation
import os
import networkx as nx
import matplotlib.pyplot as plt
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
# from pm4py.visualization.dfg.variants.frequency import apply
from pm4py.visualization.dfg.variants import performance
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from PIL import Image
from io import StringIO
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from streamlit import session_state as ss
import statisticslog
import tempfile
from copy import copy

from graphviz import Digraph

from pm4py.statistics.attributes.log import get as attr_get
from pm4py.objects.dfg.utils import dfg_utils
from pm4py.util import xes_constants as xes
from pm4py.visualization.common.utils import *
from pm4py.util import exec_utils
from enum import Enum
from pm4py.util import constants
from typing import Optional, Dict, Any, Tuple
from pm4py.objects.log.obj import EventLog
from collections import Counter


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    FORMAT = "format"
    MAX_NO_EDGES_IN_DIAGRAM = "maxNoOfEdgesInDiagram"
    START_ACTIVITIES = "start_activities"
    END_ACTIVITIES = "end_activities"
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    FONT_SIZE = "font_size"
    BGCOLOR = "bgcolor"
    STAT_LOCALE = "stat_locale"

positions={}
positions_edges={}


def removeEdges(G,filteredEdges):
    for edge in list(G.edges):
        if edge not in filteredEdges:
            G.remove_edge(*edge)
            
    return G
            
def removeNodes(G,filteredNodes):
    for node in list(G.nodes):
        if node not in filteredNodes:
            G.remove_node(node)
            
    return G

def returnTotalRepetitions(df):
    totalRep=df.groupby("Transitions").apply(lambda x: (x['case_ids'].value_counts() > 1).sum())
    return totalRep

def returnCaseFreqEdges(df):
    case_freq=df.groupby('Transitions')['case_ids'].nunique()
    return case_freq

def returnMaxRepititionsEdges(df):
    maxRep=df.groupby(["case_ids","Transitions"]).apply(lambda x: len(x)).reset_index().groupby("Transitions").apply(lambda x: max(x[0]))
    return maxRep

def returnEdgesInfo(df,concept_name,case_concept_name,timestamp,metric):
    df_sorted=df.reset_index().sort_values(by=[case_concept_name,timestamp,"index"])
    first_events=df_sorted.iloc[0:len(df_sorted)-1] 
    second_events=df_sorted.iloc[1:len(df_sorted)]
    transitions=[]
    case_ids=[]

    for (index1,row1),(index2,row2) in zip(first_events.iterrows(),second_events.iterrows()):
        if row1[case_concept_name]==row2[case_concept_name]:
            transitions.append((row1[concept_name],row2[concept_name]))
            case_ids.append(row2[case_concept_name])
        else:
            continue      
    df_edges=pd.DataFrame.from_dict({"Transitions":transitions,"case_ids":case_ids})

    if(metric == 'Case Frequency'):
        res=returnCaseFreqEdges(df_edges)
    elif(metric == 'Max Repetitions'):
        res=returnMaxRepititionsEdges(df_edges)
    elif(metric == 'Total Repetitions'):
        res = returnTotalRepetitions(df_edges)
    else:
        res={}


    return res

def df_to_dfg(dfs,nodes,metric):
    # st.write('paso 1')
    # delete_act = st.session_state.delete_act
    
    # if(delete_act is None):
    dic={}

    for key, df in dfs.items():
        dfg, sa, ea = pm4py.discover_dfg(df, activity_key=nodes)   
        grafo = defineGraphFrequency(df, dfg, nodes, metric)
        dic[key] = {'dfg': dfg, 'sa': sa, 'ea': ea, 'df': df, 'graph':grafo}
    # else:

    #     dic={}

    #     for key, df in dfs.items():
    #         df = pm4py.filter_event_attribute_values(df, 'concept:name', delete_act, retain=False, level='event')
    #         dfg, sa, ea = pm4py.discover_dfg(df, activity_key=nodes)   
    #         grafo = defineGraphFrequency(df, dfg, nodes, metric)
    #         dic[key] = {'dfg': dfg, 'sa': sa, 'ea': ea, 'df': df, 'graph':grafo}

         
    
    return dic

def defineGraphFrequency(df, dfg, nodes, metric): 
    # st.write('paso 2')
    lista_nodos = list(df[nodes].unique())
    G = nx.DiGraph()
    dic_paths = {}
    dic_max = {}
    dic_nodes = {}

    # print(metric)

    abs_freq_nodes = df[nodes].value_counts().to_dict()
     
    for key in dfg.keys():  
        # st.write(key)          
        dic_paths[key]={}
        dic_paths[key]['abs_freq'] = dfg[key]

    for key in lista_nodos:
        # st.write(case_freq_nodes[key])
        dic_nodes[key]={}
        dic_nodes[key]['abs_freq'] = abs_freq_nodes[key]
    
    for nodo, propiedades in dic_nodes.items():
        G.add_node(nodo, **propiedades)
    
    for edge, propiedades in dic_paths.items():
        actividad_origen, actividad_destino = edge
        G.add_edge(actividad_origen,actividad_destino, **propiedades)  
        
    return G

def apply_custom(dfg, start_activities, end_activities, log, parameters, activities_count, soj_time):
        
    global positions
    
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY)
    image_format = exec_utils.get_param_value(Parameters.FORMAT, parameters, "png")
    max_no_of_edges_in_diagram = exec_utils.get_param_value(Parameters.MAX_NO_EDGES_IN_DIAGRAM, parameters, 100000)
    #     start_activities = exec_utils.get_param_value(Parameters.START_ACTIVITIES, parameters, {})
    #     end_activities = exec_utils.get_param_value(Parameters.END_ACTIVITIES, parameters, {})
    font_size = exec_utils.get_param_value(Parameters.FONT_SIZE, parameters, 12)
    font_size = str(font_size)
    activities = dfg_utils.get_activities_from_dfg(dfg)
    bgcolor = exec_utils.get_param_value(Parameters.BGCOLOR, parameters, constants.DEFAULT_BGCOLOR)
    stat_locale = exec_utils.get_param_value(Parameters.STAT_LOCALE, parameters, {})

    if activities_count is None:
        if log is not None:
            activities_count = attr_get.get_attribute_values(log, activity_key, parameters=parameters)
        else:
            # the frequency of an activity in the log is at least the number of occurrences of
            # incoming arcs in the DFG.
            # if the frequency of the start activities nodes is also provided, use also that.
            activities_count = Counter({key: 0 for key in activities})
            for el in dfg:
                activities_count[el[1]] += dfg[el]
            if isinstance(start_activities, dict):
                for act in start_activities:
                    activities_count[act] += start_activities[act]

    if soj_time is None:
        if log is not None:
            soj_time = soj_time_get.apply(log, parameters=parameters)
        else:
            soj_time = {key: 0 for key in activities}

    # st.write('contenido',  positions)
        # st.write('no positions')
    st.session_state.ea = end_activities
    st.session_state.sa = start_activities

    graphviz_visualization(activities_count, dfg, image_format=image_format, measure="frequency",
                                  max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
                                  start_activities=start_activities, end_activities=end_activities, 
                                  soj_time=soj_time, font_size=font_size, bgcolor=bgcolor, stat_locale=stat_locale)

def get_min_max_value(dfg):
    """
    Gets min and max value assigned to edges
    in DFG graph

    Parameters
    -----------
    dfg
        Directly follows graph

    Returns
    -----------
    min_value
        Minimum value in directly follows graph
    max_value
        Maximum value in directly follows graph
    """
    min_value = 9999999999
    max_value = -1

    for edge in dfg:
        if dfg[edge] < min_value:
            min_value = dfg[edge]
        if dfg[edge] > max_value:
            max_value = dfg[edge]

    return min_value, max_value

def assign_penwidth_edges(dfg):
    """
    Assign penwidth to edges in directly-follows graph

    Parameters
    -----------
    dfg
        Direcly follows graph

    Returns
    -----------
    penwidth
        Graph penwidth that edges should have in the direcly follows graph
    """
    penwidth = {}
    min_value, max_value = get_min_max_value(dfg)
    for edge in dfg:
        v0 = dfg[edge]
        v1 = get_arc_penwidth(v0, min_value, max_value)
        penwidth[edge] = str(v1)

    return penwidth

def get_activities_color(activities_count):
    """
    Get frequency color for attributes

    Parameters
    -----------
    activities_count
        Count of attributes in the log

    Returns
    -----------
    activities_color
        Color assigned to attributes in the graph
    """
    activities_color = {}

    min_value, max_value = get_min_max_value(activities_count)

    for ac in activities_count:
        v0 = activities_count[ac]
        """transBaseColor = int(
            255 - 100 * (v0 - min_value) / (max_value - min_value + 0.00001))
        transBaseColorHex = str(hex(transBaseColor))[2:].upper()
        v1 = "#" + transBaseColorHex + transBaseColorHex + "FF"""

        v1 = get_trans_freq_color(v0, min_value, max_value)

        activities_color[ac] = v1

    return activities_color

def graphviz_visualization(activities_count, dfg, image_format="png", measure="performance",
                           max_no_of_edges_in_diagram=100000, start_activities=None, end_activities=None, soj_time=None,
                            font_size="12", bgcolor=constants.DEFAULT_BGCOLOR, stat_locale=None):
    """
    Do GraphViz visualization of a DFG graph

    Parameters
    -----------
    activities_count
        Count of attributes in the log (may include attributes that are not in the DFG graph)
    dfg
        DFG graph
    image_format
        GraphViz should be represented in this format
    measure
        Describes which measure is assigned to edges in direcly follows graph (frequency/performance)
    max_no_of_edges_in_diagram
        Maximum number of edges in the diagram allowed for visualization
    start_activities
        Start activities of the log
    end_activities
        End activities of the log
    soj_time
        For each activity, the sojourn time in the log
    stat_locale
        Dict to locale the stat strings
    
    Returns
    -----------
    viz
        Digraph object
    """
    #     start_activities = {'Check-in': 17}
    #     print(start_activities)
    
    # st.write('si')

    global positions
    global positions_edges
    
    if start_activities is None:
        start_activities = {}
    if end_activities is None:
        end_activities = {}
    if stat_locale is None:
        stat_locale = {}

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    #     viz = Digraph("", filename=filename.name, engine='neato', graph_attr={'bgcolor': bgcolor})
    viz = Digraph("", filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor, 'rankdir': 'LR'}, strict=True)
    
    # first, remove edges in diagram that exceeds the maximum number of edges in the diagram
    dfg_key_value_list = []
    for edge in dfg:
        dfg_key_value_list.append([edge, dfg[edge]])
    # more fine grained sorting to avoid that edges that are below the threshold are
    # undeterministically removed
    dfg_key_value_list = sorted(dfg_key_value_list, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
    dfg_key_value_list = dfg_key_value_list[0:min(len(dfg_key_value_list), max_no_of_edges_in_diagram)]
    dfg_allowed_keys = [x[0] for x in dfg_key_value_list]
    dfg_keys = list(dfg.keys())
    for edge in dfg_keys:
        if edge not in dfg_allowed_keys:
            del dfg[edge]

    # calculate edges penwidth
    penwidth = assign_penwidth_edges(dfg)
    activities_in_dfg = set()
    activities_count_int = copy(activities_count)

    for edge in dfg:
        activities_in_dfg.add(edge[0])
        activities_in_dfg.add(edge[1])

    # assign attributes color
    activities_color = get_activities_color(activities_count_int)

    # represent nodes
    viz.attr('node', shape='box')

    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities_in_dfg)))

    activities_map = {}
    #     x=0
    #     y=1

    for act in activities_to_include:
    #         position = f"{x},{y}!"
        if "frequency" in measure and act in activities_count_int:
    #             print(act)
            viz.node(str(hash(act)), act + " (" + str(activities_count_int[act]) + ")", style='filled',
                     fillcolor=activities_color[act], fontsize=font_size)
    #             pos=position)
            activities_map[act] = str(hash(act))
        else:
            stat_string = human_readable_stat(soj_time[act], stat_locale)
            viz.node(str(hash(act)), act + f" ({stat_string})", fontsize=font_size)
    #             pos=position)
            activities_map[act] = str(hash(act))
    #         x=x+3
    #         y=y+1 

    # make edges addition always in the same order
    dfg_edges = sorted(list(dfg.keys()))

    st.session_state.viz_edges = dfg_edges

    viz.attr(overlap='false',splines="true")
    # represent edges
    for edge in dfg_edges:
        if "frequency" in measure:
            label = str(dfg[edge])
        else:
            label = human_readable_stat(dfg[edge], stat_locale)
        viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size)

    start_activities_to_include = [act for act in start_activities if act in activities_map]
    end_activities_to_include = [act for act in end_activities if act in activities_map]


    if start_activities_to_include:
        viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
        for act in start_activities_to_include:
            label = str(start_activities[act]) if isinstance(start_activities, dict) else ""
            viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="dashed")

    if end_activities_to_include:
        # <&#9632;>
        viz.node("@@endnode", "<&#9632;>", shape='doublecircle', fontsize="10")
        for act in end_activities_to_include:
            label = str(end_activities[act]) if isinstance(end_activities, dict) else ""
            viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="dashed")

    viz.attr(overlap='false',splines="true")

    viz.format = image_format 

    dot_output = viz.pipe(format="plain").decode("utf-8")

    # Procesar la salida para extraer posiciones
    mapeo={}
    for line in dot_output.split("\n"):
        parts = line.split()
        # st.write(parts)
    
        if parts and parts[0] == "node":
            node_id = parts[1]  # Identificador del nodo
            x, y = float(parts[2]), float(parts[3])  # Coordenadas asignadas por dot
            raw_label = " ".join(parts[6:]).strip('"')  # Extraer la etiqueta completa

            # Limpiar el nombre de la actividad
            clean_label = re.split(r' \(|"', raw_label)[0]  # Tomar solo el texto antes de '(' o '"'
    #         if not(clean_label in positions.keys()):
            positions[clean_label] = (x, y)  # Guardar usando el nombre limpio  
            mapeo[node_id]=clean_label
            # st.write(positions)
        
        # elif(parts and parts[0]=="edge"):
        #     source_id = parts[1]
        #     target_id = parts[2]

        #     # Extraer posiciones de los nodos en la arista
        #     source_pos  = (float(parts[4]), float(parts[5]))  # Coordenadas de origen
        #     target_pos  = (float(parts[6]), float(parts[7]))  # Coordenadas de destino

        #     source_label = mapeo[source_id]
        #     target_label = mapeo[target_id]

            # positions_edges[(source_label,target_label)] = ((x1,y1), (x2,y2))
            # intermediate_points = []
            # for i in range(8, len(parts) - 4, 2):
            #     intermediate_points.append((float(parts[i]), float(parts[i + 1])))

            # positions_edges[(source_label, target_label)] = {
            #     'source_pos': source_pos,
            #     'target_pos': target_pos,
            #     'intermediate_points': intermediate_points
            # }

    st.session_state.viz = viz
    st.session_state.mapeo = mapeo

    # return positions, positions_edges

def prueba(nodes):
    prueba = st.session_state.dataframe
    for key, df in prueba.items():
        st.write('prueba')
        st.write(df)

def threshold(datos, metric, a, p, nodes):
    dic={}
    stats_list = [] 
    ident = 0

    for key, dfg in datos.items():
        df = dfg['df']
        dfg_ini = dfg['dfg']

        ac = dict(df[nodes].value_counts())

        if(p==100 and a==100):
            dfg_path = dfg_ini
            sa = dfg['sa']
            ea = dfg['ea']
        elif(p==100):
            dfg_path, sa, ea, ac = dfg_filtering.filter_dfg_on_activities_percentage(dfg['dfg'], dfg['sa'], dfg['ea'], ac, a/100)
        elif(a==100):
            dfg_path, sa, ea, ac = dfg_filtering.filter_dfg_on_paths_percentage(dfg['dfg'], dfg['sa'], dfg['ea'], ac, p/100)
        else:
            dfg_act, sa, ea, ac = dfg_filtering.filter_dfg_on_activities_percentage(dfg['dfg'], dfg['sa'], dfg['ea'], ac, a/100)
            dfg_path, sa, ea, ac = dfg_filtering.filter_dfg_on_paths_percentage(dfg_act, sa, ea, ac, p/100)

            
        G = dfg['graph']
            
        G_nodes_filtered=removeEdges(G,list(dfg_path.keys()))
        G_edges_filtered=removeNodes(G_nodes_filtered,list(ac.keys()))

        measure="abs_freq"
            
        metric_nodes=dict(G.nodes.data(measure))
            
        list_edges=list(G.edges.data())
        dfg_custom={(edge[0],edge[1]):edge[2][measure] for edge in list_edges}

        apply_custom(dfg_custom,sa,ea,None,None,metric_nodes,None)
            
        ident = ident + 1

def nodes_edges(datos, metric, a, p, nodes):
    dic={}
    stats_list = [] 
    ident = 0

    for key, dfg in datos.items():
        df = dfg['df']
        dfg_ini = dfg['dfg']

        st.session_state.nodesDFG = dfg['graph'].nodes
        st.session_state.edgesDFG = dfg['graph'].edges
        
   

def show_DFGs(stats_list, order, metric):
    # st.table(stats_list)
    # Display sorted results
    sorted_stats = sorted(stats_list, key=lambda x: x[order], reverse=True)

    # st.table(sorted_stats)

    for stat in sorted_stats:
        st.write(str(stat['key']))
        left_column, right_column = st.columns(2)

        if metric in ['Mean Cycle Time', 'Median Cycle Time', 'StDev Cycle Time', 'Total Cycle Time']:
            # Display the SVG
            with open(stat['svg_path'], 'r', encoding='utf-8') as file:
                svg_data = file.read()
            
            left_column.image(svg_data)

        else:
            left_column.write(stat['svg_path'])
        

            # Display metrics
        st1, st2, st3, st4, st5, st6 = right_column.columns(6)
        
        st1.metric('Mean case duration', str(stat["Mean case duration"]) + " " + stat["Unit"])
        st2.metric('Median cycle time', str(stat["Median cycle time"]) + " " + stat["Unitmedian"])
        st3.metric('Events', str(stat["Events"]))
        st4.metric('Traces', str(stat["Traces"]))
        st5.metric('Activities', str(stat["Activities"]))
        st6.metric('Variants', str(stat["Variants"]))
