import streamlit as st
import numpy as np
import pandas as pd
import pm4py
import copy
import deprecation
import random
import os
import recommendations
from collections import OrderedDict
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

# positions=st.session_state.positions
# positions_edges=st.session_state.positions_edges

# def prueba():

asignaciones = {}


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
    dic={}

    for key, df in dfs.items():
        if metric in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions', 'Frequency']:

            dfg, sa, ea = pm4py.discover_dfg(df, activity_key=nodes) 
            grafo = defineGraphFrequency(df, dfg, nodes, metric)
            dic[key] = {'dfg': dfg, 'sa': sa, 'ea': ea, 'df': df, 'graph':grafo}

            
        else:
            dfg, sa, ea = pm4py.discovery.discover_performance_dfg(df, activity_key=nodes)
            
            grafo = defineGraphPerformance(df, dfg, nodes, metric)
            
            dic[key] = {'dfg': dfg, 'sa': sa, 'ea': ea, 'df': df,'graph':grafo} 
      
           

   
    # st.write(dic.items())
    # prueba(set_dfgs)

    return dic

def defineGraphFrequency(df, dfg, nodes, metric): 
    lista_nodos = list(df[nodes].unique())
    G = nx.DiGraph()
    dic_paths = {}
    dic_max = {}
    dic_nodes = {}

    # print(metric)

    if(metric == 'Absolute Frequency'):
        abs_freq_nodes = df[nodes].value_counts().to_dict()
    elif(metric == 'Case Frequency'):
        case_freq_nodes = df.groupby(nodes).apply(lambda x: len(x['case:concept:name'].unique())).to_dict() 
    elif(metric == 'Max Repetitions'):
        max_repetitions_nodes = df.groupby(nodes).apply(lambda x: x['case:concept:name'].value_counts().max()).to_dict()
    elif(metric=='Total Repetitions'):
        sum_repetitions_nodes = df.groupby(nodes).apply(lambda x: (x['case:concept:name'].value_counts() > 1).sum())

    
    # max_rep, case_freq, total_repetitions = returnEdgesInfo(df,nodes,'case:concept:name','time:timestamp')
    if(metric in ['Case Frequency', 'Max Repetitions', 'Total Repetitions']):
        res = returnEdgesInfo(df,nodes,'case:concept:name','time:timestamp', metric)

    for key in dfg.keys():  
        dic_paths[key]={}
        # for metric in metrics:
        if(metric == 'Absolute Frequency'):
            dic_paths[key]['abs_freq'] = dfg[key]
        elif(metric == 'Case Frequency'): dic_paths[key]['case_freq'] = res[key]
        elif(metric == 'Max Repetitions'):
            dic_paths[key]['max_repetitions'] = res[key]
        elif(metric=='Total Repetitions'):
            dic_paths[key]['total_repetitions'] = res[key]
    

    for key in lista_nodos:
        dic_nodes[key]={}
        # for metric in metrics:
        if(metric == 'Absolute Frequency'):
            dic_nodes[key]['abs_freq'] = abs_freq_nodes[key]
        elif(metric == 'Case Frequency'):
            dic_nodes[key]['case_freq'] = case_freq_nodes[key]
        elif(metric == 'Max Repetitions'):
            dic_nodes[key]['max_repetitions'] = max_repetitions_nodes[key]
        elif(metric == 'Total Repetitions'):
            dic_nodes[key]['total_repetitions'] = sum_repetitions_nodes[key]
    
    for nodo, propiedades in dic_nodes.items():
        G.add_node(nodo, **propiedades)
    
    for edge, propiedades in dic_paths.items():
        actividad_origen, actividad_destino = edge
        G.add_edge(actividad_origen,actividad_destino, **propiedades)  
        
    # st.write(G.edges.data())
    return G
    
def defineGraphPerformance(df, dfg, nodes, metric):  

    G = nx.DiGraph()

    lista_nodos = list(df[nodes].unique())

    dic_paths = {}
    dic_nodes = {}

    for key in dfg.keys():
        
        if(metric=='Mean Cycle Time'):
            dic_paths[key] = {'mean':dfg[key]['mean']}
        elif(metric=='Median Cycle Time'):
            dic_paths[key] = {'median':dfg[key]['median']}
        elif(metric=='StDev Cycle Time'):
            dic_paths[key] = {'stdev':dfg[key]['stdev']}
        elif(metric=='Total Cycle Time'):
            dic_paths[key] = {'total':dfg[key]['sum']}

    for nodo in lista_nodos:
        G.add_node(nodo)

    for edge, propiedades in dic_paths.items():        
        actividad_origen, actividad_destino = edge
        G.add_edge(actividad_origen,actividad_destino, **propiedades)  
    case_durations = pm4py.get_all_case_durations(df, activity_key='concept:name', 
                 case_id_key='case:concept:name', timestamp_key='time:timestamp')
    
    if(metric=='Mean Cycle Time'):       
        avg_duration=np.mean(case_durations)
        G.graph['meanCTWholeProcess'] = avg_duration
    elif(metric=='Median Cycle Time'):
        avg_duration=np.median(case_durations)
        G.graph['medianCTWholeProcess'] = avg_duration   

    # st.write(G.edges.data())
    return G

def apply_custom(key, dfg, start_activities, end_activities, log, parameters, activities_count, soj_time, metric, tupla, delete_act):

    # st.write(activities_count)
        
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
    # if not positions:
    #     # st.write('no positions')
    #     return graphviz_visualization(activities_count, dfg, image_format=image_format, measure="frequency",
    #                               max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
    #                               start_activities=start_activities, end_activities=end_activities, 
    #                               soj_time=soj_time, font_size=font_size, bgcolor=bgcolor, stat_locale=stat_locale)
    # else:
        # st.write('si positions')
    if(tupla[0] == 'Existence of activities'):
        return graphviz_visualization_existence_act(delete_act, tupla, activities_count, dfg, image_format=image_format, measure=metric,
                                  max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
                                  start_activities=start_activities, end_activities=end_activities, 
                                  soj_time=soj_time, font_size=font_size, bgcolor=bgcolor, stat_locale=stat_locale)
    elif(tupla[0]=='Stable parts'):
        return graphviz_visualization_stable_parts(key, tupla, delete_act, activities_count, dfg, image_format=image_format, measure=metric,
                                  max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
                                  start_activities=start_activities, end_activities=end_activities, 
                                  soj_time=soj_time, font_size=font_size, bgcolor=bgcolor, stat_locale=stat_locale)
    else:
        return graphviz_visualization3(delete_act, activities_count, dfg, image_format=image_format, measure=metric,
                                  max_no_of_edges_in_diagram=max_no_of_edges_in_diagram,
                                  start_activities=start_activities, end_activities=end_activities, 
                                  soj_time=soj_time, font_size=font_size, bgcolor=bgcolor, stat_locale=stat_locale,)

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

    try:
        min_value, max_value = get_min_max_value(activities_count)

        for ac in activities_count:
            v0 = activities_count[ac]
            """transBaseColor = int(
                255 - 100 * (v0 - min_value) / (max_value - min_value + 0.00001))
            transBaseColorHex = str(hex(transBaseColor))[2:].upper()
            v1 = "#" + transBaseColorHex + transBaseColorHex + "FF"""

            v1 = get_trans_freq_color(v0, min_value, max_value)

            activities_color[ac] = v1
    except:
        for ac in activities_count:
            activities_color[ac] = 'white'

    return activities_color

def graphviz_visualization2(activities_count, dfg, image_format="png", measure="performance",
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
    
    positions=st.session_state.positions
    positions_edges=st.session_state.positions_edges

    # st.write('si2')
    
    if start_activities is None:
        start_activities = {}
    if end_activities is None:
        end_activities = {}
    if stat_locale is None:
        stat_locale = {}

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph("", filename=filename.name, engine='neato', graph_attr={'bgcolor': bgcolor})
    
    
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
    viz.attr('node', shape='box', overlap='false')

    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities_in_dfg)))

    activities_map = {}
    #     x=0
    #     y=1
    # viz.attr(overlap='false',splines="true")

    for act in activities_to_include:
    #            position = f"{x},{y}!"
        if(act in positions.keys()):
            position_tupla=positions[act]
            x=position_tupla[0]
            y=position_tupla[1]
            position=f"{x},{y}"
            # print(act,position)
            if "frequency" in measure and act in activities_count_int:
    #                 print(act)
                # viz.node(str(hash(act)), act + " (" + str(activities_count_int[act]) + ")",color='#f0512f')
                        #  fillcolor=activities_color[act], fontsize=font_size, pos=position, pin="true")
                viz.node(str(hash(act)), act + " (" + str(activities_count_int[act]) + ")", style='filled',
                         fillcolor=activities_color[act], fontsize=font_size, pos=position, pin="true")
                activities_map[act] = str(hash(act))
            else:
                stat_string = human_readable_stat(soj_time[act], stat_locale)
                viz.node(str(hash(act)), act + f" ({stat_string})", fontsize=font_size, pos=position)
                activities_map[act] = str(hash(act))
        else:
            if "frequency" in measure and act in activities_count_int:
    #                 print(act)
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
    viz.attr(overlap='false',splines="true")

    
    # represent edges
    for edge in dfg_edges:
        if(edge in positions_edges.keys()):
            info_edge=positions_edges[edge]
            source_pos = info_edge['source_pos']
            target_pos = info_edge['target_pos']
            intermediate_points = info_edge['intermediate_points']
            spline = f"s,{source_pos[0]:.6f},{source_pos[1]:.6f} "
            if intermediate_points:
                spline += f"point {intermediate_points[0][0]:.6f},{intermediate_points[0][1]:.6f} "
                for point in intermediate_points[1:]:
                    spline += f"point {point[0]:.6f},{point[1]:.6f} "
            spline += f"e,{target_pos[0]:.6f},{target_pos[1]:.6f}"
            # st.write(spline)

            # origen=position_tupla[0]
            # destino=position_tupla[1]
            # position=f"{1.5},{1.5}"
            # st.write(position)

            if "frequency" in measure:
                label = str(dfg[edge])
            else:
                label = human_readable_stat(dfg[edge], stat_locale)

            

            viz.edge(str(hash(edge[0])), str(hash(edge[1])), xlabel=label, penwidth=str(penwidth[edge]), fontsize=font_size,
                        pos=spline, constraint="false")
    #                  xlabel="10", fontcolor="blue", tailport="s", headport="n", decorate="true")
    #                 labeldistance="4", labelloc="t", fontcolor="red")
    # viz.edge('A','B', pos=spline, constraint="false", color='blue')
    viz.attr(overlap='false',splines="true")
    # viz.edge('A', 'P', pos="s,7,2 2,3 3,1 e,4,5", color='#f0512f')

    start_activities_to_include = [act for act in start_activities if act in activities_map]
    end_activities_to_include = [act for act in end_activities if act in activities_map]

 
    if start_activities_to_include:
        position_tupla=positions["<&#9679;> solid circle black lightgrey"]
        x=position_tupla[0]
        y=position_tupla[1]
        position=f"{x},{y}!"
        viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10", pos=position)
        for act in start_activities_to_include:
            
            label = str(start_activities[act]) if isinstance(start_activities, dict) else ""
            viz.edge("@@startnode", activities_map[act], xlabel=label, fontsize=font_size, color="#90EE90", 
                     style="dashed")

    if end_activities_to_include:
        position_tupla=positions["<&#9632;> solid doublecircle black lightgrey"]
        x=position_tupla[0]
        y=position_tupla[1]
        position=f"{x},{y}!"
        viz.node("@@endnode", "<&#9632;>", shape='doublecircle', fontsize="10", pos=position)
        for act in end_activities_to_include:
            label = str(end_activities[act]) if isinstance(end_activities, dict) else ""
            viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", 
                     style="dashed", xlp="3.5,2!")
        
    else:
        
        if start_activities_to_include:
            viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            for act in start_activities_to_include:
                label = str(start_activities[act]) if isinstance(start_activities, dict) else ""
                viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="dashed")

        if end_activities_to_include:
            viz.node("@@endnode", "<&#9632;>", shape='doublecircle', fontsize="10")
            for act in end_activities_to_include:
                label = str(end_activities[act]) if isinstance(end_activities, dict) else ""
                viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="dashed")

    viz.attr(overlap='false',splines="true")
    #     viz.attr(overlap="true", splines="false")


    viz.format = image_format



    return viz

def graphviz_visualization3(delete_act, activities_count, dfg, image_format="png", measure="performance",
                           max_no_of_edges_in_diagram=100000, start_activities=None, end_activities=None, soj_time=None,
                            font_size="12", bgcolor=constants.DEFAULT_BGCOLOR, stat_locale=None):

    

    if start_activities is None:
        start_activities = {}
    if end_activities is None:
        end_activities = {}
    if stat_locale is None:
        stat_locale = {}

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    
    viz = st.session_state.viz
    ea = st.session_state.ea
    sa = st.session_state.sa
    
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
    # if()
    activities_color = get_activities_color(activities_count_int)

    # represent nodes
    viz.attr(overlap='false')

    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities_in_dfg)))


    # st.write(activities_in_dfg)
    # st.write(activities_to_include)
    activities_map = {}
    activities_map2={}
    #     x=0
    #     y=1

    nodes_viz, edges_viz = obtener_nodos_y_aristas(viz)

    mapeo = st.session_state.mapeo


    # st.write(activities_in_dfg)
    # st.write(nodes_viz)

    for act in nodes_viz:
        act = mapeo[act]
        # st.write(activities_to_include)
        if(act in activities_in_dfg):
            # st.write('si', act)
            if(activities_count_int[act]==None):
                value = 0
            else:
                value = activities_count_int[act]
            viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
                     fontcolor='black',fillcolor=activities_color[act], fontsize=font_size, shape='box', color='black')
            activities_map[act] = str(hash(act))
        else:
            # st.write('no', act)
            # viz.node(str(hash(act)),  label='', color='#FFFFFF', shape='plaintext', style='', color='#f0512f')
            viz.node(str(hash(act)), fontcolor='#FFFFFF', style='invis', fillcolor='#FFFFFF', fontsize=font_size, color='#FFFFFF')
            activities_map[act] = str(hash(act))


    dfg_edges = sorted(list(dfg.keys()))

    # viz.attr(overlap='false',splines="true")
    # st.write(dfg_edges)

    for edge in edges_viz:
        if edge in dfg_edges:
            if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                label = str(dfg[edge])
            else:
                label = human_readable_stat(dfg[edge], stat_locale)
        
            viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
            color='black', fontcolor='black', style='')
        else:
            viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
            color='#FFFFFF', fontcolor='#FFFFFF', style='invis')


    # st.write(activities_map)
    start_activities_to_include = [act for act in start_activities if act in activities_map]
    end_activities_to_include = [act for act in end_activities if act in activities_map]

    # st.write(activities_map)
    # st.write(activities_map2)
    if start_activities_to_include:
        for act in sa:
        # viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            if act in start_activities_to_include:
                if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                    label = str(start_activities[act]) if isinstance(start_activities, dict) else ""
                else:
                    label = ""
                viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="dashed")
            else:
                viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')

    if end_activities_to_include:
        # st.write(end_activities_to_include)
        for act in ea:
        # <&#9632;>
        # viz.node("@@endnode", "<&#9632;>", shape='doublecircle', fontsize="10")
            if act in end_activities_to_include:
                if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                    label = str(end_activities[act]) if isinstance(end_activities, dict) else ""
                else:
                    label=""
                viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="dashed")
            else:
                viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')

    # viz.attr(overlap='false',splines="true")


    return viz, ''

def asignar_colores(actividades):
    colores_llamativos = [
        "orange", "yellow", "pink", "purple", "green", "red",
        "#00FFFF", "#0080FF", "#0000FF", "#8000FF", "#FF00FF", "#FF0080"
    ]
    
    # random.shuffle(colores_llamativos)
    asignacion = {}
    
    # st.write(actividades)
    for i, actividad in enumerate(actividades):
        asignacion[actividad] = colores_llamativos[i % len(colores_llamativos)]
    
    return asignacion



def graphviz_visualization_existence_act(delete_act, tupla1, activities_count, dfg, image_format="png", measure="performance",
                           max_no_of_edges_in_diagram=100000, start_activities=None, end_activities=None, soj_time=None,
                            font_size="12", bgcolor=constants.DEFAULT_BGCOLOR, stat_locale=None):
    
    mode = tupla1[1][0]
    values_act = tupla1[1][1]
    color_mode = tupla1[2]
    order_list='false'
    if start_activities is None:
        start_activities = {}
    if end_activities is None:
        end_activities = {}
    if stat_locale is None:
        stat_locale = {}

    # st.write(values_act)
    # asignaciones = asignar_colores(values_act)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    
    viz = st.session_state.viz
    ea = st.session_state.ea
    sa = st.session_state.sa
    
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
    viz.attr(overlap='false')

    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        # take unique elements as a list not as a set (in this way, nodes are added in the same order to the graph)
        activities_to_include = sorted(list(set(activities_in_dfg)))

    
    activities_to_include = [x for x in activities_to_include if x not in delete_act]
    # st.write(activities_in_dfg)
    # st.write(activities_to_include)
    activities_map = {}
    activities_map2={}


    nodes_viz, edges_viz = obtener_nodos_y_aristas(viz)
    # st.write(delete_act)
    # st.write(nodes_viz)
    # nodes_viz2 = st.session_state.nodesDFG
    # st.write(nodes_viz2)
    # merged_list = [activity for activity in nodes_viz2 if activity in nodes_viz]
    # st.write(merged_list)
    # edges_viz2 = st.session_state.edgesDFG
    # merged_list_edges = [edge for edge in edges_viz if edge in edges_viz2]

    mapeo = st.session_state.mapeo
    colores = st.session_state.colores


    if ((mode == 'All included' and set(values_act).issubset(activities_to_include)) or (mode == 'Some included')):
       
        for act in nodes_viz:
            # act = mapeo[act]
            if(act in activities_to_include):
                if(activities_count_int[act]==None):
                    value = 0
                else:
                    value = activities_count_int[act]
                if act in values_act:
                    order_list = 'true'
  
                    if(color_mode=='Same color'):
                        fillcolor='orange'
                    else:
                        fillcolor=colores[act]
                else:
                    fillcolor=activities_color[act]
                viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
                            fontcolor='black',fillcolor=fillcolor, fontsize=font_size, shape='box', color='black')
                activities_map[act] = str(hash(act))
            else:
                    # viz.node(str(hash(act)),  label='', color='#FFFFFF', shape='plaintext', style='', color='#f0512f')
                viz.node(str(hash(act)), fontcolor='#FFFFFF', style='invis', fillcolor='#FFFFFF', fontsize=font_size, color='#FFFFFF')
                activities_map[act] = str(hash(act))
    else:

        for act in nodes_viz:
            # act = mapeo[act]
            if(act in activities_to_include):
                if(activities_count_int[act]==None):
                    value = 0
                else:
                    value = activities_count_int[act]
                viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
                        fontcolor='black',fillcolor=activities_color[act], fontsize=font_size, shape='box', color='black')
                activities_map[act] = str(hash(act))
            else:
                # viz.node(str(hash(act)),  label='', color='#FFFFFF', shape='plaintext', style='', color='#f0512f')
                viz.node(str(hash(act)), fontcolor='#FFFFFF', style='invis', fillcolor='#FFFFFF', fontsize=font_size, color='#FFFFFF')
                activities_map[act] = str(hash(act))



    dfg_edges = sorted(list(dfg.keys()))

    # common_edges = set(dfg_edges).intersection(edges_viz)  
    # edges_subset = set(dfg_edges).difference(edges_viz)    
    # edges_todas = edges_viz.difference(set(dfg_edges))

    # for edge in common_edges:
    #     if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #         label = str(dfg[edge])
    #     else:
    #         label = human_readable_stat(dfg[edge], stat_locale)
        
    #     viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #     color='black', fontcolor='black', style='')

    # for edge in edges_todas:
    #     viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #     color='#FFFFFF', fontcolor='#FFFFFF', style='invis')
    
    # for edge in edges_subset:
    #     viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #     color='#f0512f', fontcolor='black', style='')




    for edge in edges_viz:
        if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
            if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                label = str(dfg[edge])
            else:
                label = human_readable_stat(dfg[edge], stat_locale)
        
            viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
            color='black', fontcolor='black', style='')
        else:

            viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
            color='#FFFFFF', fontcolor='#FFFFFF', style='invis')




    start_activities_to_include = [act for act in start_activities if act in activities_map]
    end_activities_to_include = [act for act in end_activities if act in activities_map]



    start_activities_to_include = [x for x in start_activities_to_include if x not in delete_act]
    end_activities_to_include = [x for x in end_activities_to_include if x not in delete_act]



    if start_activities_to_include:
        viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
        # node_sa = False
        for act in sa:
        # viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            if act in start_activities_to_include:
                node_sa=True
                if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                    label = str(start_activities[act]) if isinstance(start_activities, dict) else ""
                else:
                    label = ""
                viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="dashed")
            else:
                viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
        # if(node_sa==False): 
        #     viz.node("@@startnode", "<&#9679;>", style='invis')
    else:
        viz.node("@@startnode", "<&#9679;>", style='invis')
        for act in sa:
            viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')


    if end_activities_to_include:
        # st.write(end_activities_to_include)
        viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
        # node_ea = False
        for act in ea:
        # <&#9632;>
        # viz.node("@@endnode", "<&#9632;>", shape='doublecircle', fontsize="10")
            if act in end_activities_to_include:
                node_ea = True
                if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                    label = str(end_activities[act]) if isinstance(end_activities, dict) else ""
                else:
                    label=""
                viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="dashed")
            else:
                viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')
        # if(node_ea==False): 
        #     viz.node("@@endnode", "<&#9632;>", style='invis')
    else:
        viz.node("@@endnode", "<&#9632;>", style='invis')
        for act in ea:            
            viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')




    return viz, order_list


def graphviz_visualization_stable_parts(key, tupla, delete_act, activities_count, dfg, image_format="png", measure="performance",
                           max_no_of_edges_in_diagram=100000, start_activities=None, end_activities=None, soj_time=None,
                            font_size="12", bgcolor=constants.DEFAULT_BGCOLOR, stat_locale=None):
    
    # st.write(reference)
    # st.write(key)
    reference = tupla[1][0]
    mode = tupla[2]
    add = tupla[3]

    order_list='false'
    if start_activities is None:
        start_activities = {}
    if end_activities is None:
        end_activities = {}
    if stat_locale is None:
        stat_locale = {}


    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    
    viz = st.session_state.viz
    reference_nodes = st.session_state.reference_nodes
    reference_edges = st.session_state.reference_edges
    ea = st.session_state.ea
    sa = st.session_state.sa
    
    dfg_key_value_list = []
    for edge in dfg:
        dfg_key_value_list.append([edge, dfg[edge]])

    dfg_key_value_list = sorted(dfg_key_value_list, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
    dfg_key_value_list = dfg_key_value_list[0:min(len(dfg_key_value_list), max_no_of_edges_in_diagram)]
    dfg_allowed_keys = [x[0] for x in dfg_key_value_list]
    dfg_keys = list(dfg.keys())
    for edge in dfg_keys:
        if edge not in dfg_allowed_keys:
            del dfg[edge]

    penwidth = assign_penwidth_edges(dfg)
    activities_in_dfg = set()
    activities_count_int = copy(activities_count)

    for edge in dfg:
        activities_in_dfg.add(edge[0])
        activities_in_dfg.add(edge[1])

    activities_color = get_activities_color(activities_count_int)
    
    viz.attr(overlap='false')

    if len(activities_in_dfg) == 0:
        activities_to_include = sorted(list(set(activities_count_int)))
    else:
        activities_to_include = sorted(list(set(activities_in_dfg)))

    activities_to_include = [x for x in activities_to_include if x not in delete_act]

    activities_map = {}
    activities_map2={}


    nodes_viz, edges_viz = obtener_nodos_y_aristas(viz)

    # nodes_reference, edges_reference = obtener_nodos_y_aristas(reference_viz)


    mapeo = st.session_state.mapeo
    colores = st.session_state.colores

    gris_medio = 'grey'
    gris_claro = '#F5F5F5'
    gris_oscuro = '#808080'

    # Nodes


    if (reference=='Whole process' and mode!=[]):
        order_list=False
        for act in nodes_viz:
            if(act in activities_to_include):
                value = 0 if activities_count_int[act]==None else activities_count_int[act]
                text = act + " (" + str(value) + ")"
                node_shape='box'
                style = 'filled'
                if('Similarities' in mode):
                    fill_color='palegreen'
                    color_node = 'black'
                    font_color = 'black'
                elif('Differences reference model' in mode):
                    fill_color = gris_claro
                    color_node= gris_medio
                    font_color = gris_oscuro
                else:
                    fill_color=activities_color[act]
                    color_node='black'
            else:
                text = act
                # node_shape= 'box'
                node_shape='plaintext'
                style = 'filled'
                fill_color='white'
                if('Differences reference model' in mode):
                    font_color = '#f0512f'
                    color_node='#f0512f'
                else:
                    font_color = '#808080'
                    color_node='#808080'
            viz.node(str(hash(act)), text, style=style, fontcolor=font_color, fillcolor=fill_color, 
                     fontsize=font_size, shape=node_shape, color=color_node)
            
            activities_map[act] = str(hash(act))
        st.session_state.reference_nodes = activities_to_include
    
    elif(reference=='Whole process' and mode==[]):
        order_list = 'false'
        for act in nodes_viz:
            if(act in activities_to_include):
                value = 0 if activities_count_int[act]==None else activities_count_int[act]
                fillcolor=activities_color[act]
                viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
                        fontcolor='black',fillcolor=fillcolor, fontsize=font_size, shape='box', color='black')
                activities_map[act] = str(hash(act))
            else:
                viz.node(str(hash(act)), act, fontcolor='#808080', fillcolor='white', shape='plaintext',fontsize=font_size)
                activities_map[act] = str(hash(act))
        st.session_state.reference_nodes = activities_to_include
    
    elif(key==reference):
        order_list = 'true'
        for act in nodes_viz:
            if(act in activities_to_include):
                value = 0 if activities_count_int[act]==None else activities_count_int[act]
                fillcolor=activities_color[act]

                viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
                        fontcolor='black',fillcolor=fillcolor, fontsize=font_size, shape='box', color='black')
                activities_map[act] = str(hash(act))
            else:
                viz.node(str(hash(act)), fontcolor='#FFFFFF', style='invis', fillcolor='#FFFFFF', fontsize=font_size, color='#FFFFFF')
                activities_map[act] = str(hash(act))
        st.session_state.reference_nodes = activities_to_include
    

    else:
        for act in nodes_viz:
            if(act in activities_to_include):
                value = 0 if activities_count_int[act]==None else activities_count_int[act]
                text = act + " (" + str(value) + ")"
                node_shape='box'
                style = 'filled'
                if('Similarities' in mode and 'Differences DFG' in mode):
                    color_node = 'black'
                    font_color = 'black'
                    if(act in reference_nodes):
                        fill_color='palegreen'
                    else:
                        fill_color='#f66a4c'
                elif('Similarities' in mode):
                    if(act in reference_nodes):
                        color_node = 'black'
                        font_color = 'black'
                        fill_color='palegreen'
                    else:
                        color_node = gris_medio
                        font_color = gris_oscuro
                        fill_color=  gris_claro
                elif('Differences DFG' in mode):
                    if(act not in reference_nodes):
                        color_node = 'black'
                        font_color = 'black'
                        fill_color='#f0512f'
                    else:
                        color_node = gris_medio
                        font_color = gris_oscuro
                        fill_color=  gris_claro
                elif('Differences reference model' in mode):
                    color_node = gris_medio
                    font_color = gris_oscuro
                    fill_color=  gris_claro 
                else:
                    color_node = 'black'
                    font_color = 'black'
                    fill_color=  activities_color[act]    

            elif(act in reference_nodes):
                text = act
                node_shape='plaintext'
                style = 'filled'
                fill_color='white'

                if('Differences reference model' in mode):
                    font_color = '#f0512f'
                    color_node='#f0512f'
                else:
                    font_color = '#808080'
                    color_node='#808080'
            else:
                if(add==False):
                    text = ''
                    node_shape='plaintext'
                    font_color='#FFFFFF'
                    style='invis'
                    fill_color='#FFFFFF'
                    color_node='#FFFFFF'
                else:
                    text = act
                    font_color=gris_oscuro
                    node_shape='plaintext'
                    style = 'filled'
                    fill_color='white'
                    color_node='#FFFFFF'

            viz.node(str(hash(act)), text, style=style, fontcolor=font_color, fillcolor=fill_color, 
                     fontsize=font_size, shape=node_shape, color=color_node)
            
            activities_map[act] = str(hash(act))




    
  





    # if(reference=='Whole process' and mode=='Same color'):
    #     order_list = 'false'
    #     for act in nodes_viz:
    #         if(act in activities_to_include):
    #             value = 0 if activities_count_int[act]==None else activities_count_int[act]

    #             viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor='orange', fontsize=font_size, shape='box', color='black')
    #             activities_map[act] = str(hash(act))
    #         else:
    #             viz.node(str(hash(act)), fontcolor='#808080',fillcolor='#F5F5F5', fontsize=font_size, color='grey')
    #             activities_map[act] = str(hash(act))
    #     st.session_state.reference_nodes = activities_to_include

    # elif(reference=='Whole process' and mode !='Same color'):
    #     order_list = 'false'
    #     for act in nodes_viz:
    #         if(act in activities_to_include):
    #             value = 0 if activities_count_int[act]==None else activities_count_int[act]
    #             fillcolor=activities_color[act]
    #             viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor=fillcolor, fontsize=font_size, shape='box', color='black')
    #             activities_map[act] = str(hash(act))
    #         else:
    #             viz.node(str(hash(act)), fontcolor='#808080',fillcolor='#F5F5F5', fontsize=font_size, color='grey')
    #             activities_map[act] = str(hash(act))
    #     st.session_state.reference_nodes = activities_to_include

    # elif(key == reference and mode == 'Similarities and differences'):
    #     order_list = 'true'
    #     for act in nodes_viz:
    #         if(act in activities_to_include):
    #             value = 0 if activities_count_int[act]==None else activities_count_int[act]
    #             viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor='#FFD700', fontsize=font_size, shape='box', color='black')
    #             activities_map[act] = str(hash(act))
    #         else:
    #             viz.node(str(hash(act)), fontcolor='#808080',fillcolor='#F5F5F5', fontsize=font_size, color='grey')
    #             activities_map[act] = str(hash(act))
    #     st.session_state.reference_nodes = activities_to_include
    # elif(mode=='Similarities and differences' and key!=reference):
    #     for act in nodes_viz:
    #         activities_map[act] = str(hash(act))
    #         if(act in activities_to_include and act in reference_nodes):
    #             value = 0 if activities_count_int[act]==None else activities_count_int[act]
    #             viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor='#FF8C00', fontsize=font_size, shape='box', color='black')
                
    #         elif(act in activities_to_include):
    #             value = 0 if activities_count_int[act]==None else activities_count_int[act]
    #             viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor='#E05266', fontsize=font_size, shape='box', color='black')

    #         elif(act in reference_nodes):
    #             viz.node(str(hash(act)), act , style='filled',
    #                     fontcolor='black',fillcolor='#FFD700', fontsize=font_size, shape='box', color='black')
    #         else:
    #             viz.node(str(hash(act)), fontcolor='#808080',fillcolor='#F5F5F5', fontsize=font_size, color='grey')
                
    # elif(key == reference):
    #     order_list = 'true'
    #     for act in nodes_viz:
    #         if(act in activities_to_include):
    #             value = 0 if activities_count_int[act]==None else activities_count_int[act]
    #             fillcolor=activities_color[act]

    #             viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor=fillcolor, fontsize=font_size, shape='box', color='black')
    #             activities_map[act] = str(hash(act))
    #         else:
    #             viz.node(str(hash(act)), fontcolor='#FFFFFF', style='invis', fillcolor='#FFFFFF', fontsize=font_size, color='#FFFFFF')
    #             activities_map[act] = str(hash(act))
    #     st.session_state.reference_nodes = activities_to_include
    # else:
    #     # st.write(key, activities_to_include, reference_nodes)
    #     order_list = 'false'
    #     for act in nodes_viz:
    #         if(act in activities_to_include):
    #             value = 0 if activities_count_int.get(act) is None else activities_count_int[act]

    #             # aqui debo comprobar si el nodo pertenece a la parte estable (comun a key)
    #             if (act in reference_nodes and mode == 'Similarities') or (act not in reference_nodes and mode == 'Differences'):
    #                 viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='black',fillcolor='orange', fontsize=font_size, shape='box', color='black')
    #             else:
    #                 viz.node(str(hash(act)), act + " (" + str(value) + ")", style='filled',
    #                     fontcolor='#808080',fillcolor='#F5F5F5', fontsize=font_size, shape='box', color='grey')
    #             activities_map[act] = str(hash(act))
    #         else:
    #             viz.node(str(hash(act)), fontcolor='#FFFFFF', style='invis', fillcolor='#FFFFFF', fontsize=font_size, color='#FFFFFF')
    #             activities_map[act] = str(hash(act))




    # Edges

    dfg_edges = sorted(list(dfg.keys()))

    if (reference=='Whole process' and mode!=[]):
        for edge in edges_viz:
            if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
                label = str(dfg[edge]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else human_readable_stat(dfg[edge], stat_locale)
                width=str(penwidth[edge])
                style='solid'
                
                if('Similarities' in mode):
                    color_edge='green'
                    font_color = 'black'
                else:
                    color_edge= gris_medio
                    font_color=gris_medio
            else:
                label=''
                width=''
                style='dashed'
                font_color='white'
                if('Differences reference model' in mode):
                    color_edge = '#f0512f'
                else:
                    # color_edge = '#808080'
                    color_edge = gris_medio
            
            viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=width, fontsize=font_size,
                color=color_edge, fontcolor=font_color, style=style)
            
            st.session_state.reference_edges = dfg_edges

    elif(reference=='Whole process' and mode==[]):
        for edge in edges_viz:
            if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
                if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                    label = str(dfg[edge])
                else:
                    label = human_readable_stat(dfg[edge], stat_locale)
            
                viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
                color='black', fontcolor='black', style='')
            else:

                viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='',  penwidth='', fontsize=font_size, 
                color='grey', fontcolor=gris_medio, style='dashed')
        st.session_state.reference_edges = dfg_edges

    elif(reference==key):
        for edge in edges_viz:
            if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
                if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
                    label = str(dfg[edge])
                else:
                    label = human_readable_stat(dfg[edge], stat_locale)
            
                viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
                color='black', fontcolor='black', style='')
            else:

                viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
                color='#FFFFFF', fontcolor='#FFFFFF', style='invis')
        st.session_state.reference_edges = dfg_edges


    else:

        for edge in edges_viz:
            if (edge[0] not in delete_act and edge[1] not in delete_act):
                if(edge in dfg_edges):
                    label = str(dfg[edge]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else human_readable_stat(dfg[edge], stat_locale)
                    width=str(penwidth[edge])
                    style='solid'
                    font_color='black'
                    if('Similarities' in mode and 'Differences DFG' in mode):
                        if(edge in reference_edges):
                            color_edge = 'green'
                        else:
                            color_edge = '#f0512f'
                    elif('Similarities' in mode):
                        if(edge in reference_edges):
                            
                            color_edge = 'green'
                        else:
                            color_edge = gris_medio
                    elif('Differences DFG' in mode):
                        if(edge not in reference_edges):
                            color_edge = '#f0512f'
                        else:
                            color_edge = gris_medio
                    elif('Differences reference model' in mode):
                        color_edge=gris_medio
                    else:
                        color_edge='black'
                elif(edge in reference_edges):
                    label = ''
                    width= ''
                    style='dashed'
                    if('Differences reference model' in mode):
                        color_edge = '#f0512f'
                    else:
                        color_edge = gris_medio
                else:
                    if(add==False):
                        width= ''
                        color_edge='#FFFFFF'
                        font_color='#FFFFFF'
                        style='invis'
                        label = ''
                    else:
                        width= ''
                        color_edge=gris_medio
                        font_color='#FFFFFF'
                        style='dashed'
                        label = ''
            else:
                width= ''
                color_edge='#FFFFFF'
                font_color='#FFFFFF'
                style='invis'
                label = ''
                

                # viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
                # color='grey', fontcolor='grey')

            viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, fontsize=font_size, 
                color=color_edge, fontcolor=font_color, style=style, penwidth=width)





    # if(reference=='Whole process' and mode=='Same color'):
    #     for edge in edges_viz:
    #         if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #             color='orange', fontcolor='black', style='')
    #         else:

    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #             color='grey', fontcolor='grey')
    #     st.session_state.reference_edges = dfg_edges

    # elif(reference=='Whole process' and mode!='Same color'):
    #     for edge in edges_viz:
    #         if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #             color='black', fontcolor='black', style='')
    #         else:

    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #             color='grey', fontcolor='grey')
    #     st.session_state.reference_edges = dfg_edges
    # elif(key == reference and mode == 'Similarities and differences'):
    #     for edge in edges_viz:
    #         if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #             color='#FFD700', fontcolor='black', style='')
    #         else:
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #             color='grey', fontcolor='grey')
    #     st.session_state.reference_edges = dfg_edges

    # elif(key != reference and mode == 'Similarities and differences'):
    #     for edge in edges_viz:
    #         if ((edge in dfg_edges) and (edge in reference_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #             color='#FF8C00', fontcolor='black', style='')
    #         elif((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #             color='#E05266', fontcolor='black', style='')
    #         elif((edge in reference_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             label=''
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth='', fontsize=font_size,
    #             color='#FFD700', fontcolor='black', style='')
    #         else:
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #             color='grey', fontcolor='grey')        
    # elif(key == reference):
    #     for edge in edges_viz:
    #         if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
            
    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #             color='black', fontcolor='black', style='')
    #         else:

    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #             color='#FFFFFF', fontcolor='#FFFFFF', style='invis')
    #     st.session_state.reference_edges = dfg_edges
    # else:
    #     for edge in edges_viz:
    #         if ((edge in dfg_edges) and (edge[0] not in delete_act and edge[1] not in delete_act)):
    #             if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions']:
    #                 label = str(dfg[edge])
    #             else:
    #                 label = human_readable_stat(dfg[edge], stat_locale)
    #             if (edge in reference_edges and mode == 'Similarities') or (edge not in reference_edges and mode == 'Differences'):
    #                 viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #                 color='orange', fontcolor='black', style='')
    #             else:
    #                 viz.edge(str(hash(edge[0])), str(hash(edge[1])), label=label, penwidth=str(penwidth[edge]), fontsize=font_size,
    #                 color='grey', fontcolor='grey', style='')
    #         else:

    #             viz.edge(str(hash(edge[0])), str(hash(edge[1])), label='', fontsize=font_size, 
    #             color='#FFFFFF', fontcolor='#FFFFFF', style='invis')


    start_activities_to_include = [act for act in start_activities if act in activities_map]
    end_activities_to_include = [act for act in end_activities if act in activities_map]



    start_activities_to_include = [x for x in start_activities_to_include if x not in delete_act]
    end_activities_to_include = [x for x in end_activities_to_include if x not in delete_act]

    reference_ea = st.session_state.reference_ea
    reference_sa = st.session_state.reference_sa



    # Startpoints

    if(reference=='Whole process' and mode!=[]):
        if start_activities_to_include:
            viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            for act in sa:
                if act in start_activities_to_include:
                    node_sa=True
                    label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    style = 'solid'
                    if('Similarities' in mode):
                        color_edge = 'green'
                    else:
                        color_edge = gris_medio
                else:
                    label = ''
                    style='dashed'
                    if('Differences reference model' in mode):
                        color_edge = '#f0512f'
                    else:
                        color_edge = gris_medio
                viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color=color_edge, style=style)
        else:
            viz.node("@@startnode", "<&#9679;>", style='invis')
            for act in sa:
                viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')

    elif(reference=='Whole process' and mode==[]):
        if start_activities_to_include:
            viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            for act in sa:
                if act in start_activities_to_include:
                    node_sa=True
                    label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="solid")
                else:
                    viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="grey", style='dashed')
        else:
            viz.node("@@startnode", "<&#9679;>", style='invis')
            for act in sa:
                viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
    
    elif(reference==key):
        if start_activities_to_include:
            viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            for act in sa:
                if act in start_activities_to_include:
                    node_sa=True
                    label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="solid")
                else:
                    viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
        else:
            viz.node("@@startnode", "<&#9679;>", style='invis')
            for act in sa:
                viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')

        st.session_state.reference_sa = start_activities_to_include

    else:
        if start_activities_to_include:
            viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
            for act in sa:
                if act in start_activities_to_include:
                    label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    font_color='black'
                    style='solid'
                    if('Similarities' in mode and 'Differences DFG' in mode):
                        if(act in reference_sa):
                            color_edge = 'green'
                        else:
                            color_edge='#f0512f'
                    elif('Similarities' in mode):
                        if(act in reference_sa):
                            color_edge = 'green'
                        else:
                            color_edge=gris_medio
                    elif('Differences DFG' in mode):
                        if(act not in reference_sa):
                            color_edge = '#f0512f'
                        else:
                            color_edge=gris_medio
                    elif('Differences reference model' in mode):
                        color_edge = gris_medio
                    else:
                        color_edge='#90EE90'
                        
                else:
                    label=''
                    if('Differences reference model' in mode and act in reference_ea):
                        color_edge='green'
                        style='dashed'
                    else:
                        if(add==False):
                            color_edge=gris_medio
                            style='invis'
                        else:
                            color_edge = gris_medio
                            style = 'dashed'
                viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color=color_edge, style=style)
        else:
            viz.node("@@startnode", "<&#9679;>", style='invis')
            for act in sa:
                viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')


    
    # if(reference=='Whole process' and mode=='Same color'):
    #     if start_activities_to_include:
    #         viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
    #         for act in sa:
    #             if act in start_activities_to_include:
    #                 node_sa=True
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="orange", style="dashed")
    #             else:
    #                 viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@startnode", "<&#9679;>", style='invis')
    #         for act in sa:
    #             viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
    # elif(reference=='Whole process' and mode!='Same color'):
    #     if start_activities_to_include:
    #         viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
    #         for act in sa:
    #             if act in start_activities_to_include:
    #                 node_sa=True
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="dashed")
    #             else:
    #                 viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@startnode", "<&#9679;>", style='invis')
    #         for act in sa:
    #             viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
    # elif(key == reference and mode == 'Similarities and differences'):
    #     if start_activities_to_include:
    #         viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
    #         for act in sa:
    #             if act in start_activities_to_include:
    #                 node_sa=True
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#FFD700", style="dashed")
    #             else:
    #                 viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@startnode", "<&#9679;>", style='invis')
    #         for act in sa:
    #             viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
    #     st.session_state.reference_sa = start_activities_to_include

    # elif(key != reference and mode == 'Similarities and differences'):
    #     if start_activities_to_include:
    #         viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
    #         for act in sa:
    #             if act in start_activities_to_include and act in reference_sa:
    #                 node_sa=True
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#FF8C00", style="dashed")
    #             elif(act in start_activities_to_include):
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#E05266", style="dashed")
    #             elif(act in reference_sa):
    #                 viz.edge("@@startnode", activities_map[act], label='', fontsize=font_size, color="#FFD700", style="dashed")
    #             else:
    #                 viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@startnode", "<&#9679;>", style='invis')
    #         for act in sa:
    #             viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')

    # elif(key == reference):
    #     if start_activities_to_include:
    #         viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
    #         for act in sa:
    #             if act in start_activities_to_include:
    #                 node_sa=True
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="#90EE90", style="dashed")
    #             else:
    #                 viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
    #     else:
    #         viz.node("@@startnode", "<&#9679;>", style='invis')
    #         for act in sa:
    #             viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')

    #     st.session_state.reference_sa = start_activities_to_include

    # else:

    #     if start_activities_to_include:
    #         viz.node("@@startnode", "<&#9679;>", shape='circle', fontsize="10")
    #         for act in sa:
    #             if act in start_activities_to_include:
    #                 node_sa=True
    #                 label = str(start_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 if (act in reference_sa and mode == 'Similarities') or (act not in reference_sa and mode == 'Differences'):
    #                     viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="orange", style="dashed")
    #                 else:
    #                     viz.edge("@@startnode", activities_map[act], label=label, fontsize=font_size, color="grey", style="dashed")
    #             else:
    #                 viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')
    #     else:
    #         viz.node("@@startnode", "<&#9679;>", style='invis')
    #         for act in sa:
    #             viz.edge("@@startnode", activities_map[act], label="", fontsize=font_size, color="#FFFFFF", style='invis')






    # Endpoints

    if(reference=='Whole process' and mode!=[]):
        if end_activities_to_include:
            viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
            for act in ea:
                if act in end_activities_to_include:
                    node_sa=True
                    label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    style = 'solid'
                    if('Similarities' in mode):
                        color_edge = 'green'
                    else:
                        color_edge = gris_medio
                else:
                    label = ''
                    style='dashed'
                    if('Differences reference model' in mode):
                        color_edge = '#f0512f'
                    else:
                        color_edge = gris_medio
                viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color=color_edge, style=style)
        else:
            viz.node("@@endnode", "<&#9632;>", style='invis')
            for act in ea:            
                viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#808080")
    
    elif(reference=='Whole process' and mode==[]):
        if end_activities_to_include:
            viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
            for act in ea:
                if act in end_activities_to_include:
                    node_ea = True
                    label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="solid")
                else:
                    viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey", style='dashed')
        else:
            viz.node("@@endnode", "<&#9632;>", style='invis')
            for act in ea:            
                viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#808080")

    elif(reference==key):
        if end_activities_to_include:
            viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
            for act in ea:
                if act in end_activities_to_include:
                    node_ea = True
                    label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="solid")
                else:
                    viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')
        else:
            viz.node("@@endnode", "<&#9632;>", style='invis')
            for act in ea:            
                viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')

        st.session_state.reference_ea = end_activities_to_include
    
    else:

        if end_activities_to_include:
            viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
            for act in ea:
                if act in end_activities_to_include:
                    label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
                    font_color='black'
                    style='solid'
                    if('Similarities' in mode and 'Differences DFG' in mode):
                        if(act in reference_ea):
                            color_edge = 'green'
                        else:
                            color_edge='#f0512f'
                    elif('Similarities' in mode):
                        if(act in reference_ea):
                            color_edge = 'green'
                        else:
                            color_edge=gris_medio
                    elif('Differences DFG' in mode):
                        if(act not in reference_ea):
                            color_edge = '#f0512f'
                        else:
                            color_edge=gris_medio
                    elif('Differences reference model' in mode):
                        color_edge = gris_medio
                    else:
                        color_edge='#FF9999'
                        
                else:
                    label=''
                    if('Differences reference model' in mode and act in reference_ea):
                        color_edge='#f0512f'
                        style='dashed'
                    else:
                        if(add==False):
                            color_edge=gris_medio
                            style='invis'
                        else:
                            color_edge = gris_medio
                            style = 'dashed'
                viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color=color_edge, style=style)
        else:
            viz.node("@@endnode", "<&#9632;>", style='invis')
            for act in ea:
                viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')
        
    
    # if(reference=='Whole process' and mode=='Same color'):
    #     if end_activities_to_include:
    #         viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
    #         for act in ea:
    #             if act in end_activities_to_include:
    #                 node_ea = True
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="orange", style="dashed")
    #             else:
    #                 viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@endnode", "<&#9632;>", style='invis')
    #         for act in ea:            
    #             viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    # elif(reference=='Whole process' and mode!='Same color'):
    #     if end_activities_to_include:
    #         viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
    #         for act in ea:
    #             if act in end_activities_to_include:
    #                 node_ea = True
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="dashed")
    #             else:
    #                 viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@endnode", "<&#9632;>", style='invis')
    #         for act in ea:            
    #             viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    # elif(key == reference and mode == 'Similarities and differences'):
    #     if end_activities_to_include:
    #         viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
    #         for act in ea:
    #             if act in end_activities_to_include:
    #                 node_ea = True
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FFD700", style="dashed")
    #             else:
    #                 viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@endnode", "<&#9632;>", style='invis')
    #         for act in ea:            
    #             viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    #     st.session_state.reference_ea = end_activities_to_include
    # elif(key != reference and mode == 'Similarities and differences'):
    #     if end_activities_to_include:
    #         viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
    #         for act in ea:
    #             if act in end_activities_to_include and act in reference_ea:
    #                 node_ea = True
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF8C00", style="dashed")
    #             elif act in end_activities_to_include:
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#E05266", style="dashed")
    #             elif act in reference_ea: 
    #                 viz.edge(activities_map[act], "@@endnode", label='', fontsize=font_size, color="#FFD700", style="dashed")
    #             else:
    #                 viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    #     else:
    #         viz.node("@@endnode", "<&#9632;>", style='invis')
    #         for act in ea:            
    #             viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="grey")
    # elif(key == reference):
    #     if end_activities_to_include:
    #         viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
    #         for act in ea:
    #             if act in end_activities_to_include:
    #                 node_ea = True
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="#FF9999", style="dashed")
    #             else:
    #                 viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')
    #     else:
    #         viz.node("@@endnode", "<&#9632;>", style='invis')
    #         for act in ea:            
    #             viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')

    #     st.session_state.reference_ea = end_activities_to_include

    # else:
        
    #     if end_activities_to_include:
    #         viz.node("@@endnode", "<&#9632;>", shape='circle', fontsize="10")
    #         for act in ea:
    #             if act in end_activities_to_include:
    #                 node_ea = True
    #                 label = str(end_activities[act]) if measure in ['Absolute Frequency', 'Case Frequency', 'Max Repetitions', 'Total Repetitions'] else ""
    #                 if (act in reference_ea and mode == 'Similarities') or (act not in reference_ea and mode == 'Differences'):
    #                     viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="orange", style="dashed")
    #                 else:
    #                     viz.edge(activities_map[act], "@@endnode", label=label, fontsize=font_size, color="grey", style="dashed")
    #             else:
    #                 viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')
    #     else:
    #         viz.node("@@endnode", "<&#9632;>", style='invis')
    #         for act in ea:            
    #             viz.edge(activities_map[act], "@@endnode", label="", fontsize=font_size, color="#FFFFFF", style='invis')




    return viz, order_list



def obtener_nodos(viz):
    nodos = set()
    for linea in viz.body:
        # Ver si la línea define un nodo (busca "nombre [ atributo ]")
        match = re.match(r'^\s*([\w\d]+)\s+\[', linea)
        if match:
            nodos.add(match.group(1))  # Agregar el nodo encontrado
        else:
            # Buscar nodos dentro de aristas (ejemplo: A -> B)
            partes = re.findall(r'([\w\d]+)\s*->\s*([\w\d]+)', linea)
            for inicio, fin in partes:
                nodos.add(inicio)
                nodos.add(fin)
    return nodos

def obtener_nodos_y_aristas(viz):
    nodos = set()
    aristas = set()
    mapeo = st.session_state.mapeo

    for linea in viz.body:
        # Buscar nodos con atributos (ejemplo: -12345 [label="Nodo A"])
        match_nodo = re.match(r'^\s*([-]?\d+)\s+\[', linea)  # Permite números negativos
        if match_nodo:
            nodos.add(mapeo[match_nodo.group(1)])  # Agregar nodo encontrado

        # Buscar aristas (ejemplo: -12345 -> 67890)
        match_aristas = re.findall(r'([-]?\d+)\s*->\s*([-]?\d+)', linea)  # Permite negativos
        for origen, destino in match_aristas:
            nodos.add(mapeo[origen])
            nodos.add(mapeo[destino])
            aristas.add((mapeo[origen], mapeo[destino]))  # Guardar arista como tupla

    return nodos, aristas


def threshold(datos, metric, a, p, nodes, tupla, delete_act):


    if(tupla[0]=='Stable parts' and tupla[1][0]!='Whole process'):
        datos = OrderedDict([(tupla[1][0], datos[tupla[1][0]])] + [(k, v) for k, v in datos.items() if k != tupla[1][0]])
    
   
    dic={}
    stats_list = [] 
    ident = 0

    for key, dfg in datos.items():
        df = dfg['df']
        dfg_ini = dfg['dfg']
        translater={"Absolute Frequency":"abs_freq","Case Frequency":"case_freq",
                    "Max Repetitions":"max_repetitions", "Total Repetitions":
                    "total_repetitions", "Median Cycle Time":"median","Mean Cycle Time":"mean",
                    "StDev Cycle Time":"stdev","Total Cycle Time":"sum"}


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
                # st.write(G)
                
        G_nodes_filtered=removeEdges(G,list(dfg_path.keys()))
        G_edges_filtered=removeNodes(G_nodes_filtered,list(ac.keys()))

        measure=translater[metric]
                
        metric_nodes=dict(G.nodes.data(measure))
        # st.write(metric_nodes)
                
        list_edges=list(G.edges.data())
        dfg_custom={(edge[0],edge[1]):edge[2][measure] for edge in list_edges}

        gviz, order_list = apply_custom(key, dfg_custom,sa,ea,None,None,metric_nodes,None, metric, tupla, delete_act)
                
                 

        meandf, unitdf = statisticslog.mean_case(df)
        mediandf, unitmediandf = statisticslog.median_case(df)


        import copy
        if(order_list=='false'):
            stats_list.extend([{
                        "key": key,
                        "svg_path": copy.deepcopy(gviz),
                        "Mean case duration": meandf,
                        "Unit": unitdf,
                        "Median case duration": mediandf,
                        "Unitmedian": unitmediandf,
                        "Number of events": len(df),
                        "Number of traces": df['case:concept:name'].nunique(),
                        "Number of activities": df['concept:name'].nunique(),
                        "Number of variants": statisticslog.n_variants(df)
                    }])
        else:
            stats_list.insert(0, {
                        "key": key,
                        "svg_path": copy.deepcopy(gviz),
                        "Mean case duration": meandf,
                        "Unit": unitdf,
                        "Median case duration": mediandf,
                        "Unitmedian": unitmediandf,
                        "Number of events": len(df),
                        "Number of traces": df['case:concept:name'].nunique(),
                        "Number of activities": df['concept:name'].nunique(),
                        "Number of variants": statisticslog.n_variants(df)
                    })
        # st.write(stats_list)


        ident = ident + 1

    return stats_list

def show_DFGs(stats_list, order, metric):
    # st.table(stats_list)
    # Display sorted results
    if(order == 'By the search'):
        sorted_stats = stats_list
    else:
        sorted_stats = sorted(stats_list, key=lambda x: x[order], reverse=True)

    # st.table(sorted_stats)

    st.markdown(
    """
    <style>
        div[data-testid="metric-container"] {
            text-align: center; 
        }
        div[data-testid="metric-container"] > div[data-testid="stMetricValue"] {
            font-size: 14px !important;  /* Reducir tamaño del valor */
            font-weight: normal !important;
        }
    </style>
    """,
    unsafe_allow_html=True
    )

    for stat in sorted_stats:
        
        left_column, right_column = st.columns(2)

        # left_column.write(str(stat['key']))
        left_column.markdown(f" #### {str(stat['key'])} ")

        
        st1, st2, st3, st4, st5, st6 = left_column.columns(6)
        
        st1.metric('Mean case duration', str(stat["Mean case duration"]) + " " + stat["Unit"])
        st2.metric('Median case duration', str(stat["Median case duration"]) + " " + stat["Unitmedian"])
        st3.metric('Events', str(stat["Number of events"]))
        st4.metric('Traces', str(stat["Number of traces"]))
        st5.metric('Activities', str(stat["Number of activities"]))
        st6.metric('Variants', str(stat["Number of variants"]))

        st.write(stat['svg_path'])

        st.markdown("""---""")
