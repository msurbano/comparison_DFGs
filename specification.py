import streamlit as st
import numpy as np
import pandas as pd
import pm4py
import copy
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py.algo.filtering.dfg import dfg_filtering
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.statistics.rework.cases.log import get as cases_rework_get
from pm4py.statistics.start_activities.log.get import get_start_activities
from pm4py.statistics.end_activities.log.get import get_end_activities
import networkx as nx
import heapq
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
import datetime
import statisticslog
from pm4py.visualization.dfg.variants.frequency import apply
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from PIL import Image
from io import StringIO
from pm4py.visualization.dfg.variants.frequency import apply
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from collections import Counter
from itertools import combinations
import traceback

# st.set_page_config(page_title="Pattern specification")

maxt = 0

# st.markdown("# Pattern specification ❄️")
# st.markdown("# Pattern specification ❄️")

# st.write("DataFrame manipulado en la página 1:")


def search(expr, param, dic, inicial, measure): 
    return function(dic,expr,param, inicial, measure)
    
def function(graph, expr, paramF, inicial, measure):
    if(expr == 'percentageReworkActivityPerEvents'):
        return percentageReworkPerActivityEventsDFG(graph)
    elif(expr== 'percentageReworkPerActivity'):
        return percentageReworkPerActivityDFG(graph)

    elif(expr == 'Identify DFGs by the number of unique nodes'):
        return uniqueActivitiesDFG(graph, paramF)

    elif (expr == 'Identify DFGs by the number of unique resources'):
        return uniqueActivitiesDFG(graph, paramF)
    
    elif(expr == 'Identify infrequent activities'):
        return infreqact(graph, paramF, measure)
    
    elif(expr == 'Identify the most frequent activities'):
        return mostfreqact(graph, paramF, measure)

    elif(expr == 'Identify the most frequent fragment'):
        return mostfreqfrag(graph, param)

    elif(expr == 'Identify transitions with high duration'):
        return transduration(graph, paramF, measure)

    elif(expr == 'Identify activities with high duration'):
        # if(comprobar si es posible calcularlo):
            # return actduration(graph, paramF, measure)
        # else:
        st.markdown(" **Note: The duration associated with the transitions is used for this calculation.** ")
        return transduration(graph, paramF, measure)

    elif(expr == 'Identify transitions as bottlenecks'):
        return transbot(graph, paramF, inicial, measure)

    elif(expr == 'Identify activities as bottlenecks'):
        # if(comprobar si es posible calcularlo):
            # rreturn actbot(graph, paramF, inicial, measure)
        # else:
        st.markdown(" **Note: The duration associated with the transitions is used for this calculation.** ")
        return transbot(graph, paramF, inicial, measure)

    elif(expr == 'Identify resources with high workload'):
        return mostfreqresour(graph, paramF, measure)

    elif(expr == 'Identify resources as bottlenecks'):
        return resourbot(graph, paramF, inicial, measure)

    



def uniqueActivitiesDFG(dic, param):
    # st.write('n nodos de cada grafo', len(graph.nodes))
    prueba = {}
    prueba2 = {}

    min_nodos = min(len(datos['graph'].nodes) for datos in dic.values())
    max_nodos = max(len(datos['graph'].nodes) for datos in dic.values())

    for key, datos in dic.items():
        graph = datos['graph']

        if param == 'Minimum number of nodes':
            if len(graph.nodes) <= min_nodos:
                prueba[key] = datos
                prueba2[key] = len(graph.nodes)
                # st.markdown(f" **(Min. nodes: {min_nodos})**")
                # return prueba, min_nodos

        elif param == 'Maximum number of nodes':
            if len(graph.nodes) >= max_nodos:
                # st.markdown(f" **(Max. nodes: {max_nodos})**")
                prueba[key] = datos
                prueba2[key] = len(graph.nodes)
                # return prueba, max_nodos
        else:
            if(len(graph.nodes) >= param):
                prueba[key] = datos
                prueba2[key] = len(graph.nodes)
                # param = 'Otro'
                # return prueba, 0
    # return prueba
    # Devolver el diccionario filtrado y el valor correspondiente
    if param == 'Minimum number of nodes':
        st.markdown(f" **({param}: {min_nodos})**")
        return prueba, prueba2
    elif param == 'Maximum number of nodes':
        st.markdown(f" **({param}: {max_nodos})**")
        return prueba, prueba2
    else:
        param == 'Specific number of nodes'
        return prueba, prueba2

    

def infreqact(dic, param, measure):
    # Obtener todos los datos de todos los grafos
    all_data = []
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        all_data.extend([valor[measure] for nombre, valor in data])

    # Calcular el promedio general de la medida
    promedio_abs_freq = sum(all_data) / len(all_data)

    # Filtrar los datos según el parámetro y el promedio general
    prueba = {}
    prueba2={}
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        
        if param == 'Mean frequency':
            res = [(nombre, valor) for nombre, valor in data if valor[measure] < promedio_abs_freq]
        elif param == 'Less than 10 (frequency)':
            res = [(nombre, valor) for nombre, valor in data if valor[measure] <= 10]  
        else:
            res = [(nombre, valor) for nombre, valor in data if valor[measure] <= param]
        
        if len(res) > 0:
            prueba[key] = datos 
            prueba2[key] = res

    return prueba, prueba2

def mostfreqact(dic, param, measure):
    # Obtener todos los datos de todos los grafos
    all_data = []
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        all_data.extend([valor[measure] for nombre, valor in data])

    # Calcular el promedio general de la medida
    promedio_abs_freq = sum(all_data) / len(all_data)

    # Filtrar los datos según el parámetro y el promedio general
    prueba = {}
    prueba2={}
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        
        if param == 'Mean frequency':
            res = [(nombre, valor) for nombre, valor in data if valor[measure] > promedio_abs_freq]
        elif param == 'More than 10 (frequency)':
            res = [(nombre, valor) for nombre, valor in data if valor[measure] >= 10]  
        else:
            res = [(nombre, valor) for nombre, valor in data if valor[measure] >= param]
        
        if len(res) > 0:
            prueba[key] = datos 
            prueba2[key] = res

    return prueba, prueba2
    
def transduration(dic, param, measure):
    # Obtener todos los datos de todas las transiciones
    all_data = []
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.edges.data()
        all_data.extend([edge[2][measure] for edge in data])

    # Calcular el promedio general de la duración de las transiciones
    promedio = sum(all_data) / len(all_data)
    # st.write(promedio)

    # Filtrar las transiciones según el parámetro y el promedio general
    prueba = {}
    prueba2={}
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.edges.data()
        
        if (param == 'Mean cycle time of transitions' or param == 'Mean cycle time of activities') :
            res = [edge for edge in data if edge[2][measure] > promedio] 
        else:
            res = [edge for edge in data if edge[2][measure] > param * 60]
    
        if len(res) > 0:
            prueba[key] = datos
            prueba2[key] = res

    # st.write(prueba)

    return prueba, prueba2

def actduration(dic, param):

    all_data = []
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        all_data.extend([edge[2][measure] for edge in data])

    # Calcular el promedio general de la duración de las transiciones
    promedio = sum(all_data) / len(all_data)
    # st.write(promedio)

    # Filtrar las transiciones según el parámetro y el promedio general
    prueba = {}
    prueba2={}
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        
        if param == 'Mean cycle time of transitions':
            res = [edge for edge in data if edge[2][measure] > promedio] 
        else:
            res = [edge for edge in data if edge[2][measure] > param * 60]
    
        if len(res) > 0:
            prueba[key] = datos 
            prueba2[key] = res

    return prueba,prueba2

def transbot(dic, param, inicial, measure):

    # dfg, sa, ea = pm4py.discover_dfg(inicial, activity_key=nodes)
    prueba = {}
    prueba2={}
    maximo = 0
    maximos=[]
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.edges.data()
        max2 = heapq.nlargest(3, (item[2][measure] for item in data))
        maximos.extend(max2)
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.edges.data()
        lista_max = sorted(maximos, reverse=True)
        if(param=='Transition with the maximum duration' or param=='Activity with the maximum duration'):
            valores_mas_altos = lista_max[0]
            res = [edge for edge in data if edge[2][measure] >= valores_mas_altos] 
        else: 
            valores_mas_altos = lista_max[:param] 
            res = [edge for edge in data if edge[2][measure] >= min(valores_mas_altos)]
        if(len(res)>0):
            prueba[key] = datos 
            prueba2[key] =res
    
    return prueba,prueba2

def actbot(dic, param, inicial, measure):
    dfg, sa, ea = pm4py.discover_dfg(inicial, activity_key=nodes)
    prueba = {}
    prueba2={}
    maximo = 0
    maximos=[]
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        max2 = heapq.nlargest(3, (item[2][measure] for item in data))
        maximos.extend(max2)
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        lista_max = sorted(maximos, reverse=True)
        if(param=='Transition with the maximum duration'):
            valores_mas_altos = lista_max[0]
            res = [edge for edge in data if edge[2][measure] >= valores_mas_altos] 
        else: 
            valores_mas_altos = lista_max[:param] 
            res = [edge for edge in data if edge[2][measure] >= min(valores_mas_altos)]
        if(len(res)>0):
            prueba[key] = datos 
            prueba2[key] = res
    
    return prueba,prueba2

def mostfreqresour(dic, param, measure):
    # Obtener todos los datos de todos los grafos
    all_data = []
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        all_data.extend([valor[measure] for nombre, valor in data])

    # Calcular el promedio general de la medida
    promedio_abs_freq = sum(all_data) / len(all_data)

    # Filtrar los datos según el parámetro y el promedio general
    prueba = {}
    prueba2={}
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.nodes.data()
        
        if(param == 'Mean frequency'):
            res = [(nombre, valor) for nombre, valor in data if valor[measure] > promedio_abs_freq]
        else:
            res = [(nombre, valor) for nombre, valor in data if valor[measure] >= param]
        
        if len(res) > 0:
            prueba[key] = datos 
            prueba2[key] = res
    
    return prueba,prueba2

def resourbot(dic, param, inicial, measure):
    dfg, sa, ea = pm4py.discover_dfg(inicial, activity_key=nodes)
    prueba = {}
    prueba2={}
    maximo = 0
    maximos=[]
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.edges.data()
        max2 = heapq.nlargest(3, (item[2][measure] for item in data))
        maximos.extend(max2)
    for key, datos in dic.items():
        graph = datos['graph']
        data = graph.edges.data()
        lista_max = sorted(maximos, reverse=True)
        if(param=='Maximum Cycle Time of resources'):
            valores_mas_altos = lista_max[0]
            res = [edge for edge in data if edge[2][measure] >= valores_mas_altos] 
        else: 
            valores_mas_altos = lista_max[:param] 
            res = [edge for edge in data if edge[2][measure] >= min(valores_mas_altos)]
        if(len(res)>0):
            prueba2[key] = res
    
    return prueba,prueba2

def threshold(datos, metric, a, p, nodes):
    dic={}
    stats_list = [] 
    ident = 0

    for key, dfg in datos.items():
        # st.write(dfg)
        df = dfg['df']
        dfg_ini = dfg['dfg']

        if metric in ['Mean Cycle Time', 'Median Cycle Time', 'StDev Cycle Time', 'Total Cycle Time']:  
            translater={"Median Cycle Time":"median","Mean Cycle Time":"mean","StDev Cycle Time":"stdev","Total Cycle Time":"sum"}

            ac = dict(df[nodes].value_counts())   
            dfg_discovered, sa, ea = pm4py.discover_dfg(df, activity_key=nodes)      

            if(p==100 and a==100):
                dfg_path = dfg_ini
            elif(p==100):
                dfg_path, sa, ea, ac = dfg_filtering.filter_dfg_on_activities_percentage(dfg_discovered, sa, ea, ac, a/100)
            elif(a==100):
                dfg_path, sa, ea, ac = dfg_filtering.filter_dfg_on_paths_percentage(dfg_discovered, sa, ea, ac, p/100)
            else:
                dfg_act, sa, ea, ac = dfg_filtering.filter_dfg_on_activities_percentage(dfg_discovered, sa, ea, ac, a/100)
                dfg_path, sa, ea, ac = dfg_filtering.filter_dfg_on_paths_percentage(dfg_act, sa, ea, ac, p/100)

            
            dfg['sa'] = {key: dfg['sa'][key] for key in list(sa.keys())}
            dfg['ea'] = {key: dfg['ea'][key] for key in list(ea.keys())}
            dfg['dfg'] = {key: dfg['dfg'][key] for key in list(dfg_path.keys())}


            measure=translater[metric]

            pm4py.save_vis_performance_dfg(dfg['dfg'],dfg['sa'],dfg['ea'], './figures/dfg' + str(ident) + '.svg', aggregation_measure=measure)
            # metricas.save_vis_performance_dfg2(dfg['dfg'],dfg['sa'],dfg['ea'], './figures/dfg' + str(ident) + '.svg', aggregation_measure='mean')
            
            meandf, unitdf = statisticslog.mean_case(df)
            mediandf, unitmediandf = statisticslog.median_case(df)

            stats_list.append({
                "key": key,
                "svg_path": './figures/dfg' + str(ident) + '.svg',
                "Mean case duration": meandf,
                "Unit": unitdf,
                "Median cycle time": mediandf,
                "Unitmedian": unitmediandf,
                "Events": len(df),
                "Traces": df['case:concept:name'].nunique(),
                "Activities": df['concept:name'].nunique(),
                "Variants": statisticslog.n_variants(df)
            })

            ident = ident + 1


        else:
            translater={"Absolute Frequency":"abs_freq","Case Frequency":"case_freq",
                "Max Repetitions":"max_repetitions", "Total Repetitions":
                "total_repetitions"}


            ac = dict(df[nodes].value_counts())

            if(p==100 and a==100):
                dfg_path = dfg_ini
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

            measure=translater[metric]
            
            metric_nodes=dict(G.nodes.data(measure))
            
            list_edges=list(G.edges.data())
            dfg_custom={(edge[0],edge[1]):edge[2][measure] for edge in list_edges}

            gviz=apply(dfg_custom,None,None,metric_nodes,None)
            
            
            # left_column.write(gviz)    

            meandf, unitdf = statisticslog.mean_case(df)
            mediandf, unitmediandf = statisticslog.median_case(df)

            stats_list.append({
                "key": key,
                "svg_path": gviz,
                "Mean case duration": meandf,
                "Unit": unitdf,
                "Median cycle time": mediandf,
                "Unitmedian": unitmediandf,
                "Events": len(df),
                "Traces": df['case:concept:name'].nunique(),
                "Activities": df['concept:name'].nunique(),
                "Variants": statisticslog.n_variants(df)
            })


            ident = ident + 1

    return stats_list

def determine_case(dic2):
     # Obtener un valor del diccionario
    sample_value = next(iter(dic2.values()))
 
        # Verificar si es un caso simple (enteros o flotantes)
    if isinstance(sample_value, (int, float)):
        return "simple"
        
    else:
        return "complex"
     

def show_DFGs(stats_list, order, metric, dic2):
    # st.table(stats_list)
    # Display sorted results
    # sorted_stats = {}
    # st.write(stats_list)
    # st.write(dic2)
    order_options = ["Mean case duration", "Median cycle time", "Events", "Traces", "Activities", "Variants"]
    # st.write(determine_case(dic2))
    if(order in order_options):
        sorted_stats = sorted(stats_list, key=lambda x: x[order], reverse=True)
    elif(determine_case(dic2)=='simple'):
        # st.write(stats_list,dic2)
        sorted_stats = sorted(
                    stats_list, 
                    key=lambda x: dic2.get(x["key"], float('inf')), 
                    reverse=True)
    elif(determine_case(dic2)=='complex'):
        # Crear un diccionario con las métricas extraídas
        metric_values = {
            key: dic2[key][0][2].get(metric, float('inf'))  # Usa el valor de la métrica o infinito si no existe
            for key in dic2.keys()
        }

        # Ordenar stats_lists usando los valores del diccionario metric_values
        sorted_stats = sorted(
            stats_list,
            key=lambda x: metric_values.get(x["key"], float('inf'))  # Busca la métrica en metric_values
        )
    else:
        st.error('Error')
    


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
        # if(option!=''):
        #     st7.metric(option, )




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

def returnEdgesInfo(df,concept_name,case_concept_name,timestamp):
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
    max_rep=returnMaxRepititionsEdges(df_edges)
    case_freq=returnCaseFreqEdges(df_edges)
    total_rep = returnTotalRepetitions(df_edges)

    return max_rep,case_freq, total_rep

def returnTotalRepetitions(df):
    totalRep=df.groupby("Transitions").apply(lambda x: (x['case_ids'].value_counts() > 1).sum())
    return totalRep

def returnCaseFreqEdges(df):
    case_freq=df.groupby('Transitions')['case_ids'].nunique()
    return case_freq

def returnMaxRepititionsEdges(df):
    maxRep=df.groupby(["case_ids","Transitions"]).apply(lambda x: len(x)).reset_index().groupby("Transitions").apply(lambda x: max(x[0]))
    return maxRep

def check_log(data):
    if (type(data) != pm4py.objects.log.obj.EventLog):
        data = log_converter.apply(data)
    return data




# if 'dataframe' not in st.session_state:
#     st.session_state.dataframe = pd.DataFrame()

def pattern(inicial, data, nodes, metric, perc_act, perc_path,i):
    # generate_pressed_original = st.session_state.generate_pressed
    # dic = {}
    # datos = {}

    # if len(st.session_state.dataframe):
    if data is not None:
        # dic = st.session_state.dataframe
        # st.write(st.session_state.datos)

        dic = data
        left_column, right_column = st.columns(2)

        if (nodes == 'concept:name'):
            if metric in ['Mean Cycle Time', 'Median Cycle Time', 'StDev Cycle Time', 'Total Cycle Time']:  
                pattern = st.selectbox(
                        'Pattern search',
                        ('Identify DFGs by the number of unique nodes', 
                        'Identify transitions with high duration',
                        'Identify activities with high duration',   # impememtado pero falta comprobar que existe tiempo asociado a actividades
                        'Identify transitions as bottlenecks',
                        'Identify activities as bottlenecks'))    # impememtado pero falta comprobar que existe tiempo asociado a actividades
                        # 'Identify activity loops as bottlenecks'))  # no implementado aun
            else:
                pattern = st.selectbox(
                        'Pattern search',
                        # ('Identify the most frequent fragment',
                        ('Identify DFGs by the number of unique nodes',  
                        'Identify infrequent activities',
                        'Identify the most frequent activities'))

        elif(nodes == 'org:resource'):
            if metric in ['Mean Cycle Time', 'Median Cycle Time', 'StDev Cycle Time', 'Total Cycle Time']: 
                pattern = st.selectbox(
                        'Pattern search',
                        ('Identify DFGs by the number of unique resources', 
                        'Identify resources as bottlenecks',
                        'Identify transitions as bottlenecks',
                        'Identify transitions with high duration'))
            else:
                pattern = st.selectbox(
                        'Pattern search',
                        ('Identify DFGs by the number of unique resources', 
                        'Identify resources with high workload'))
        
        else:
            if metric in ['Mean Cycle Time', 'Median Cycle Time', 'StDev Cycle Time', 'Total Cycle Time']:  
                pattern = st.selectbox(
                    'Pattern search',
                    ('Identify DFGs by the number of unique nodes', 
                    'Identify transitions as bottlenecks',   
                    'Identify transitions with high duration',
                    'Identify activities with high duration',
                    'Identify activities as bottlenecks',))   
            else:
                pattern = st.selectbox(
                    'Pattern search',
                    ('Identify DFGs by the number of unique nodes',
                    'Identify infrequent activities',
                    'Identify the most frequent activities'))

    # return pattern


        param = 0
        prueba = {}
        if(pattern == 'Identify DFGs by the number of unique nodes' or pattern == 'Identify DFGs by the number of unique resources'):
            param = st.selectbox('Number of nodes', 
            ['Minimum number of nodes', "Maximum number of nodes", 'Specific number of nodes'])
            if param == 'Specific number of nodes':
                param = st.number_input('More than X nodes:', step=1, min_value=0)
                option = 'Number of nodes'
            else:
                option=""
            
        # elif (pattern == 'Identify activities with high duration'): #solo es posible si hay tiempo de inicio y fin de actividades
        #     param = st.number_input('Minimum minutes to consider an activity with high duration', step=1) 

        elif (pattern == 'Identify infrequent activities'):
            param = st.selectbox('Maximum frequency to consider an infrequent activity', 
            ['Mean frequency', "Less than 10 (frequency)", 'Other'])
            if param == 'Other':
                param = st.number_input('Maximum absolute frequency to consider an infrequent activity', step=1, min_value=0) 
                option = 'Frequency of activities'
            else:
                option="Frequency of activities"

        elif (pattern == 'Identify the most frequent activities'):
            param = st.selectbox('Minimum threshold to consider the most frequent activities', 
            ['Mean frequency', "More than 10 (frequency)", 'Other'])
            if param == 'Other':
                param = st.number_input('Minimum absolute frequency to consider the most frequent activities', step=1, min_value=0) 
                option = 'Frequency of activities'
            else:
                option="Frequency of activities"

        elif (pattern == 'Identify the most frequent fragment'):
            param = st.number_input('Number of activities of the fragment', step=1, min_value=3) 

        elif (pattern == 'Identify transitions with high duration'):
            param = st.selectbox('Minimum minutes to consider a transition with high duration', 
            ['Mean cycle time of transitions',  'Other'])
            if param == 'Other':
                param = st.number_input('Minimum minutes to consider a transition with high duration', step=1, min_value=0)
                option='Duration of transitions'
            else:
                option=''

        elif (pattern == 'Identify activities with high duration'):
            param = st.selectbox('Minimum minutes to consider an activity with high duration', 
            ['Mean cycle time of activities',  'Other'])
            if param == 'Other':
                param = st.number_input('Minimum minutes to consider a activity with high duration', step=1, min_value=0)
                option='Duration of activities'
            else:
                option='Duration of activities'

        elif (pattern == 'Identify transitions as bottlenecks'):
            param = st.selectbox('Number of transitions', 
            ['Transition with the maximum duration',  'Other'])
            if param == 'Other':
                param = st.number_input('Top-k transitions with the maximum duration', min_value=1,step=1) 
                option='Duration of transitions'
            else:
                option='' 

        elif(pattern == 'Identify resources with high workload'):
            param = st.selectbox('Minimum threshold to consider resources with high workload', 
            ['Mean frequency', 'Other'])
            if param == 'Other':
                param = st.number_input('Minimum workload value (frequency)', step=1, min_value=1)
                option='Workload of resources'
            else:
                option=''

        elif (pattern == 'Identify activities as bottlenecks'):
            param = st.selectbox('Number of activities', 
            ['Activity with the maximum duration',  'Other'])
            if param == 'Other':
                param = st.number_input('Top-k activities with the maximum duration', min_value=1,step=1) 
                option='Duration of activities'
            else:
                option=''

        elif (pattern == 'Identify activity loops as bottlenecks'):
            param = st.number_input('Number of times an activity must occur to be considered bottleneck', step=1, min_value=0)
            # elif (pattern == 'Identify loops'):
            #     param = 
            # elif(pattern == 'Identify decision points'):
            #     param = 

        elif (pattern == 'Identify resources as bottlenecks'):
            param = st.selectbox('Resources with the maximum duration', 
            ['Maximum Cycle Time of resources',  'Other'])
            if param == 'Other':
                param = st.number_input('Top-k resources', min_value=1,step=1) 
                option='Duration of resources'
            else:
                option=''

        elif(pattern == 'Identify rework of activities'):
            param = st.selectbox('Rework of activities', 
            ['Mean rework',  'Other value as maximum rework'])
            if param == 'Other value as maximum rework':
                param = st.number_input('Maximum rework', min_value=1,step=1)      

        translater={"Absolute Frequency":"abs_freq","Case Frequency":"case_freq",
                        "Max Repetitions":"max_repetitions", "Total Repetitions":
                        "total_repetitions","Median Cycle Time":"median","Mean Cycle Time":"mean","StDev Cycle Time":"stdev","Total Cycle Time":"total"}

    # return pattern,param

# def apply_pattern(pattern,param,inicial, dic, nodes, metric, perc_act, perc_path):

    measure=translater[metric]

    selected, selected_data = search(pattern, param, dic, inicial, measure)
    # st.markdown(f" **({param}: {number})**")
    # st.write(selected_data)
    
    copia_dict = copy.deepcopy(selected)

    if(option==''):
        order_options = ["Mean case duration", "Median cycle time", "Events", "Traces", "Activities", "Variants"]
    else:
        order_options = ["Mean case duration", "Median cycle time", "Events", "Traces", "Activities", "Variants", option]

    order_by = st.selectbox("Order by:", order_options, index=0, key='order_specif'+str(i))
    
    stats = threshold(copia_dict, metric, perc_act, perc_path, nodes)
    show_DFGs(stats, order_by, metric, selected_data)
   

            
        
