import os
from copy import copy
from typing import Optional
from typing import Union, List, Dict, Any, Tuple

import streamlit as st

import pandas as pd

from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.heuristics_net.obj import HeuristicsNet
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
from pm4py.utils import get_properties

def save_vis_performance_dfg2(dfg: dict, start_activities: dict, end_activities: dict, file_path: str,
                         aggregation_measure=None, bgcolor: str = "white"):
    """
    Saves the visualization of a performance DFG

    Parameters
    ----------------
    dfg
        DFG object
    start_activities
        Start activities
    end_activities
        End activities
    file_path
        Destination path
    aggregation_measure
        Aggregation measure (default: mean): mean, median, min, max, sum, stdev
    bgcolor
        Background color of the visualization (default: white)
    """
    format = os.path.splitext(file_path)[1][1:]
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
    from pm4py.visualization.dfg.variants import performance as dfg_perf_visualizer
    dfg_parameters = dfg_perf_visualizer.Parameters
    parameters = {}
    parameters[dfg_parameters.FORMAT] = format
    parameters[dfg_parameters.START_ACTIVITIES] = start_activities
    parameters[dfg_parameters.END_ACTIVITIES] = end_activities
    # aggregation_measure = int(str(aggregation_measures[0]) + str(aggregation_measures[1]))
    parameters[dfg_parameters.AGGREGATION_MEASURE] = aggregation_measure
    parameters["bgcolor"] = bgcolor

    gviz = dfg_perf_visualizer.apply(dfg, parameters=parameters)
    # st.graphviz_chart(gviz, use_container_width=True)
    dfg_visualizer.save(gviz, file_path)


