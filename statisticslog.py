import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.statistics.traces.generic.log import case_arrival
import pm4py
import statistics

def total_cycle_time(df):
    # Agrupar por caso para encontrar el inicio y fin de cada caso
    cycle_times = df.groupby("case:concept:name").agg(
        start_time=("time:timestamp", "min"),
        end_time=("time:timestamp", "max")
    )

    # Calcular tiempo de ciclo en minutos
    cycle_times["cycle_time"] = (cycle_times["end_time"] - cycle_times["start_time"]).dt.total_seconds() / 60

    # st.write(cycle_times)

    # Calcular el tiempo promedio de ciclo
    average_cycle_time = cycle_times["cycle_time"].mean()
    return round(average_cycle_time, 2)


def calculate_average_cycle_time(df):
    # Agrupar por caso para encontrar el inicio y fin de cada caso
    cycle_times = df.groupby("case:concept:name").agg(
        start_time=("time:timestamp", "min"),
        end_time=("time:timestamp", "max")
    )

    # Calcular tiempo de ciclo en minutos
    cycle_times["cycle_time"] = (cycle_times["end_time"] - cycle_times["start_time"]).dt.total_seconds()
    
    # Calcular el tiempo promedio en segundos
    avg_seconds = cycle_times["cycle_time"].mean()
    
    # Determinar la unidad más apropiada
    if avg_seconds < 3600:  # Menos de 1 hora -> minutos
        avg_time = avg_seconds / 60
        unit = "minutos"
    elif avg_seconds < 86400:  # Menos de 1 día -> horas
        avg_time = avg_seconds / 3600
        unit = "horas"
    else:  # 1 día o más -> días
        avg_time = avg_seconds / 86400
        unit = "días"
    
    return f"{avg_time:.2f} {unit}"

def mean_case(df):
    avg_seconds = sum(pm4py.stats.get_all_case_durations(df))/ len(pm4py.stats.get_all_case_durations(df)) 
    
    # Determinar la unidad más apropiada
    if avg_seconds < 3600:  # Menos de 1 hora -> minutos
        avg_time = avg_seconds / 60
        unit = "min"
    elif avg_seconds < 86400:  # Menos de 1 día -> horas
        avg_time = avg_seconds / 3600
        unit = "h"
    else:  # 1 día o más -> días
        avg_time = avg_seconds / 86400
        unit = "d"
    
    return round(avg_time, 1), unit

def median_case(df):
    avg_seconds = statistics.median(pm4py.get_all_case_durations(df))

    # Determinar la unidad más apropiada
    if avg_seconds < 3600:  # Menos de 1 hora -> minutos
        avg_time = avg_seconds / 60
        unit = "min"
    elif avg_seconds < 86400:  # Menos de 1 día -> horas
        avg_time = avg_seconds / 3600
        unit = "h"
    else:  # 1 día o más -> días
        avg_time = avg_seconds / 86400
        unit = "d"
    
    return round(avg_time, 1), unit
    

def n_variants(df):
    df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp'])  # Asegúrate de que los eventos estén ordenados por caso y timestamp
    variants = df_sorted.groupby('case:concept:name')['concept:name'].apply(lambda x: '->'.join(x)).unique()
    num_variants = len(variants)
    return num_variants

