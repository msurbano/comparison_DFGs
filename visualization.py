import streamlit as st
import pm4py

def small_text(text):
    return f"<p style='font-size:12px; color:grey; font-style:italic;'>{text}</p>"

def search_differences(keys):
    col1, col2, col3, col4 = st.columns(4)

    df = st.session_state.original
    
    col11, col12 = col1.columns(2)
    search = col11.selectbox('Search for', ('Existence of activities', 'Identify control-flow differences'))
                            # 'Difference in frequency','Other'))
    if(search=='Identify control-flow differences):
       search='Stable parts'
    else:
        search=search
    
    explanations = {
                'Existence of activities': "Highlight a set of activities that are fully ('All included') or partially ('Some included') included in the DFGs.",
                'Stable parts': "Highlight common nodes and edges between DFGs using a reference model, which could be the entire process or a DFG from the collection.",
                # 'Difference in frequency': "(none)"
            }
    col1.markdown(small_text(explanations[search]), unsafe_allow_html=True)
    add = False

    # color_selectbox(2, 'green')

    if(search == 'Existence of activities'):
        mode = col12.selectbox('Mode',('All included', 'Some included'), label_visibility="hidden")
        values = col2.multiselect('Activities', df['concept:name'].unique(), label_visibility="hidden")
        color_mode = col3.radio('Color', ['Same color', 'Different color'], horizontal=True)
        explanations = {
                'Same color': "Use only one color for all the activities selected.",
                'Different color': "Use a different color to highlight each activity selected."
            }
        col3.markdown(small_text(explanations[color_mode]), unsafe_allow_html=True)


    elif(search == 'Difference in frequency'):
        mode = col12.selectbox('Reference value',('Nodes', 'Edges'), label_visibility="hidden")
        if(mode=='Nodes'):
            values = col2.multiselect('Nodes', df['concept:name'].unique(), label_visibility="hidden")
        else:
            values = values = col2.multiselect('Nodes', df['concept:name'].unique(), label_visibility="hidden")
        color_mode=False


    elif(search == 'Stable parts'):
        mode = col12.selectbox('Reference model', ['Whole process'] + list(keys))
        values = []
       
        # color_mode = col2.multiselect('Highlight', ['Similarities', 'Differences DFG', 'Differences reference model'], 
        #                               placeholder='Choose some options')
        if(mode=='Whole process'):
            color_mode = col2.multiselect('Highlight', ['Similarities',  'Differences reference model'], 
                                      placeholder='Choose some options')
        else:
            color_mode = col2.multiselect('Highlight', ['Similarities', 'Differences DFG', 'Differences reference model'], 
                                      placeholder='Choose some options')
            if('Similarities' in color_mode and 'Differences DFG' in color_mode and 'Differences reference model' in color_mode):
                add = col3.checkbox('Show the activities of the whole process')
        #     color_mode = col2.radio('Highlight', ['Differences', 'Similarities', 'Similarities and differences'], horizontal=True)
        #     explanations = {
        #         'Similarities': "For each DFG, highlight the nodes and transitions that are included in the reference model.",
        #         'Differences': "For each DFG, highlight the nodes and transitions that are not included in the reference model.",
        #         'Similarities and differences': "The nodes and edges that belong only to the DFG are highlighted in yellow. Those common to both the DFG and "
        #         "the reference are in orange. The ones exclusive to the reference DFG are in red, while those that do not belong to "
        #         "either are shown in gray."
        #     }
        #     col2.markdown(small_text(explanations[color_mode]), unsafe_allow_html=True)
        # else:
        #     color_mode=col2.radio('Highlight', ['Same color', 'Freq. color'], horizontal=True)
        #     explanations = {
        #         'Same color': "Use only one color for the activities included in the DFG.",
        #         'Freq. color': "Use different colors according to the frequency of the activities included in the DFG.",
        #     }
        #     col2.markdown(small_text(explanations[color_mode]), unsafe_allow_html=True)
    else:
        values = []
        mode=''
        color_mode=False
    

    return (search, (mode,values), color_mode, add)

def zoom_fragment(col1, dic):
    df = st.session_state.original
    filtered_dataframe={}
    # col1, col2, col3, col4 = st.columns(4)


    z = col1.selectbox('Visualization focus', ('Whole process','Zoom subprocess'))
    col11, col12 = col1.columns(2)
    if(z=='Zoom subprocess'):

        # col12, col22 = col2.columns(2)
        activityFROM = col11.selectbox('From', df['concept:name'].unique())
        # ("visible", "hidden", or "collapsed"))
        activityTO = col12.selectbox('To', df['concept:name'].unique(), index=len(df['concept:name'].unique())-1)

        for key,group in dic.items():
                    
            filt = pm4py.filter_between(group, 
                        activityFROM,activityTO, activity_key='concept:name', 
                                case_id_key='case:concept:name', timestamp_key='time:timestamp')
            if(len(filt)!=0):
                if(key==''):
                    filtered_dataframe[str(activityFROM) + " -- " + str(activityTO)] = filt
                else:
                    filtered_dataframe[key + " " + str(activityFROM) + " -- " + str(activityTO)] = filt
    else:
        filtered_dataframe = dic


    return filtered_dataframe
                
def show_activities(col2, df):
    delete_act = set()
    # col1, col2, col3, col4 = st.columns(4)
    z = col2.selectbox('Visualization of activities', ('All activities', 'Hide activities', 'Filter events by activities'), key='act_delete')
    if(z=='Hide activities'):
        delete_act = col2.multiselect('Activities to hide', df['concept:name'].unique(), label_visibility="hidden", key='delete')
    # elif(z=='Filter events by activities'):
    #     delete_act = col2.multiselect('Activities to keep', df['concept:name'].unique(), label_visibility="hidden", key='delete')

    return z, delete_act

def filter_events(dic, act):
    filtered_subsets = {}
    
    for key,subset in dic.items():
        grupo = pm4py.filter_event_attribute_values(subset, 'concept:name', act, retain=False, level='event')
        if(len(grupo)!=0):
            filtered_subsets[key] = grupo

    return filtered_subsets, []



    
    

    


