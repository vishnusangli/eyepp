import streamlit as st
import numpy as np
import pandas as pd
import backend.data as data


st.title("File Upload")
st.sidebar.markdown("# File Upload")

if 'df_use' not in st.session_state:
    st.session_state.df_use = None

if 'df_master' not in st.session_state:
    st.session_state.df_master = None

if 'file_info' not in st.session_state:
    st.session_state.file_info = None

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
if uploaded_file is not None and st.session_state.df_master is None:
    df = pd.read_csv(uploaded_file)
    info, df_clean = data.clean(df)
    df_clean.columns = ["Left", "Right"] #Change here accordingly
    st.session_state.df_master = df_clean
    st.session_state.file_info = info

if st.session_state.df_master is not None:

    st.write("# File Info")
    st.write(f"Headers {st.session_state.file_info}")

    if st.checkbox('View Cleaned Table'):
        st.dataframe(st.session_state.df_master)

    st.session_state.df_use = st.session_state.df_master
    if st.checkbox('Exclude parts of data'):
        add_slider = st.slider('Select range to include', min_value = 0, max_value = len(st.session_state.df_master), value = (0, len(st.session_state.df_master)))
        st.session_state.df_use = st.session_state.df_master.iloc[add_slider[0]:add_slider[1]]


    st.line_chart(st.session_state.df_use)