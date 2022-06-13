import streamlit as st
import numpy as np
import pandas as pd
import backend.data as data
from scipy.signal import savgol_filter
np.set_printoptions(precision=2)

st.title("Apply Low Pass")
st.sidebar.markdown("# Apply Low Pass")

if 'df_use' not in st.session_state:
    st.session_state.df_use = None

df_use = st.session_state.df_use

def col_vals(arr, name):
    st.sidebar.markdown(f"Apply Low Pass for Column: {name}")


    threshold = st.sidebar.slider("Enter Threshold Frequency (Hz)", min_value=0, max_value=100, step = 1, value= 15, key = f"threshold_{name}")
    order = st.sidebar.slider("Enter Filter Order", min_value=0, max_value=10, step = 1, value= 1, key = f"order_{name}")
    sampling_freq = st.sidebar.slider("Enter Sampling Frequency (Hz)", min_value=0, max_value=10000, step = 100, value= 1000, key = f"order_{name}")
    t_rad = threshold * 2 * np.pi
    s_rad = sampling_freq * 2 * np.pi

    lowpass_vals = data.butter_lowpass_filter(arr, t_rad, s_rad, order)
    st.write(f"# Column: {name}")
    st.write("Values")
    st.line_chart(lowpass_vals)
    st.write("Derivatives")
    st.line_chart(savgol_filter(lowpass_vals, 5, 2, deriv =1))

if df_use is not None:
    for i in df_use.columns:
        col_vals(df_use[i], i)




