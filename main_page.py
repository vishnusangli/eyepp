# %%
import numpy as np
import streamlit as st
import pandas as pd
import backend.data as data

from scipy.signal import savgol_filter
np.set_printoptions(precision=2)
# %%
st.set_page_config(layout="wide")
st.markdown("# No habla espanol")
st.sidebar.markdown("# Main page")

st.write(""" 
# First Test
Hello *world!*
""")

# %%
uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    info, df_clean = data.clean(df)

    st.write("# File Info")
    st.write(f"Headers {info}")
    st.dataframe(df_clean)

    st.line_chart(df_clean)

    st.write("# Low Pass Filter")

    left_column, right_column = st.columns(2)
    threshold = left_column.slider("Enter Threshold Frequency (Hz)", min_value=0, max_value=100, step = 1, value= 15)
    order = left_column.slider("Enter Filter Order", min_value=0, max_value=10, step = 1, value= 1)
    sampling_freq = right_column.slider("Enter Sampling Frequency (Hz)", min_value=0, max_value=10000, step = 100, value= 1000)
    t_rad = threshold * 2 * np.pi
    s_rad = sampling_freq * 2 * np.pi

    lowpass_vals = data.butter_lowpass_filter(df_clean['CH1'], t_rad, s_rad, order)
    init_rem = 0 #st.slider("Remove val", min_value=0, max_value=len(lowpass_vals), step = 100, value= 1000)
    left_column, right_column = st.columns(2)
    left_column.line_chart(lowpass_vals[init_rem:])
    right_column.line_chart(savgol_filter(lowpass_vals, 5, 2, deriv =1))
