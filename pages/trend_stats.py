import streamlit as st
import pandas as pd
import numpy as np

from scipy.signal import savgol_filter
np.set_printoptions(precision=2)


def col_vals(lowpass_vals, name):
    st.line_chart(lowpass_vals)
    first_derivative = savgol_filter(lowpass_vals, 5, 2, deriv = 1)
    sec_derivative = savgol_filter(lowpass_vals, 5, 2, deriv =2)

    st.line_chart(first_derivative)
    st.line_chart(sec_derivative)



if 'lowpass_info' not in st.session_state:
    st.session_state.lowpass_info = None

lowpass_info = st.session_state.lowpass_info

if lowpass_info is not None:
    for pair in lowpass_info:
        col_vals(*pair)

