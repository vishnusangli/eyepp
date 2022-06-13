# %%
import numpy as np
import streamlit as st
import pandas as pd
import backend.data as data
import altair as alt

from scipy.signal import savgol_filter
np.set_printoptions(precision=2)

# %%

st.set_page_config(layout="wide")
st.title("Main Page")
st.sidebar.markdown("# Main page")

st.write(""" 
# First Test
Hello *world!*

Go to Upload file Page to get started
""")


