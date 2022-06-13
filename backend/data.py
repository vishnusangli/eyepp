import pandas as pd
import numpy as np
import streamlit as st
from scipy.signal import butter, lfilter
from scipy.signal import freqs
import altair as alt

def clean(df):
    """
    Clean the raw DataFrame from input file
    
    """
    vals = []
    for elem in df.iloc[:, 0]:
        newval = elem
        if type(newval) == str:
            newval = newval.split()
        vals.append(newval)

    vals_start = np.where(df.iloc[:, 0] == 'CH1\tCH2')[0][0]
    info = vals[:vals_start - 1]
    cols, data = vals[vals_start], vals[vals_start + 1:]
    data = np.array(data, dtype=float)
    df = pd.DataFrame(data[100:], columns = cols)
    return info, df


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y
rad = 2* np.pi
cutOff = 15 * rad#cutoff frequency in rad/s
fs = 1000 * rad#sampling frequency in rad/s
order = 4 #order of filter

#print sticker_data.ps1_dxdt2


def summ():
    source = pd.DataFrame(np.cumsum(np.random.randn(100, 3), 0).round(2),
                        columns=['alcohol', 'beer', 'coke'], index=pd.RangeIndex(100, name='x'))
    source = source.reset_index().melt('x', var_name='category', value_name='y')

    line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
        alt.X('x', title='Year'),
        alt.Y('y', title='Amount in liters'),
        color='category:N'
    ).properties(
        title='Sales of consumer goods'
    )

    st.altair_chart(line_chart)