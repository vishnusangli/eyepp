# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
import sp_filters
import plot as myplot

from scipy.signal import detrend
from scipy.signal import savgol_filter
from scipy import signal

np.set_printoptions(precision=8)
# %%
archive_filename = "Siri_60sec_Volts.csv"
current_filename = "Device_3_Volts.csv"
dir = ['Chandrika_Yadav', 'Sirisha']
alt_row = 'CH1\tCH2\tCH3'
df = pd.read_csv(f"data_files/21-06-22/{dir[0]}/{current_filename}")
#info, df = data.clean(df, row = alt_row)

def new_clean(df):
    pos = np.where(df.iloc[:, 0] == 'CH1')[0][0]
    print(pos)
    return pd.DataFrame(np.array(df.iloc[pos + 1:]), columns = df.iloc[pos].tolist(), dtype=np.float64)
df = new_clean(df)
# %%
use_data = df["CH1"]
use_data = list(use_data[6000:30000])
# %%
cutoff = 6.3
fs = 256
filtered_data = sp_filters.butter_lowpass_filter(use_data, cutoff, fs, order = 2)
# %%
cutoff=0.05
b, a = signal.butter(2, cutoff, btype='lowpass') #low pass filter
filtered_data= signal.filtfilt(b, a, use_data)
#filtered_data = data.butter_lowpass_filter(use_data, 0.005, 100, 2)
#filtered_data = data.butter_highpass_filter(filtered_data, 0.05, 1000, 2)

fig, ax = plt.subplots(1,3, figsize = (10, 3))
ax[0].plot(filtered_data, label = 'raw')
ax[0].set_title("Raw")
ax[0].grid()

first_deriv = savgol_filter(filtered_data, window_length= 31, polyorder= 2, deriv = 1)
ax[1].plot(first_deriv)
ax[1].set_title("First Deriv")
ax[1].grid()

second_deriv = savgol_filter(filtered_data, 51, 2, 2)
ax[2].plot(second_deriv)
ax[2].set_title("Second Deriv")
ax[2].grid()
# %%
#### SAVGOL METHOD FOR CHANGEPOINTS ####
minimas, maximas = data.ChangePointDetect.savgol_method(filtered_data, e = 1e-6)
plt.plot(filtered_data, label = 'data')
plt.scatter(minimas, data.point_locate(filtered_data, minimas), 
            label = 'minimas', alpha = 0.1, color = 'green')
plt.scatter(maximas, data.point_locate(filtered_data, maximas),
            label = 'maximas', alpha = 0.1, color ='orange')
plt.legend()
plt.grid()

# %%


# %%
import neurokit2 as nk
# %%
eog_cleaned = nk.eog_clean(filtered_data, sampling_rate=100, method='neurokit')

# %%
plt.plot(data.detrend_standardize( eog_cleaned), alpha =0.7, label = "Cleaned")
plt.legend()

# %%
plt.plot(filtered_data, alpha = 0.7, label = "dtd")
# %%
