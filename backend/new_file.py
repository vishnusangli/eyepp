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
from io import StringIO
import neurokit2 as nk

from scipy.special import softmax
np.set_printoptions(precision=8)
# %%
archive_filename = "Siri_60sec_Volts.csv"
current_filename = "Device_1_Volts.xls"
dir = ['Chandrika_Yadav', 'ST']
alt_row = 'CH1\tCH2\tCH3'
df = pd.read_csv(f"data_files/21-06-22/{dir[1]}/{current_filename}")
#info, df = data.clean(df, row = alt_row)

def new_clean(df):
    pos = np.where(df.iloc[:, 0] == 'CH1')[0][0]
    print(pos)
    return pd.DataFrame(np.array(df.iloc[pos + 1:]), columns = df.iloc[pos].tolist(), dtype=np.float64)
df = new_clean(df)
# %%
f = open(f"data_files/21-06-22/{dir[1]}/{current_filename}", 'r')
my_list = [line.rstrip('\n') for line in f]
# %%
my_list[23]
# %%
df_eog = pd.read_csv(f"data_files/21-06-22/{dir[1]}/{current_filename}", sep ="\t", skiprows = 23)

# %%
video_filename = "SiriSponteneousblink_video.csv"
df_video = pd.read_csv(f"data_files/21-06-22/{dir[1]}/{video_filename}")
# %%
plt.plot(df_eog['CH1'])

# %%
plt.plot(-df_video['Blink'])
# %%
sec_index = np.linspace(0, len(df_eog), len(df_video))
# %%
df_video["sec_index"] = sec_index
# %%
plt.figure(figsize= (10, 8))
plt.plot(df_eog['CH1'], label = "EOG", color = "red")
plt.plot(df_video["sec_index"], -df_video['Blink'], label ="vid", color = 'blue')
plt.legend()
plt.grid()
# %%
og_xrange = [1800, 3000]
xrange = [1800, 3000]

df_new_vid = df_video[(df_video["sec_index"] < xrange[1]) & (df_video["sec_index"] > xrange[0])]
f, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(df_eog['CH1'][og_xrange[0]: og_xrange[1]], label = "EOG", color = "red")
ax[1].plot(df_new_vid["sec_index"], -df_new_vid['Blink'], label ="vid", color = 'blue')
#%%
use_data = df_eog['CH1']
use_vids = -df_new_vid['Blink']
# %%
cutoff=0.05
b, a = signal.butter(2, cutoff, btype='lowpass') #low pass filter
filtered_data= signal.filtfilt(b, a, use_data)
eog_cleaned = nk.eog_clean(filtered_data, sampling_rate=166, method='neurokit')
# %%
cutoff=0.05
b, a = signal.butter(2, cutoff, btype='lowpass') #low pass filter
filtered_vids= signal.filtfilt(b, a, use_vids)
eog_vids = nk.eog_clean(filtered_vids, sampling_rate=30, method='neurokit')
# %%
df_new_vid = df_video[(df_video["sec_index"] < xrange[1]) & (df_video["sec_index"] > xrange[0])]
f, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(eog_cleaned[og_xrange[0]: og_xrange[1]], label = "EOG", color = "red")
ax[1].plot(df_new_vid["sec_index"], eog_vids, label ="vid", color = 'blue')
# %%
blinks = nk.signal_findpeaks(use_data[og_xrange[0]: og_xrange[1]], relative_height_min=0)
df_blinks = pd.DataFrame(blinks)
# %%
vids = nk.signal_findpeaks(use_vids, relative_height_min=2)
df_vids = pd.DataFrame(vids)
# %%
