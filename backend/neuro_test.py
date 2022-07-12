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
import neurokit2 as nk
# %%
eog_cleaned = nk.eog_clean(filtered_data, sampling_rate=100, method='neurokit')
plt.plot(data.detrend_standardize( eog_cleaned), alpha =0.7, label = "Cleaned")
plt.plot(data.detrend_standardize( filtered_data), alpha =0.7, label = "Raw")
plt.legend()
plt.grid()
# %%
#eog_cleaned = np.divide(eog_cleaned, np.max(eog_cleaned))
blinks = nk.signal_findpeaks(eog_cleaned, relative_height_min=0)
print(f"{len(blinks['Peaks'])} Blinks found")
# %%
df_blinks = pd.DataFrame(blinks)

# %%
loc = 20
elem = df_blinks.iloc[loc]
plt.plot(range(int(elem['Onsets']), int(elem['Offsets'])), eog_cleaned[int(elem['Onsets']): int(elem['Offsets'])])
plt.scatter(elem['Peaks'], eog_cleaned[int(elem['Peaks'])])
print(elem)
# %%
events = nk.epochs_create(eog_cleaned, blinks['Peaks'], sampling_rate=200, epochs_start=-0.4, epochs_end=0.6)
events = nk.epochs_to_array(events)  # Convert to 2D array
# %%
import scipy
def fit_gamma(x, loc, a, scale):
    x = nk.rescale(x, to=[0, 10])
    gamma = scipy.stats.gamma.pdf(x, a=a, loc=loc, scale=scale)
    y = gamma / np.max(gamma)
    return y


def fit_scr(x, time_peak, rise, decay1, decay2):
    x = nk.rescale(x, to=[0, 10])
    gt = np.exp(-((x - time_peak) ** 2) / (2 * rise ** 2))
    ht = np.exp(-x / decay1) + np.exp(-x / decay2)

    ft = np.convolve(gt, ht)
    ft = ft[0 : len(x)]
    y = ft / np.max(ft)
    return y


x = np.linspace(0, 100, num=len(events))

p_gamma = np.full((events.shape[1], 3), np.nan)
p_bateman = np.full((events.shape[1], 3), np.nan)
p_scr = np.full((events.shape[1], 4), np.nan)

for i in range(events.shape[1]):
    if np.isnan(events[:, i]).any():
        break
    events[:, i] = nk.rescale(events[:, i], to=[0, 1])  # Reshape to 0-1 scale
    try:
        p_gamma[i, :], _ = scipy.optimize.curve_fit(fit_gamma, x, events[:, i], p0=[3, 3, 0.5])
        p_scr[i, :], _ = scipy.optimize.curve_fit(fit_scr, x, events[:, i], p0=[3.5, 0.5, 1, 1])
    except RuntimeError:
        pass
participant = 1

p_gamma = pd.DataFrame(p_gamma[~np.isnan(p_gamma).any(axis=1)], columns=["loc", "a", "scale"])
p_gamma["Participant"] = participant
p_gamma["Task"] = 1
params_gamma = pd.DataFrame(p_gamma)
p_scr = pd.DataFrame(p_scr[~np.isnan(p_scr).any(axis=1)], columns=["time_peak", "rise", "decay1", "decay2"])
p_scr["Participant"] = participant
p_scr["Task"] = 1
params_scr = p_scr
# %%
x = np.linspace(0, 100, num=len(events))
template_gamma = fit_gamma(x, *np.nanmedian(params_gamma.iloc[:, [0, 1, 2]], axis=0))
template_scr = fit_scr(x, *np.nanmedian(params_scr.iloc[:, [0, 1, 2, 3]], axis=0))

plt.plot(events, linewidth=0.02, color="black")
plt.plot(template_gamma, linewidth=2, linestyle='-', color="#4CAF50", label='Gamma')
plt.plot(template_scr, linewidth=2, linestyle='-', color="#9C27B0", label='SCR')
plt.legend(loc="upper right")
#plt.savefig("figures/fig2.png")
plt.show()
# %%

# %%
blink_range = [0, 2000]
new_data = eog_cleaned[blink_range[0]:blink_range[1]]

plt.figure(figsize = (10, 8))
plt.plot(new_data)
plt.grid()

for count, i in df_blinks.iterrows():
    if i['Offsets'] <= blink_range[1] and i['Onsets'] >= blink_range[0]:
        start = int(i['Onsets'])
        end = int(i['Offsets'])
        plt.scatter(start, new_data[start], color = 'red', alpha = 0.7)
        plt.scatter(end, new_data[end], color = 'blue', alpha = 0.7)
        plt.scatter(i['Peaks'], new_data[int(i['Peaks'])], color = 'green', alpha = 0.5)
        plt.text(i['Peaks'] - 50, new_data[int(i['Peaks'])] + 0.00005, f"{round(new_data[int(i['Peaks'])], 6)}", fontsize = 10, color = 'red')
        plt.text(i['Peaks'] - 50, new_data[int(i['Peaks'])] + 0.00007, f"{round(i['Width'], 2)}", fontsize = 10, color = 'green')
        plt.text(i['Peaks'] - 50, new_data[int(i['Peaks'])] + 0.00003, f"{count}", fontsize = 10)
plt.legend()
# %%

"""
blink start, end

Width at -
    0.25, 0.5, 0.75
max slope while closing, while opening

slope at 3 points
duration between blinks
blink duration
acceleration

"""

class blink_segmentation:
    def __init__(self) -> None:
        pass

    def nkt_find_peaks(self) -> pd.DataFrame:
        pass

    
# %%
p = False
width50, width_vals, b, c = signal.peak_widths(eog_cleaned, df_blinks["Peaks"], rel_height = 0.5)
width75, width_vals, b, c = signal.peak_widths(eog_cleaned, df_blinks["Peaks"], rel_height = 0.25)
width25, width_vals, b, c = signal.peak_widths(eog_cleaned, df_blinks["Peaks"], rel_height = 0.75)
new_list = []
for i, blink in df_blinks.iterrows():
    try:
        start = int(blink['Onsets'])
        end = int(blink['Offsets'])
    except Exception:
        new_list.append([np.NaN, np.NaN, np.NaN])
        continue
    blink_range = eog_cleaned[start: end]

    first_deriv = savgol_filter(blink_range, window_length= 3, polyorder= 2, deriv = 1)
    sec_deriv = savgol_filter(blink_range, window_length= 3, polyorder= 2, deriv = 2)
    max_sp = np.where(first_deriv ==np.max(first_deriv))[0]
    min_sp = np.where(first_deriv ==np.min(first_deriv))[0]
    if p:
        f, ax = plt.subplots(3, 1, figsize = (4, 10))
        ax[0].plot(blink_range)

        ax[1].plot(first_deriv)
        ax[1].scatter(min_sp, first_deriv[min_sp])
        ax[1].scatter(max_sp, first_deriv[max_sp])


        ax[2].plot(sec_deriv)
    if i + 1< len(df_blinks):
        b_inter = df_blinks["Onsets"][i + 1] - end
    else: b_inter = np.NaN

    new_list.append([max_sp[0], min_sp[0], b_inter])
temp = pd.DataFrame(new_list, columns = ["min_deriv", "max_deriv", "for_inter"])
temp["0.25width"] = width25
temp["0.5width"] = width50
temp["0.75width"] = width75
# %%

# %%
df_blinks = pd.concat([df_blinks, temp])
# %%

from scipy.signal import chirp, find_peaks, peak_widths
import matplotlib.pyplot as plt
x = blink_range
peaks, _ = find_peaks(x)
results_half = peak_widths(x, peaks, rel_height=0.5)
results_quarter = peak_widths(x, peaks, rel_height=0.25)

# %%
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.hlines(*results_half[1:], color="red")
plt.hlines(*results_quarter[1:], color="green")
plt.show()
# %%
print(results_half)
# %%
