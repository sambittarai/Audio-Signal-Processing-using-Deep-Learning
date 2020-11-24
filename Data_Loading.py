import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


def plot_signals(signals):
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Time Series', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(signals.keys())[i])
			axes[x,y].plot(list(signals.values())[i])
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_fft(fft):
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Fourier Transforms', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			data = list(fft.values())[i]
			Y, freq = data[0], data[1]
			axes[x,y].set_title(list(fft.keys())[i])
			axes[x,y].plot(freq, Y)
			axes[x, y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_fbank(fbank):
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Filter Bank Coefficients', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(fbank.keys())[i])
			axes[x,y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def plot_mfccs(mfccs):
	fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(mfccs.keys())[i])
			axes[x,y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
			axes[x,y].get_xaxis().set_visible(False)
			axes[x,y].get_yaxis().set_visible(False)
			i += 1

def calc_fft(signal, rate):
	n = len(signal)
	freq = np.fft.rfftfreq(n, d=1/rate)
	Y = abs(np.fft.rfft(signal)/n)
	return (Y, freq)
 
def envelope(signal, rate, threshold):
	#Get rid of the low magnitude signal from the signal
	mask = []
	y = pd.Series(signal).apply(np.abs)
	y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
	for mean in y_mean:
		if mean > threshold:
			mask.append(True)
		else:
			mask.append(False)
	return mask


# Loading the data
df = pd.read_csv("/content/drive/MyDrive/Audio Signal Processing/Data/instruments.csv")
df = df.drop(columns='Unnamed: 0')
df.set_index('Index', inplace=True)

for f in tqdm(df.index):
	rate, signal = wavfile.read(f)
	df.at[f, 'length'] = signal.shape[0]/rate #This will give us the length of the signal

classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels = class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

#Visualizing the data
signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
	wav_file = df[df.Label == c].iloc[0,0]
	signal, rate = librosa.load(wav_file, sr=44100)
	mask = envelope(signal, rate, 5e-4)
	signal = signal[mask]
	signals[c] = signal
	fft[c] = calc_fft(signal, rate)
	bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
	fbank[c] = bank
	mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
	mfccs[c] = mel

plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()
plot_fbank(fbank)
plt.show()
plot_mfccs(mfccs)
plt.show()

#Downsampling the audio files that we will be using for modelling
clean_data_path = '/content/drive/MyDrive/Audio Signal Processing/Clean Data'
for f in tqdm(df.Index):
	signal, rate = librosa.load(f, sr=16000)#downsample to 16000
	mask = envelope(signal, rate, 5e-4)
	path = f.split('/')[-2:]
	if os.path.exists(os.path.join(clean_data_path, path[0])) == False:
		os.mkdir(os.path.join(clean_data_path, path[0]))
	wavfile.write(filename = os.path.join(clean_data_path, path[0], path[1]),
	 rate=rate, data=signal[mask])
