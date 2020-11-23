import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def plot_signal(signals):
	fig, axes = plt.subplots(nrows=2, nclos=5, sharex=False, sharey=True, figsize=(20,5))
	fig.suptitle('Time Series', size=16)
	i = 0
	for x in range(2):
		for y in range(5):
			axes[x,y].set_title(list(signals.keys())[i])
			axes[x,y].plot(list(signals.values())[i])
			axes[x,y].get_xaxis().set_visible(False)
			axes.get_yaxis().set_visible(False)
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