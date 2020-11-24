import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc


def build_rand_feat():
	#Creating Samples
	X, y = [], []
	_min, _max = float('inf'), -float('inf')
	for _ in tqdm(range(n_samples)):
		rand_class = np.random.choice(class_dist.index, p=prob_dist)
		file = np.random.choice(df[df.Label==rand_class].index)
		rate, wav = wavfile.read(file)
		label = df.at[file, 'Label']
		rand_index = np.random.randint(0, wav.shape[0] - config.step)
		sample = wav[rand_index:rand_index + config.step]
		X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
		_min = min(np.amin(X_sample), _min)
		_max = max(np.amax(X_sample), _max)
		X.append(X_sample if config.mode == 'conv' else X_sample.T)
		y.append(classes.index(label))
	X, y = np.array(X), np.array(y)
	X = (X - _min) / (_max - _min)
	if config.mode == 'conv':
		X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
	elif config.mode == 'time':
		X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
	y = to_categorical(y, num_classes=10)

	return X, y



class Config:
	def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
		self.mode = mode
		self.nfilt = nfilt
		self.nfeat = nfeat
		self.nfft = nfft
		self.rate = rate
		self.step = int(rate/10)


df = pd.read_csv("/content/drive/MyDrive/Audio Signal Processing/Clean Data/instruments_clean.csv")
df.set_index('Index', inplace=True)

#for f in df.index:
#	rate, signal = wavfile.read(f)

classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['length'].mean()

n_samples = 2*int(df['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum() #Probability Distribution
choices = np.random.choice(class_dist.index, p=prob_dist)

config = Config(mode='conv')
if config.mode == 'conv':
	X, y = build_rand_feat()

elif config.mode == 'time':
	X, y = build_rand_feat()

