
from os import listdir
from os.path import isfile, join
#import pandas as pd


#demo_df=pd.read_csv("/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/speakers_all.csv")
dir_path="/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/wav"
#dir_path="/home/kavin/Silo/CollegeWork/DL/Project/new testing files/wav"

#selected_countries=['usa','china','uk','india','canada']
#selected_prefixes=list(demo_df[demo_df["country"].isin(selected_countries)]["filename"])
#selected_prefixes=[f for f in selected_prefixes if(not isfile(join("/home/kavin/Silo/CollegeWork/DL/Project/speech-accent-archive/png", f+".png")))]
selected_audiofiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]

import numpy as np

import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt


for file_ in selected_audiofiles[:1]:
    print(file_)
    file_prefix=file_.split("/")[-1].split(".")[0]

    sample_rate, samples = wav.read(file_)
    f, t, Zxx = signal.stft(samples, fs=sample_rate)
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='hot')

    png_f=join("/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/png", file_prefix+"_stft.png")
    print (sample_rate, samples.shape)
    
    #specgram(samples, Fs=sample_rate, NFFT=32, noverlap=8, sides='onesided')
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    plt.savefig(png_f,dpi=150, bbox_inches='tight', pad_inches = -0.03)
    