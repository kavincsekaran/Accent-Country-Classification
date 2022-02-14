
from os import listdir
from os.path import isfile, join
#import pandas as pd


#demo_df=pd.read_csv("/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/speakers_all.csv")
dir_path="/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/wav"
#dir_path="/home/kavin/Silo/CollegeWork/DL/Project/new testing files/wav"

#selected_countries=['usa','china','uk','india','canada']
#selected_prefixes=list(demo_df[demo_df["country"].isin(selected_countries)]["filename"])
selected_audiofiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
selected_audiofiles=[f for f in selected_audiofiles if(not isfile(join("/home/kavin/Silo/CollegeWork/DL/Project/speech-accent-archive/png", f.split("/")[-1].split(".")[0]+".png")))]

from pydub import AudioSegment
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
from scipy.io import wavfile

for file_ in selected_audiofiles:
    print(file_)
    file_prefix=file_.split("/")[-1].split(".")[0]
    
    '''
    sound = AudioSegment.from_mp3(file_)
    sound = sound.set_frame_rate(44100)
    sound.export(join("/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/wav", file_prefix+".wav"), format="wav")
    '''
    #for mode_ in ['magnitude', 'psd', 'angle', 'phase']:
    
    png_f=join("/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/png", file_prefix+"_extent.png")
    sample_rate, X = wavfile.read(file_)    
    print (sample_rate, X.shape)
    specgram(X, Fs=sample_rate, sides='twosided', xextent=(0, 192000))
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    plt.savefig(png_f,dpi=100, bbox_inches='tight', pad_inches = -0.05)
    