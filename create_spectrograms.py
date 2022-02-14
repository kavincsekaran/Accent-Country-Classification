
from os import listdir
from os.path import isfile, join

dir_path="/home/kavin/Silo/CollegeWork/DL/Project/speech-accent-archive/recordings"
selected_audiofiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]

from pydub import AudioSegment
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
from scipy.io import wavfile

for file_ in selected_audiofiles:
    print(file_)
    sound = AudioSegment.from_mp3(file_)
    file_prefix=file_.split("/")[-1].split(".")[0]
    
    sound.export(join("/home/kavin/Silo/CollegeWork/DL/Project/speech-accent-archive/wav", file_prefix+".wav"), format="wav")
    png_f=join("/home/kavin/Silo/CollegeWork/DL/Project/speech-accent-archive/png", file_prefix+".png")
    sample_rate, X = wavfile.read(join("/home/kavin/Silo/CollegeWork/DL/Project/speech-accent-archive/wav", file_prefix+".wav"))    
    print (sample_rate, X.shape)
    specgram(X, Fs=sample_rate, xextent=(0,192000))
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    plt.savefig(png_f,dpi=300)