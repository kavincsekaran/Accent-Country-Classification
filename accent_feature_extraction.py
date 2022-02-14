import numpy as np
from os import listdir
from os.path import isfile, join
import itertools
import pandas as pd
#import tensorflow as tf
from yaafelib import *
import h5py
import time

if __name__ == "__main__":
    demo_df=pd.read_csv("/home/kavin/Silo/CollegeWork/DeepLearning/Project/speakers_all.csv")
    dir_path="/home/kavin/Silo/CollegeWork/DeepLearning/Project/speech-accent-archive/wav"
    #dir_path="/home/kavin/Silo/CollegeWork/DL/Project/new testing files/wav"
    #selected_audiofiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    selected_countries=['usa','china','uk','india','canada']
    gender_list=['female','male']
    selected_audiofiles=[]
    #Random Oversampling
    threshold=350
    range_50=range(50)
    for ind_country in selected_countries:
        country_files=[]
        c_files=list(demo_df[demo_df["country"]==ind_country]["filename"])
        c_files=[join(dir_path, f+".wav") for f in c_files if isfile(join(dir_path, f+".wav"))]
        num_c_files=len(c_files)
        country_files=list(c_files)
        if(num_c_files>0 and num_c_files<threshold):
            fileList_index=range(num_c_files)
            for _ in range((threshold-num_c_files)+np.random.choice(range_50)):
                rand_file=c_files[np.random.choice(fileList_index)]
                country_files.append(rand_file)
        print(len(country_files))
        selected_audiofiles+=country_files
    #selected_prefixes=list(demo_df[demo_df["country"].isin(selected_countries)]["filename"])
    #selected_audiofiles = [join(dir_path, f+".mp3") for f in selected_prefixes if isfile(join(dir_path, f+".mp3"))]
    #selected_audiofiles = [join(dir_path, f+".wav") for f in selected_prefixes if isfile(join(dir_path, f+".wav"))]
    
    #print(len(selected_audiofiles))
    x_select=[]
    y_select_c=[]
    y_select_g=[]
    #print(selected_audiofiles)
    feature_names=["mfcc", "energy","loud","sharp","flat","sr", "sf"]
        
    for audiofile in selected_audiofiles:
        #print(audiofile)
        #country=demo_df[demo_df["filename"]==audiofile.split("/")[-1].split('_')[0]]["country"].item()
        country=demo_df[demo_df["filename"]==audiofile.split("/")[-1].split(".")[0]]["country"].item()
        gender=demo_df[demo_df["filename"]==audiofile.split("/")[-1].split(".")[0]]["sex"].item()
        #country="usa"
        #gender="male"
        
        fp = FeaturePlan(sample_rate=44100, normalize=0.98, resample=True)
        fp.addFeature('mfcc: MFCC blockSize=1024 stepSize=512')
        fp.addFeature('energy: Energy blockSize=1024 stepSize=512')
        fp.addFeature('loud: Loudness blockSize=1024 stepSize=512')
        fp.addFeature('sharp: PerceptualSharpness blockSize=1024 stepSize=512')  
        fp.addFeature('flat: SpectralFlatness blockSize=1024 stepSize=512')
        fp.addFeature('sr: SpectralRolloff blockSize=1024 stepSize=512')
        fp.addFeature('sf: SpectralFlux blockSize=1024 stepSize=512')
        df = fp.getDataFlow()

        # configure an Engine
        engine = Engine()
        engine.load(df)
        # extract features from an audio file using AudioFileProcessor
        afp = AudioFileProcessor()
        afp.processFile(engine,audiofile)
        feats = engine.readAllOutputs()
        #print(feats.values()[0].shape)
        extracted_features=np.hstack([feats.get(key, []) for key in feature_names])
        #print((extracted_features.shape[0], 1))
        #print(country)
        y_select_c.append(np.full((extracted_features.shape[0], 1), country))
        y_select_g.append(np.full((extracted_features.shape[0], 1), gender))
        
        x_select.append(extracted_features)
        #time.sleep(2)
    #print("Processed all files")
    #'''
    feature_column=[]
    for key in feature_names:
        for i in range(feats.get(key, []).shape[1]):
            feature_column.append(key+str(i))
    #'''
    x_select=np.vstack(x_select)
    y_select_c=np.vstack(y_select_c)
    y_select_g=np.vstack(y_select_g)
    
    rand_indices=range(x_select.shape[0])
    #np.random.shuffle(rand_indices)

    train_index=rand_indices[:int(x_select.shape[0]*0.6)]
    val_index=rand_indices[int(x_select.shape[0]*0.6): int(x_select.shape[0]*0.8)]
    test_index=rand_indices[int(x_select.shape[0]*0.8): x_select.shape[0]]
    
    #country_labels=list(np.unique(y_select))

    country_labels=np.array([selected_countries.index(c) for c in y_select_c])
    gender_labels=np.array([gender_list.index(g) for g in y_select_g])
    #sess = tf.InteractiveSession()
    #country_labels=tf.one_hot(country_labels, 20).eval(session=sess)
    #gender_labels=tf.one_hot(gender_labels, 2).eval(session=sess)
    y_select_c=[]
    y_select_g=[]
    '''
    data_save={"train":{"features":x_select[train_index], "labels":labels[train_index], "feature_names":feature_names, "label_names":country_labels},
           "validation":{"features":x_select[val_index], "labels":labels[val_index], "feature_names":feature_names, "label_names":country_labels},
           "test":{"features":x_select[test_index], "labels":labels[test_index], "feature_names":feature_names, "label_names":country_labels}
          }
    '''
    hf = h5py.File('accent_data_top_5_country_gender_normalized_oversampled.h5', 'w')
    g1=hf.create_group("train")
    g1.create_dataset('features', data=x_select[train_index], compression="gzip", compression_opts=9)
    g1.create_dataset('country', data=country_labels[train_index], compression="gzip", compression_opts=9)
    g1.create_dataset('gender', data=gender_labels[train_index], compression="gzip", compression_opts=9)
    g2=hf.create_group("validation")
    g2.create_dataset('features', data=x_select[val_index], compression="gzip", compression_opts=9)
    g2.create_dataset('country', data=country_labels[val_index], compression="gzip", compression_opts=9)
    g2.create_dataset('gender', data=gender_labels[val_index], compression="gzip", compression_opts=9)
    g3=hf.create_group("test")
    g3.create_dataset('features', data=x_select[test_index], compression="gzip", compression_opts=9)
    g3.create_dataset('country', data=country_labels[test_index], compression="gzip", compression_opts=9)
    g3.create_dataset('gender', data=gender_labels[test_index], compression="gzip", compression_opts=9)
    hf.create_dataset('feature_names', data=feature_column)
    hf.create_dataset('country_names', data=selected_countries)
    hf.create_dataset('gender_names', data=gender_list)
    hf.close()
    
    #np.save("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_train.npy", data_save["train"])
    #np.save("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_validation.npy", data_save["validation"])
    #np.save("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_test.npy", data_save["test"])
    
    '''
    pickle.dump(data_save["train"], open("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_train.pickle", "wb" ), pickle.HIGHEST_PROTOCOL)
    pickle.dump(data_save["validation"], open("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_validation.pickle", "wb" ), pickle.HIGHEST_PROTOCOL)
    pickle.dump(data_save["test"], open("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_test.pickle", "wb" ), pickle.HIGHEST_PROTOCOL)
    '''