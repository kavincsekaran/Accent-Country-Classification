import numpy as np
from os import listdir
from os.path import isfile, join
import itertools
import pandas as pd
import tensorflow as tf
from yaafelib import *
import h5py

if __name__ == "__main__":
    dir_path="/home/kavin/Silo/CollegeWork/DL/Project/new testing files/"
    split_audiofiles = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    
    x_select=[]
    y_select=[]

    for audiofile in split_audiofiles:
        
        #country=demo_df[demo_df["filename"]==audiofile.split("/")[-1].split('_')[0]]["country"].item()
        #country=demo_df[demo_df["filename"]==audiofile.split("/")[-1].split(".")[0]]["country"].item()
        
        fp = FeaturePlan(sample_rate=44100, resample=True)
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
        feature_names=["mfcc", "energy","loud","sharp","flat","sr", "sf"]
        #print(feats.values()[0].shape)
        extracted_features=np.hstack([feats.get(key, []) for key in feature_names])
        #print((extracted_features.shape[0], 1))
        #print(country)
        #y_select.append(np.full((extracted_features.shape[0], 1), country))
        x_select.append(extracted_features)
    feature_column=[]
    for key in feature_names:
        for i in range(feats.get(key, []).shape[1]):
            feature_column.append(key+str(i))
    x_select=np.vstack(x_select)
 #   y_select=np.vstack(y_select)
    
#    rand_indices=range(x_select.shape[0])
    #np.random.shuffle(rand_indices)

#    train_index=rand_indices[:int(x_select.shape[0]*0.6)]
#    val_index=rand_indices[int(x_select.shape[0]*0.6): int(x_select.shape[0]*0.8)]
#    test_index=rand_indices[int(x_select.shape[0]*0.8): x_select.shape[0]]
    
#    country_labels=list(np.unique(y_select))

#    labels=[country_labels.index(c) for c in y_select]
#    sess = tf.InteractiveSession()
#    labels=tf.one_hot(labels, len(country_labels)).eval(session=sess)
#    y_select=[]
    '''
    data_save={"train":{"features":x_select[train_index], "labels":labels[train_index], "feature_names":feature_names, "label_names":country_labels},
           "validation":{"features":x_select[val_index], "labels":labels[val_index], "feature_names":feature_names, "label_names":country_labels},
           "test":{"features":x_select[test_index], "labels":labels[test_index], "feature_names":feature_names, "label_names":country_labels}
          }
    '''
    hf = h5py.File('/home/kavin/Silo/CollegeWork/DL/Project/new testing files/accent_data_one_file.h5', 'w')
#    g1=hf.create_group("train")
    hf.create_dataset('features', data=x_select, compression="gzip", compression_opts=9)
#    g1.create_dataset('labels', data=labels[train_index], compression="gzip", compression_opts=9)
#    g2=hf.create_group("validation")
#    g2.create_dataset('features', data=x_select[val_index], compression="gzip", compression_opts=9)
#    g2.create_dataset('labels', data=labels[val_index], compression="gzip", compression_opts=9)
#    g3=hf.create_group("test")
#    g3.create_dataset('features', data=x_select[test_index], compression="gzip", compression_opts=9)
#    g3.create_dataset('labels', data=labels[test_index], compression="gzip", compression_opts=9)
    hf.create_dataset('feature_names', data=feature_column)
#    hf.create_dataset('label_names', data=country_labels)
    hf.close()
    
    #np.save("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_train.npy", data_save["train"])
    #np.save("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_validation.npy", data_save["validation"])
    #np.save("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_test.npy", data_save["test"])
    
    '''
    pickle.dump(data_save["train"], open("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_train.pickle", "wb" ), pickle.HIGHEST_PROTOCOL)
    pickle.dump(data_save["validation"], open("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_validation.pickle", "wb" ), pickle.HIGHEST_PROTOCOL)
    pickle.dump(data_save["test"], open("/home/kavin/Silo/CollegeWork/DL/Project/accent_data_test.pickle", "wb" ), pickle.HIGHEST_PROTOCOL)
    '''