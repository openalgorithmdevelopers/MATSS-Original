# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:46:55 2021

@author: bhupendra.singh
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile


totalSubjects = 21
totalUtterances = 60    #the number of utterances of words in a folder for every subject
featureName = "mfcc"
features_length = 6373

#utterancesFolder = ".//WebBasedTest/word_level_utterances"
utterancesFolder = ".//WebBasedTest/word_level_utterances_manual"
featuresFolder = ".//WebBasedTest/features_files"

avialableSubjectList = list(range(totalSubjects))
avialableSubjectList = [x+1 for x in avialableSubjectList]  # adding 1 to all the itesms since the subject folder starts from 1 and not 0
#avialableSubjectList.remove(4)

totalUtterancesList = list(range(totalUtterances))
totalUtterancesList = [x+1 for x in totalUtterancesList]  # adding 1 to all the itesms since the subject folder starts from 1 and not 0

features_set = np.zeros((len(totalUtterancesList), features_length))
fs = 48000  #i found it to be this value, change it as per your information

for currentSubjectNo in tqdm(avialableSubjectList):     # iterate for all the available subjects
    #currentSubjectNo += 1
    print("Current Subject = " + str(currentSubjectNo))
    
    for currentUtteranceNo in totalUtterancesList: #iterate for for all the utterances
        #print("Current Subject = " + str(currentUtteranceNo))
        #utteranceFileName = utterancesFolder + "/utterances_subject_" + str(currentSubjectNo) + "/utterance" + str(currentUtteranceNo) + ".wav"
        utteranceFileName = utterancesFolder + "/utterances_subject_" + str(currentSubjectNo) + "/" + str(currentUtteranceNo) + ".wav"
        #print(fileName)
        #sound_file = AudioSegment.from_wav(utteranceFileName)
        #mfcc_features = mfcc(np.array(sound_file.get_array_of_samples()), fs)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        y = smile.process_file(utteranceFileName)
        #mfcc_features.resize(128) #standardizing the size
        y = y.to_numpy()
        #y = y[0,:]
        #y = y.resize(6372) #standardizing the size
        features_set[currentUtteranceNo - 1, :] = y
        
    df = pd.DataFrame(features_set)
    featuresFileName = featuresFolder + "/Subject" + str(currentSubjectNo) + "_" + featureName + ".csv"
    df.to_csv(featuresFileName)