'''
Created on 29/08/2018

@author: wblgers
'''
from __future__ import print_function
import warnings
import os
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import librosa
warnings.filterwarnings('ignore')

def extract_mfcc(full_audio_path):
    wave, sample_rate =  librosa.load(full_audio_path)
    Amp = 1
    wave_noise = np.random.normal(0, 1e-3, wave.shape)
    wave = wave+wave_noise
    mfcc_features = librosa.feature.mfcc(wave, sr = sample_rate, n_mfcc = 12)
    #print(mfcc_features.shape)
    return mfcc_features

def buildDataSet(dir, train):
    # Filter out the wav audio files under the dir
    #print(os.listdir(dir))
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    #print(fileList)
    dataset = {}
    for fileName in fileList:
        label = fileName[0]
        if(train == 0):
            if fileName[2] == 'b':
                #print("Not test")
                continue
        else:
            if fileName[2] == 'a':
                #print("Not train")
                continue
        #print(fileName)
        feature = extract_mfcc(dir+fileName)
        feature = np.transpose(feature)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
        #print(label)
        #print(feature.shape)
    #print(dataset.keys())
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        #print(len(dataset[label]))
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[1]
        print(length)
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
        #print(model)
        print(label)
        print(model.transmat_)
    return GMMHMM_Models

def main():
    trainDir = './data/digit10/'
    trainDataSet = buildDataSet(trainDir, 0)
    #print(len(trainDataSet))
    print("Finish prepare the training data")

    hmmModels = train_GMMHMM(trainDataSet)
    #print(hmmModels)
    print("Finish training of the GMM_HMM models for digits 0-9")

    testDir = './data/digit10/'
    testDataSet = buildDataSet(testDir, 1)
    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            #print(score)
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")


if __name__ == '__main__':
    main()