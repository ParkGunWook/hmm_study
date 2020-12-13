# hmmlearn을 이용한 0~9 음성 인식기

## Used Library
1. hmm learn : 모델 생성 및 모델 학습
2. librosa : 음성 데이터 로드 및 전처리(mfcc)
3. numpy

## how to train SPEECH RECOGNITION

## Codes
> code source by https://github.com/wblgers/hmm_speech_recognition_demo/blob/master/demo.py

### 1. main func

```python
def main():
    trainDir = './train_audio/'
    trainDataSet = buildDataSet(trainDir)
    print("Finish prepare the training data")

    hmmModels = train_GMMHMM(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")

    testDir = './test_audio/'
    testDataSet = buildDataSet(testDir)

    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")
```

buildDataSet 함수는 음성을 mfcc 전처리해서 train/test dataset을 불러오는 함수이다.

train_GMMHMM은 불러온 train dataset을 통해서 GMMHMM모델을 형성하고 훈련을 진행한다.

마지막으로 만들어진 hmmModels를 통해서 testdataset을 검증한다.

### 2. buildDataSet(dirpath)

```python
def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1]
        feature = extract_mfcc(dir+fileName)
        feature = np.transpose(feature)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset
```

데이터 셋은 딕셔너리 형태를 취하며 mfcc된 value값과 숫자를 의미하는 key값을 가진다.

```python
def extract_mfcc(full_audio_path):
    wave, sample_rate =  librosa.load(full_audio_path)
    mfcc_features = librosa.feature.mfcc(wave, sr = sample_rate, n_mfcc = 12)
    return mfcc_features
```

librosa 라이브러리는 wav 및 여러가지 음성파일을 관리해주는 라이브러리이다.
librosa.load를 통해서 wav의 numpy의 파형과 sampling rate를 리턴한다.
librosa.feature.mfcc에서 3가지의 파라미터를 받는데 첫번쨰는 음성, 두번쨰 sr은 sampling rate, 세번쨰 n_mfcc는 MFCC의 수로서 우리가 FFT를 통해서 얻어낸 주파수 영역을 묶는 기준 수가 된다.

#### MFCC
MFCC는 9가지 단계를 통해서 시계열 데이터를 사용자가 학습하기 용이하게 바꾸어준다.

음성 데이터는 **Pre-empasis**를 통해서 고주파 신호의 세기를 늘려서 저주파와 비교되게끔 한다.

high pass 필터를 지난 음성데이터는 **Windowing**으로 일정 시간의 데이터를 사용자가 지정한 시간에 맞게 프레임을 나누어준다. 이 프레임의 모양은 끝부분이 continuous하게 설정해주면 주파수 분석시에 조금더 좋은 결과를 보인다.

**DFT(FFT를 사용함)**를 이용해서 시영역 데이터를 주파수 영역으로 바꾸어준다.

**Mel-Fiter Bank**는 앞서 말한 mfcc함수의 파라미터 중 n_mfcc와 관련이 있다. 이 단계에서는 주파수 영역의 데이터를 n_mfcc의 수만큼 triangular band-pass filter를 거쳐서 원하는 값을 추출하는데 주파수 영역을 압축시켜서 학습에 용이하게 한다.

이 수는 20개 이하일때는 GMM-HMM 모델에 적합하고 이상이면 DNN 모델에 적합하다. 위 학습에서는 12개를 사용했다.

**log**를 취해서 위의 주파수 영역을 실제보다 적은 값으로 바꾸어준다.

이 주파수 영역의 데이터는 IDFT를 이용해서 다시 시영역 값이 나오게 되고 시계열 데이터로서 GMM-HMM모델에 적합한 데이터가 된다.

> reference : http://www.inf.ed.ac.uk/teaching/courses/asr/2019-20/asr04-signal.pdf

### 3. train_GMMHMM(dataset)

```python
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
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[1]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models
```

GMMHMM_Model은 딕셔너리이고 각 데이터셋의 key(숫자)에 맞게 모델이 저장된다.

hmm.GMMHMM 모델의 파라미터는 n_componets는 모델속의 state 수, n_mix는 GMM에 존재하는 state 수, transmat_prior은 디히클레 분포에 따라서 각 row마다 전이가 일어날 확률, startprob_prior은 디히클레 분포, n_iter은 이터레이션 수이다.

모델을 위와 같이 초기화하고 파라미터를 할당한 후에, label(key)에 맞게 dataset을 지정하고 각 데이터의 길이를 저장한다. 여기서 데이터의 길이는 시계열 데이터의 길이이고 mfcc를 통해서 n_mfcc*length의 값을 가진다.

마지막으로 model.fit을 통해서 모델을 학습시켜서 최적의 파라미터를 찾아낸다.

### 4. test in main func

```python
for label in testDataSet.keys():
        feature = testDataSet[label]
        
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")
```

test dataset은 한개의 숫자에 하나의 데이터를 가지고 있다.

feature은 mfcc를 통해서 추출한 데이터,label은 정답 값, predict는 예측 값이다. model은 hmmModels 즉 이전에 train_GMMHMM이 리턴한 모델이다. 이 모델의 model.score에 feature값을 넣어서 각 모델의 라벨에 로그 우도 함수를 리턴하고 그중 가장 높은 값을 가진 key값을 predict값으로 대입한다.

## 결과

Finish training of the GMM_HMM models for digits 0-9

Test on true label  1 : predict result label is  1

Test on true label  10 : predict result label is  9

Test on true label  2 : predict result label is  2

Test on true label  3 : predict result label is  3

Test on true label  4 : predict result label is  4

Test on true label  5 : predict result label is  5

Test on true label  6 : predict result label is  6

Test on true label  7 : predict result label is  7

Test on true label  8 : predict result label is  6

Test on true label  9 : predict result label is  6

Final recognition rate is 70.00 %

위의 결과가 나왔고 transmatPrior의 초기값을 다양하게 주어봤지만 10, 8, 9는 계속 예측에 실패하고 있다.

# demo_0_9.py

위의 코드는 한국어 데이터셋이 아닌 중국어 데이터이다. 그리고 sampling rate또한 한국어는 16000이고 중국어는 8000이다. 그래서 한국어 데이터 셋으로 경우에는 transmat_이 합이 1이 아닌 다른 값이 나오게 되서 model의 학습이 되지 않고 model.score를 통해서 예측을 실행하면 에러코드가 출력된다.

이 원인은 https://github.com/hmmlearn/hmmlearn/issues/110에 나와 있는데 0*log0이 곱해지면서 transmat_ prob이 이상한 값이 나오게 학습된다고 되어있다.

그리고 원인중 하나는 기존 model에는 5개의 state가 존재하는데 5개의 state중 미방문 state가 생기기 때문이었다.

그래서 state수를 줄임으로서 값을 얻어낼 수 있었다.

## Code(only modifed)

### 1. train_GMMHMM

```python
def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 3
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-1)
    transmatPrior = np.array([[tmp_p, tmp_p, 0], \
                               [0, 0.5, 0.5], \
                               [0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0],dtype=np.float)
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=100)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[1]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models
```

기존의 코드와 같지만 state_num을 줄임에 따라서 5\*5였던 transmatPrior을 3\*3으로 줄이면서 기본 확률값도 조정을 했다. startprobPrior또한 3으로 바꾸었다. 그리고 미방문 state 가능성을 없애기 위해서 iteration을 기존보다 10배 높여주었다.

### 2. buildDataSet

```python
def buildDataSet(dir, train):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        label = fileName[0]
        if(train == 0):
            if fileName[2] == 'b':
                continue
        else:
            if fileName[2] == 'a':
                continue
        feature = extract_mfcc(dir+fileName)
        feature = np.transpose(feature)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset
```

기존의 데이터는 디렉토리가 구분되어 있었지만 한국어 데이터는 a와 b를 통해서 test data와 train data가 구분되어 있다. 그래서 train이라는 변수를 통해서 train과 test data를 구분해주었다.

### 3. test in main

```python
for label in testDataSet.keys():
        feature = testDataSet[label]
        for i in range(3):
            scoreList = {}
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(feature[i])
                #print(score)
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            print("Test on true label ", label, ": predict result label is ", predict)
            if predict == label:
                score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")
```

원래는 1개의 test data만 있던것과 달리 label당 3개의 data가 있기 때문에 for문을 통해서 3번의 반복을 실행한다.

## 한국어 데이터셋 결과

Test on true label  0 : predict result label is  6

Test on true label  0 : predict result label is  6

Test on true label  0 : predict result label is  6

Test on true label  1 : predict result label is  1

Test on true label  1 : predict result label is  1

Test on true label  1 : predict result label is  6

Test on true label  2 : predict result label is  2

Test on true label  2 : predict result label is  2

Test on true label  2 : predict result label is  6

Test on true label  3 : predict result label is  3

Test on true label  3 : predict result label is  3

Test on true label  3 : predict result label is  3

Test on true label  4 : predict result label is  4

Test on true label  4 : predict result label is  4

Test on true label  4 : predict result label is  4

Test on true label  5 : predict result label is  5

Test on true label  5 : predict result label is  9

Test on true label  5 : predict result label is  5

Test on true label  6 : predict result label is  7

Test on true label  6 : predict result label is  6

Test on true label  6 : predict result label is  6

Test on true label  7 : predict result label is  7

Test on true label  7 : predict result label is  7

Test on true label  7 : predict result label is  6

Test on true label  8 : predict result label is  8

Test on true label  8 : predict result label is  8

Test on true label  8 : predict result label is  8

Test on true label  9 : predict result label is  5

Test on true label  9 : predict result label is  9

Test on true label  9 : predict result label is  9

Final recognition rate is 70.00 %

예측은 약 70퍼센트의 정확도를 가졌고 한번 맞은 숫자여도 다른 케이스에서는 틀리는 경우가 있다.