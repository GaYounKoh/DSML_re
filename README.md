# DSML_re
DSML_restart
<br><br>
<i><b><q>중요한 내용</q></b>은 앞으로 뺌</i>
<br><br><br><br>

# 가장 중요한 사실 221212
X shape (7727, 10, 3595)으로 업데이트 된 거였음. ~~4068 제발 잊어.~~ <br>



# 읽어보세요.
[AdaBoost 정리된 글](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-14-AdaBoost) <br>

# 질문
<<220227>> <br>
~~preds2_mean_lst.append(np.mean(preds2))~~ <br>
~~<span style = 'color : red'>예측 for문에서 df 만들 때 pred에 이미 평균을 냈는데 평균을 왜 또 내..?</span>~~ <br>
==> for문이 feature별로 돌아가면 예측은 환자수만큼 나와. <br>
그니까 feature 별로 하나의 값으로 보려면 나온 값들의 평균으로 보는게 가장 좋음.

<br>

# 안 보는 자료
[6-1], [6-2]는 각각 조합과 CuDNNLSTM 관련 내용이므로 지금은 일단 버림
<br><br>


# cudnnLSTM
[cudnnLSTM](https://buomsoo-kim.github.io/keras/2019/08/02/Easy-deep-learning-with-Keras-22.md/)
<br><br><br>

# 220223
```
정리

mimic-iii data: 2001년~2012년 미국 중환자실 (ICU) data
7727: 폐렴환자 수, 근데 이제 abnormal(x matrix) sum이 0인 아무 의미 없는 환자들을 제외한. # DSML_re/[해외논문] Journal of Biomedical Informatics/01 데이터 전처리.ipynb
4068: 폐렴환자들에게서 발견된 item 수 (4069 -> 4068이 된 이유? : 7727로 줄여보니까 해당 아이템이 싹 0이었어서)

퇴원을 d-day로 두고 [d-10 ~ d-1]까지를 x의 time에,
[d-day]는 y에.

x [1: item을 통해 문제 발견(반응 有), 0: item에 대한 반응 無], shape: (7727, 10, 4068)
y [1:사망, 0:퇴원], shape: (7727, 1) # 1은 d-day를 의미

<!-- x,y의 80%를 train에 사용 (80%로 모델을 생성하고 저장)
x,y의 20%는 test에 사용 (20%로 예측) -->


x,y 모두를 학습에 사용한 모델을 사용해서
x,y 모두에 대해 예측.



[07. best model saving]
가장 예측력이 좋은 모델 뽑아서 저장하는 코드가 담긴 file.

[absum 함수는 line plot 그릴 때 쓰는 것.]


*** 예측을 정렬했을 때 순위 == Feature Importance (FI)


```

<br>

# 220222
```
잊으면 아쉬운 변수이름 생성해주는 globals() 함수
for i in sequential_data:
    globals()['{}'.format(i)]

ref.
[5-2] violin plot.ipynb 에 다시 갖다 놓았으니 확인 바람.

# # 참고로 첨 적용했을 때의 file 경로는
# # /project/godshu/2021 usw univ contest/가연/211218 (가연)
```

<br>

# 나중에 220209 txt에 올릴 내용.
```
변경사항 없을 때, 파일 자체 save안하고 staging & commit 하려고 하면
On branch master Your branch is up to date with 'origin/master'.  nothing to commit, working tree clean
이런 에러 남.
```
<br>

# 220210.txt
```
이미 pull한 상태이거나 혹은 pull할 게 없다면
상태를 변경해주고 pull을 해도 아무 충돌이 일어나지 않음. 아마 push는 물론 staging도 전이어서 그랬던 것 같음.
근데 그거 아무 일도 일어나지 않은 것에 놀라서 그만 스테이징도 안하고 push를 했는데
staging 돼있는게 없어서 아무 일도 안일어남.

staging, commit, push 얘네는 늘 세트인가봄.
```
<br>

# 220211
```
Method1, df_all, ReLU, Sign.ipynb 정리해서 putty 업로드 완료.
```
[데이터구조 간단정리??? 참고자료](https://velog.io/@jha0402/Data-structure-%EA%B0%9C%EB%B0%9C%EC%9E%90%EB%9D%BC%EB%A9%B4-%EA%BC%AD-%EC%95%8C%EC%95%84%EC%95%BC-%ED%95%A0-7%EA%B0%80%EC%A7%80-%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0)

<br>

# 220214

[판다스로 파이썬 기본 설정](https://mindscale.kr/course/pandas-basic/options/)

[출력 관련](https://blackhippo.tistory.com/entry/Python-print%EB%AC%B8-%EB%8B%A4%EC%96%91%ED%95%9C-%EC%98%B5%EC%85%98-%EC%A7%80%EC%A0%95%ED%95%98%EA%B8%B0-%ED%83%88%EC%B6%9C%EB%AC%B8%EC%9E%90-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)


<br>

# 220218

Method1 & 2 allfit 안씀. <br>
Method2_scaling 안씀. <br>

<그래프 그리는 과정>
1. absum 함수를 불러옴.
2. 그 다음 PPL feature list 불러옴.
3. PPL feature의 index를 불러옴.
4. absum 적용.

ref. <br>
[4] Method1 = ReLU x Sign x Entropy, graph까지.ipynb <br>

<br>

# 220221

axis는 어렵다.<br>
단순하게 생각해서 데이터의 어떤 부분을 사용할 건지, 그랬을 때의 shape은 어때야 하는지를 고려하면서 하면 쉽다.<br>
(axis로 sum 등을 했을 때, 어떤 shape이 나오는지 확인하면서 계산하면 됨.)<br><br>

    ref.
    [4] Method1 = ReLU x Sign x Entropy, graph까지.ipynb


[Library]<br>
```python
#데이터 분석 라이브러리
import numpy as np
print("numpy version: {}". format(np.__version__))

import pandas as pd

#시각화 라이브러리
import matplotlib as mpl
import matplotlib.pyplot as plt
print("matplotlib version: {}". format(mpl.__version__))

import seaborn as sns
print("seaborn version: {}". format(sns.__version__))


#한글설정
import matplotlib.font_manager as fm

font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
    
# 한글 출력을 위해서 폰트 옵션을 설정합니다.
# "axes.unicode_minus" : 마이너스가 깨질 것을 방지

sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')


#딥러닝 라이브러리
import tensorflow as tf

#기타 라이브러리
import os
import random
import time

#경고 에러 무시
import warnings
warnings.filterwarnings('ignore')
print('-'*50)

#시간 확인을 용이하게 해주는 라이브러리
from tqdm import tqdm

#결과 확인을 용이하게 하기 위한 코드
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# GPU
#https://www.tensorflow.org/guide/gpu#allowing_gpu_memory_growth
#프로세스의 요구량만큼 메모리 사용 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

```
<br>

[Series name 정하기] <br>
Entropy_sr = pd.Series(Entropy_arr, name = 'Entropy') <br>

ref. <br>
https://ponyozzang.tistory.com/624 <br>


[df colnames 바꾸는 방법 2가지] <br>
relu_and_sign.columns = ['feature', 'ReLU', 'Sign'] # [['feature', 'ReLU', 'Sign']]여도 됨. <br>
relu_and_sign.rename(columns = {'diff':'ReLU', 'diff_preds':'Sign'}, inplace = True) <br>




<br>

# 220225

feature 간의 correlation 고려한 예측 해보기.
m1_new = [0->1의 전체 평균] - [1->0의 전체 평균]
preds를 하면 환자 수 만큼 preds값이 나옴.

LSH M1 New 4가지 scoring 함수 부분 항상 실행해서 결과 얻을 것.




[CudnnLSTM](https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/layers/CuDNNLSTM)

>>[CudnnLSTM과 LSTM의 차이](https://stackoverflow.com/questions/49987261/what-is-the-difference-between-cudnnlstm-and-lstm-in-keras)

>>[CudnnLSTM의 activation function](https://stackoverflow.com/questions/52993397/what-is-the-default-activation-function-of-cudnnlstm-in-tensorflow)

[CudnnLSTM 2](https://stackoverflow.com/questions/54559111/different-results-while-training-with-cudnnlstm-compared-to-regular-lstmcell-in)

[CudnnLSTM 3](https://github.com/keras-team/keras/issues/13399)
<br>
<br>
<br>
[keras model](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)

[modeling correlation 고려](https://orange-code.tistory.com/10)




<br>


# 220227

[pandas df](https://opentutorials.org/module/3873/23171)<br><br>
[CuDNNLSTM] <br>
[순환형 신경망 8 - CuDNNGRU & CuDNNLSTM](https://buomsoo-kim.github.io/keras/2019/08/02/Easy-deep-learning-with-Keras-22.md/)

[Tensorflow CuDNN RNN vs 그냥 RNN 비교](https://utto.tistory.com/18)

[Tensorflow와 그 버전과 호환되는 CUDA, cuDNN까지 설치하는 법](https://coding-groot.tistory.com/87)

[07) 패딩(Padding)](https://wikidocs.net/83544)

[가변 길이 입력 시퀀스에 대한 데이터 준비](http://www.nextobe.com/2020/05/14/%EA%B0%80%EB%B3%80-%EA%B8%B8%EC%9D%B4-%EC%9E%85%EB%A0%A5-%EC%8B%9C%ED%80%80%EC%8A%A4%EC%97%90-%EB%8C%80%ED%95%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A4%80%EB%B9%84/)

[07) 케라스(Keras) 훑어보기](https://wikidocs.net/32105)

[4) IMDB 리뷰 감성 분류하기(IMDB Movie Review Sentiment Analysis)](https://wikidocs.net/24586)

[Sequential 모델](https://www.tensorflow.org/guide/keras/sequential_model?hl=ko)


<br>

# 220228

%%time #cell의 맨 위에서 실행해야함.

계층적(stratified) cross validation :<br>
```python
sklearn.model_selection.StratifiedShuffleSplit <br>
```
<br>
[dic to df] <br>
[dic to df](http://daplus.net/python-python-dict%EB%A5%BC-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%94%84%EB%A0%88%EC%9E%84%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98/) <br>
df = pd.DataFrame(dic.items(), columns=['key', 'value']) <br>

[df 원하는 위치에 column 추가](https://steadiness-193.tistory.com/94) <br>
df.insert(0, 'feature', dic.keys())

[today 참고 논문] <br>
[Correlation-based Feature Selection 기법과 RF모델을 활용한 BMI 예측에 관한 연구](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201932365650170&SITE=CLICK) <br>
[Feature Selection이란?](https://subinium.github.io/feature-selection/) <br>

<br><br>

# 백그라운드에서 코드 돌리는 방법
## tmux 사용법
```
우선 test.py를 만들어야함. (tqdm)

vi test.py # vi는 편집창
i #누르고 편집 시작
esc
:wq        # 여기까지 하면 .py파일이 저장된 것. (저장하고 나간다.)
```

## 실행 방법
```py
tmux new -s 지어주고싶은 tmux이름
python test.py                        # 하면 아까 만든 test.py 실행됨. # 실수하면 exit()
exit                                  # 하면 tmux 끝내는 것. tmux가 날라감.


tmux attach -t 실행하고싶은 tmux이름    # 서버에 접속


[파일 열고 편집창 명령어]
ctrl + c          # 코드 실행 강제 종료
q                 # 바꾼 거 없을 때 그냥 나간다.
q!                # 강제종료, 그냥 나가짐.

```
[코드가 잘못됐다면 py file 바꾸면 됨.]


<br>

# 220303

[numpy sorting](https://rfriend.tistory.com/357)
```python
np.sort(arr)
```
<br>

[axes, axis???: axes는 그림판 안의 그림판?; axis는 말 그대로 축이 맞음](https://wikidocs.net/14604) <br>

[그림 여러개] <br>
[fig.add_subplot(rows, cols, python_idx+1)](https://artiiicy.tistory.com/64) <br>

[matplotlib 축 범위 지정 plt.xlim()](https://codetorial.net/matplotlib/axis_range.html) <br>
[ax.set_xlim()](https://www.delftstack.com/ko/howto/matplotlib/how-to-set-limit-for-axes-in-matplotlib/) <br>
[plt.axis('off') # 축 지우기](https://jimmy-ai.tistory.com/18) <br>

[labels](https://wikidocs.net/92089) <br>
[ax.set_xticklabels](https://www.delftstack.com/ko/howto/matplotlib/how-to-rotate-x-axis-tick-label-text-in-matplotlib/#xticks-%25EB%25A0%2588%25EC%259D%25B4%25EB%25B8%2594-%25ED%2585%258D%25EC%258A%25A4%25ED%258A%25B8%25EB%25A5%25BC-%25ED%259A%258C%25EC%25A0%2584%25ED%2595%2598%25EB%258A%2594-ax.set_xticklabels-xlabels-rotation) <br>

[legend 투명도 plt.rcParams["legend.framealpha"]= None이 default, 0.5등 float으로 써주면 됨.](https://kongdols-room.tistory.com/89)<br>
[plt legend ref](https://matplotlib.org/stable/api/legend_api.html)

<br>

# 220307

[visualization] <br>
[레전드 위치](https://codetorial.net/matplotlib/set_legend.html)<br>
[legend 작성 공식 ref](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.axes.Axes.legend.html)<br>
[그래프 색 종류](https://wikidocs.net/92085)<br>
[그래프 위에 text 쓰기](https://seong6496.tistory.com/104)<br>
[text 위치 조정 verticalalignment = <top, bottom, center, baseline>, horizontalalignment = <center, left, right>](https://pyvisuall.tistory.com/52)<br>
[레전드, 복잡해서 안봄](https://zephyrus1111.tistory.com/23)<br>
[파이썬 그래프 마커 종류](https://jiwoncho20213135python.tistory.com/11)<br>
[파이썬 f-string으로 소수점 관리 f'{변수:.2f}'](https://blockdmask.tistory.com/534)<br>

<br>

# 220310
검색어>> 모델 종류를 알고싶으면 종류를 뒤에 붙일 것. <br>
naive bayes 알고리즘 종류 <br>
[scikit learn에서 제공하는 naive bayes 알고리즘 종류](https://docs.likejazz.com/multinomial-naive-bayes/) <br>

scikit learn multinomial naive bayes <br>
[sklearn.naive_bayes.MultinomialNB official doc](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) <br>

bidirectional lstm 설명 <br>
[bidirectional lstm 설명](https://wiserloner.tistory.com/1276) <br>
[tf.keras.layers.Bidirectional official doc](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional) <br>

overleaf 사용법 표 위치 조정 <br>
[overleaf 사용법 표 위치 고정](https://blog.naver.com/PostView.nhn?blogId=ljy5995&logNo=221710034550&redirect=Dlog&widgetTypeCall=true&directAccess=false) <br>

overleaf htp <br>
[overleaf 위치 조정 관련](https://tex.stackexchange.com/questions/35125/how-to-use-the-placement-options-t-h-with-figures) <br>

<br>

# 220311

[LaTex 그림 배치 subplot](http://willkwon.dothome.co.kr/wp-content/uploads/2018/01/lecture3.pdf) <br>
[sns.lineplot(linestyle, marker)](https://wikidocs.net/44312) <br>
[파이썬 텍스트 파일에서 특정 패턴 다음에 오는 텍스트 가져오기; str.index("~~")+1](https://sancs.tistory.com/21)<br>
[파이썬 문자열 찾기; startswith, endswith](https://chanheumkoon.tistory.com/entry/Python-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%AC%B8%EC%9E%90%EC%97%B4-%EC%B0%BE%EA%B8%B0-find-rfind-startswith-endswith) <br>

<br>

# 220317

[EMR과 EHR의 차이](https://www.jobindexworld.com/circle/view/12096) <br>

* 오늘의 issue
정보과학회에 제출할 국내논문 쓰기로 함!!! (~220421?) <br>
<br>

# 220322
[permutation FI의 핵심](https://soohee410.github.io/iml_permutation_importance) <br>
그 feature의 값들을 무작위로 섞어서(permutation) 그 feature를 노이즈로 만드는 것입니다! <br>
우리가 사용한 방법: item을 정해서(item으로 for loop 돌리는 것..) 환자끼리 그 아이템의 one hot을 무작위로 바꾸는 것. <br>
<br>

# 220324

[ensemble learning에 대한 깔끔하고 이해하기 쉬운 설명](https://libertegrace.tistory.com/entry/Classification-2-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5Ensemble-Learning-Voting%EA%B3%BC-Bagging) <br>

[교수님 meeting 때 본 블로그 for adaboosting, adaboost 정리 잘 돼있음.](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/) <br>
<br>

# 220325
[에러메세지 'EarlyStopping' object has no attribute '_implements_train_batch_hooks': 클래스 import시 keras 앞에 tensorflow 붙이면 해결](https://tsy0668.tistory.com/23) <br>

[에러메세지 'module' object is not callable: class가 아닌 모듈만 불러왔을 때 모듈.class로해서 class 써주면 문제없음. 1](https://iksflow.tistory.com/85) <br>

[에러메세지 'module' object is not callable: class가 아닌 모듈만 불러왔을 때 모듈.class로해서 class 써주면 문제없음. 2](https://bewan.tistory.com/62)
<br>

> class가 아닌 모듈만 불러왔을 때 <br>
== <span style='color:red'>from 모듈 import class </span> 안하고 그냥 <span style='color:red'>import 모듈</span>만 했을 때!!! <br>

> module이 class를 포함하는 더 큰 개념.
<br>

# 220326
[검색어: adaboost lstm 학습 안되는 이유] <br>
[혼합 약한 분류기를 이용한 AdaBoost 알고리즘의 성능 개선 방법](https://www.koreascience.kr/article/JAKO200918133142360.pdf)<br>
> 검출기의 성능은 <span style='color:red'>약한 분류기의 성능</span>에 영향을 받는다. 약한 분류기의 성능이 <span style='color:red'>너무 약하거나 강하더라도</span> 검출 성능이 <u>떨어진다.<u>
<br>

[머신러닝 boosting allgorithm](https://hyunlee103.tistory.com/25)<br>

lstm과 약한 분류기? 약한 학습기? <br>
[머신러닝에 관한 부스팅과 AdaBoost](https://soobarkbar.tistory.com/42) <br>
> AdaBoost 돌아가는 과정, 개념 설명
<br>

[머신러닝의 그래디언트 부스팅 알고리즘에 대한 친절한 소개](https://soobarkbar.tistory.com/41) <br>
[Keras에서 LSTM 네트워크에 대한 입력데이터를 재구성하는 방법](https://soobarkbar.tistory.com/46) <br>
[암호화폐 가격 예측을 위한 딥러닝 앙상블 모델링: Deep 4-LSTM Ensemble Model](http://www.koreascience.or.kr/article/JAKO202011236744256.pdf) <br>
> 4. 모델링 결과 및 성능평가 부분이 맘에 듦.

[AdaBoost-LSTM 논문 abstract](https://link.springer.com/chapter/10.1007/978-3-319-93713-7_55) <br>

[AdaBoost 개념, 이론](https://dodonam.tistory.com/332) <br>

[머신러닝 다양한 모델을 결합한 앙상블 학습](https://dev-youngjun.tistory.com/6) <br>
<br>

# 220329
[stacking, boosting, bagging 비교, 괜찮은 참고자료](https://data-matzip.tistory.com/4) <br>
stacking은 overfitting 문제 발생... <br>
그 문제 해결을 위해 CV 방식 도입 <br>

[model ensemble stacking code](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/) <br>


[Blas xGEMM launch failed 에러 해결 (by gpu setting)](https://skeo131.tistory.com/202) <br>

* os.makedirs(폴더명)이랑, to_categorical(변수명)은 또 하면 에러남.
makedirs의 경우 동일한 이름의 폴더를 또 생성하려고 할 때, <br>
to_categorical은 동일한 변수를 또 카테고라이즈하려고 할 때. <br>
<br>
to_categorical()에러는 다음과 같음. <br>
ValueError: Shapes (...) and (...) are incompatible <br>
<br>

* verbose뜻: 말 수가 많은
verbose = 0 : 과정 끔 <br>
verbose = 1 : 과정 보여줌 <br>
verbose = 2 : 과정 함축해서 보여줌. <br>
[verbose ref](https://www.inflearn.com/questions/208344) <br>
<br>

* patience @ early stop
loss나 acc val이 가장 좋은 값이 나왔을 때 patience만큼 더 해보고도 안바뀌면 early stop해라. <br>
<br>

* 앞에 있는 loss, acc는 training set`s
* 뒤에 있는 val_loss, val_acc는 validation set`s, 얘네로 monitoring
<br>

* get_params()

* lambda에 들어가는 (data에 적용하는) model이 이미 실행된 함수였던게 오늘의 가장 큰 문제였다.
get_model() 와 get_model차이... <br>
lambda에는 이미 실행된 함수가 들어가면 안됨. <br>
get_model()를 한 모델을 lambda에 집어넣어 버리면 함수가 이미 실행된게 들어가서 문제였던 것... <br>
<br>

* hp 바꿔가면서 모델 실행해올 것.

* Permutation FI 짜보기...

* classifier 횟수(n_estimator)는 못 세는지..? 

# 220331
* ML, DL 관련 알쓸신팁.
batch_size = 늘릴 수 있을만큼늘리되 gpu가 감당할 수 있을만큼만 해라. <br>
진행바 왼쪽이 input size / batch의 반올림한 값. <br>

hyper parameter를 튜닝하면서 학습시키는게 맞음. 한경훈 교수님 영상 [10강. 하이퍼파라미터](https://youtu.be/iVI51L0h3LQ?t=107) 으로 확인함. <br>

early stop 할 때는 epoch 수 상관 없음. <br>
<br>
[model.fit() parameters](https://keras.io/api/models/model_training_apis/) <br>
<br>



> * [교수님 meeting log] <br><br>
sample weight만 반영되면 됨. <br>
방법1. resampling (if문 어쩌고 했던거) <br>
방법2. loss function 구할 때 weight 같이 곱해줌. (가중치) <br>
더 많이 틀린 애를 집중적으로 학습하는 방향으로  <br>
<br>
sample weight 개수(len) 곱하라고 하신 이유?????? <br>
sample_weight 총 합이 원래는 1이되게 하는데 그걸 7727이 되게...... sample_weight가 너무 작아서 문제였으니까 그거를 커지게... <br>
<br>
지금은 training set과 validation set에서 서로 적용되는 loss func 식이 달라서 문제임. <br>
<br>
binary crossentropy 대신 weighted binary crossentropy 가능한지 <br>
<br>
원래는 만들어서 쓰는게 맞고, 만드는게 어려우면 <br>
validation data를 우리가 fit할 때 만들어주면 됨. <br>
<br>
fit하는 class에 <br>
kwarg에 validation data 넣어주고... <br>
train과 validation으로 나눠주기. <br>
sample_weight 수정 다시해야함. <br>
<br>
train .... validation 나눌 때 index만 뽑아주면 됨. index가 있으면 idx로 나눌 수 있음. <br>
<br>
sample_weight도 train val 각각의 sw로 나눠줘야. <br>
<br>
정규화는 training sw만 하면 됨. <br>
len, sw가 다르니까..? 같이 하면 안된다고...? <br>
<br>
val set은 어떻게 만들어야 하냐면,,, <br>
val x, y는 우리가 한 번 더 샘플링 해줘야함. 위에서 resampling 했던 것처럼. <br>
<br>
new_val_x, new_val_y <br>

<br>

>> ref <br>
[hyperparameter tuning] <br>
(https://velog.io/@emseoyk/%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D)
<br>
(https://seamless.tistory.com/34)
<br><br>
[.fit] <br>
[.fit1](https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)
<br>
[.fit2](https://keras.io/ko/getting-started/sequential-model-guide/)
<br>
[.fit3](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
<br>
[.fit4](https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046)
<br>
[.fit5](https://www.justintodata.com/hyperparameter-tuning-with-python-keras-guide/)
<br><br>
[model.fit!!!!!](https://keras.io/api/models/model_training_apis/)
<br><br>
[weighted binary crossentropy](https://data-newbie.tistory.com/645)
<br>
<br>

batch size 크게 하면 patience도 크게 해야.... <br>
<br>

training acc가 안오르면 학습이 안되는 것임.] <br>
그냥 acc가 training acc <br>
val_acc가 validation acc <br>

==> training val이 계속 똑같거나 머무르면 학습이 안되는 것임. <br>
<br>

# 220412 논문 쓰는 중...
* LSTM 정의 <br>
[LSTM 개발자의 논문](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf) <br>
[LSTM에 대하여](http://www.incodom.kr/LSTM) <br>

* LSTM에 learning rate 설정하는 게 없는가에 대하여 <br>
[[ Python ] Neural Network의 적당한 구조와 hyperparameter 찾는 방법](https://data-newbie.tistory.com/255) <br>
[하이퍼파라미터 튜닝](https://sswwd.tistory.com/37) <br>

# 220419
Push 에러,,,,

# 220428
[word to pdf 이미지 화질 개선 방법) 주피터 이미지 관련, 확장자.svg 혹은 .eps로 저장해서 사용](http://daplus.net/python-%EB%A7%A4%EC%9A%B0-%EB%86%92%EC%9D%80-%ED%92%88%EC%A7%88%EB%A1%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%A0%80%EC%9E%A5/)
<br>
[feature importance의 일환으로 shap이 있음.](https://todayisbetterthanyesterday.tistory.com/56) <br>


# 220707
New 방법 추가 (기존 방법 수정된 게 아니고 새로운 방법 추가된 것임.) <br>
방법 상세: 특이마커 고르기 <br>

- shape 바뀐 X에 대한 뉴 코드 <br>
/project/LSH/** 해외_Journal of Biomedical Informatics/03 Method2 - 사망, 생존 TOP10 & 그래프 <br>
/project/kelly/03 Method2 - 특이 마커 TOP10 & 그래프 <br>
<br>

- [/project/kelly/03 Method2 - 특이 마커 TOP10 & 그래프] 작성에 참고한 코드 <br>
/project/kelly/[7-2] method2 (결과 4 역전마커) # 역전마커 코드에 참고 (아마 원근 코드 모두에 참고할 것 같음.) <br>
/project/kelly/[7-1] 결과2,3 method1_entropy ; (E(all to 1) - E(all to 0))*entropy & method1_부록 # entropy 들어있는 df가 어떻게 생겼는지 확인 차 참고 <br>
<br>

# 220710 issue
기껏 전처리 코드 다 치고 싹 정리해놨더니 노션에 옮기는 과정에서 공중분해됨. 어디로 사라진 것인지 찾을 수가 없었음. <br>
<br>


# 220715
[검색어: 파이썬 시계열 모델링 기초 코드 설명] <br>
[코드, 예제, LSTM 설명 페이지 추천](https://rk1993.tistory.com/entry/Python%EC%8B%9C%EA%B3%84%EC%97%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%AA%A8%EB%8D%B8%EB%A7%81-%EA%B8%B0%EC%B4%88%EB%B2%84%EC%A0%84) <br>
[LSTM 자체에 대한 설명](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr) <br>
<br>

[검색어: 파이썬 시계열 모델링 분석 LSTM예제] <br>
[LSTM에 대한 이해가 되는 설명, notion 설명에 참고할 예정](https://hhhh88.tistory.com/38) <br>
<br>

[ARIMA 모델에 대한 설명, youtube 영상 추천](https://taek98.tistory.com/44) <br>
[ARIMA 모델에 대한 설명, 코드, 예제](https://happy-chipmunk.tistory.com/entry/Time-Series-Forecasting-2-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%8B%9C%EA%B3%84%EC%97%B4-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B8%EB%A7%81-ARIMA-Auto-regressive-integrated-moving-average-Auto-ARIMA) <br>
<br>


# 220716 notion 정리 중 ...💛
refs for notion<br>
[LSTM으로 시계열 예측하기: 절차](https://hhhh88.tistory.com/38) <br>

[검색어: keras LSTM X shape] <br>
[keras - Keras를 통한 LSTM의 구현](https://3months.tistory.com/168) <br>
[Keras LSTM 입력 포맷의 이해 Understanding Input shapes in LSTM | Keras](https://swlock.blogspot.com/2019/04/keras-lstm-understanding-input-and.html) <br>
<br>

[[케라스] 무작정 튜토리얼 11 - LSTM(feat.RNN) 구현하기](https://ebbnflow.tistory.com/135) <br>


[[공식문서] keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) <br>
[[공식문서] LSTM layer, LSTM class](https://keras.io/api/layers/recurrent_layers/lstm/) <br>

[시계열수치입력 수치예측 모델 레시피](https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/) <br>



[검색어: keras lstm dropout] <br>
[RNN에서의 Dropout](https://wjddyd66.github.io/keras/Keras(4-2)/) <br>
💛💛 RNN Network에서 dropout을 Sequential()에 층을 쌓는 방식이 아닌 LSTM Layer 안에서의 option으로 사용해야하는 이유 ; 과거의 중요 정보까지는 잃지 않기 위해. <br>
<br>


[검색어: keras lstm stateful return_sequences] <br>
[Keras를 사용한 반복적 인 신경망 (RNN)](https://www.tensorflow.org/guide/keras/rnn?hl=ko) <br>
<br>


## 221205
[statistical Feature Selection] <br>
[Pneumonia identification using statistical feature selection](https://academic.oup.com/jamia/article/19/5/817/718233) <br>
[Statistical interpretation of machine learning-based feature importance scores for biomarker discovery](https://academic.oup.com/bioinformatics/article/28/13/1766/234473) <br>
[Interpretable and explainable AI (XAI) model for spatial drought prediction](https://www.sciencedirect.com/science/article/abs/pii/S0048969721048725) <br>
[Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)](https://ieeexplore.ieee.org/abstract/document/8466590) <br>
