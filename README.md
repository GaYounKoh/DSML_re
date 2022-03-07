# DSML_re
DSML_restart
<br><br>
<i><b><q>중요한 내용</q></b>은 앞으로 뺌</i>
<br>

# 질문
<<220227>> <br>
~~preds2_mean_lst.append(np.mean(preds2))~~ <br>
~~<span style = 'color : red'>예측 for문에서 df 만들 때 pred에 이미 평균을 냈는데 평균을 왜 또 내..?</span>~~ <br>
==> for문이 feature별로 돌아가면 예측은 환자수만큼 나와.
그니까 feature 별로 하나의 값으로 보려면 나온 값들의 평균으로 보는게 가장 좋음.

<br>

# 안 보는 자료
6-1, 6-2는 각각 조합과 CuDNNLSTM 관련 내용이므로 지금은 일단 버림
<br><br>


# cudnnLSTM
[cudnnLSTM](https://buomsoo-kim.github.io/keras/2019/08/02/Easy-deep-learning-with-Keras-22.md/)
<br><br><br>

# 220223
```
정리
7727: 폐렴환자 수, 근데 이제 abnormal(x matrix) sum이 0인 아무 의미 없는 환자들을 제외한.
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

Method1 & 2 allfit 안씀.<br>
Method2_scaling 안씀.<br>

<그래프 그리는 과정>
1. absum 함수를 불러옴.
2. 그 다음 PPL feature list 불러옴.
3. PPL feature의 index를 불러옴.
4. absum 적용.

ref.
[4] Method1 = ReLU x Sign x Entropy, graph까지.ipynb

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

[Series name 정하기]
Entropy_sr = pd.Series(Entropy_arr, name = 'Entropy')

ref.
https://ponyozzang.tistory.com/624


[df colnames 바꾸는 방법 2가지]
relu_and_sign.columns = ['feature', 'ReLU', 'Sign'] # [['feature', 'ReLU', 'Sign']]여도 됨.
relu_and_sign.rename(columns = {'diff':'ReLU', 'diff_preds':'Sign'}, inplace = True)




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

계층적(stratified) cross validation : sklearn.model_selection.StratifiedShuffleSplit

[dic to df] <br>
[dic to df](http://daplus.net/python-python-dict%EB%A5%BC-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%94%84%EB%A0%88%EC%9E%84%EC%9C%BC%EB%A1%9C-%EB%B3%80%ED%99%98/) <br>
df = pd.DataFrame(dic.items(), columns=['key', 'value'])

[df 원하는 위치에 column 추가](https://steadiness-193.tistory.com/94)
df.insert(0, 'feature', dic.keys())

[today 참고 논문]<br>
[Correlation-based Feature Selection 기법과 RF모델을 활용한 BMI 예측에 관한 연구](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201932365650170&SITE=CLICK)<br>
[Feature Selection이란?](https://subinium.github.io/feature-selection/)<br>

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