1. 왜 rf를 사용했는지 <br>
feature selection 제공하는 모듈 중 가장 근본이기 때문 <br>

2. 그래서 발견한 폐렴 사망 원인에 대한 새로운 단서는? <br>
보통 폐렴은 합병증으로 진행될 때 <u>사망률</u>이 증가. <br>
만성질환자의 경우에는 항생제 내성균에 의한 폐렴의 <u>발생</u> 위험도가 높음.

[폐렴 사망 원인](https://www.yuhan.co.kr/Mobile/Introduce/Health/?Cateid=290&mode=view&idx=36036&ref=36010&p=1&sm=-1&listUrl=%2FMobile%2FIntroduce%2FHealth%2FSearch%2Findex%2Easp%3FCateid%3D290) <br>

3. LSTM과 RNN의 차이 <br>
장기 기억 문제 개선 <br>
LSTM : RNN에 중요하지 않은 정보는 잊을 수 있는 forget 게이트가 추가된 형태.<br>
<br>
완전히 잊는 것이 아닌 weight를 부과하여 덜 기억하도록 하는 것. <br>

4. 손실함수로 bce 사용한 이유? -는 질문거리가 안될 것 같음. <br>
<손실함수의 종류>
회귀 모델: MSE, MAE, RMSE.
분류 모델: Binary cross entropy, Categorical cross entropy. <br>
<br>
일반적으로 신경망 학습에서는 MSE와 교차 엔트로피 오차를 사용. <br>
<br>
이진분류기 훈련시 binary cross entropy 손실함수가 적절. <br>
<br>

> 손실함수는 <u>예측값과 실제값이 같으면 0이 되는 특성</u>을 갖고있어야 함. <br>
이진 분류기의 경우 예측값이 0과 1 사이의 확률값으로 나오며, 0또는 1에 가까울 수록 둘 중 한 클래스에 가깝다는 것. <br>

<br>

> categorical cross entropy 는 분류해야할 클래스가 3개 이상인 경우에 사용. <br>
라벨이 one hot encoding의 형태로 제공될 때 주로 사용.
주로 softmax 활성함수와 함께 쓰여서 softmax activation function이라고도 불림. <br>

> sparse categorical cross entropy 는 분류해야할 클래스가 3개 이상이며, 라벨이 정수의 형태로 제공될 때 주로 사용. <br>


[손실함수, entropy식, bce식](https://velog.io/@yuns_u/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98-%EA%B0%84%EB%9E%B5-%EC%A0%95%EB%A6%AC) <br>
[cross entropy의 이해: 정보이론과의 관계, 엔트로피란?](https://3months.tistory.com/436) <br>
[크로스 엔트로피 손실 개요, 이진분류의 cross entropy식](https://wandb.ai/wandb_fc/korean/reports/---VmlldzoxNDI4NDUx) <br>

```
l = -(ylog(p) + (1-y)log(1-p))

p : 예측확률
y : 지표(label, 이진 분류의 경우 0 또는 1)
```
<br>

5. 최적화 알고리즘으로 adam 사용한 이유? <br>
최적화 알고리즘은 효율성(훈련 속도 관련)을 높이기 위한 도구 <br>
Adam = 모멘텀 + RMSprop <br>
딥러닝에서 가장 흔히 사용되는 최적화 알고리즘. <br>
Momentum의 v, RMSprop의 S 지정. <br>
Momentum의 Bias correction 실시 <br>
Momentum과 RMSprop의 가중치 업데이트 방식을 모두 사용해 가중치 업데이트 <br>
<br>


각 파라미터마다 다른 크기의 업데이트를 적용하는 방법. <br>
과거 그래디언트 정보의 영향력 감소 위해 decaying average of gradients 사용. <br>
1차 모멘트 : 모평균 <br>
2차 모멘트 : 1차 모멘트(모평균)와 E(X^2)로 구한 모분산 <br>

표본평균과 표본제곱의 평균으로 1,2차 모멘트를 추정해야함. <br>

-> 여기서 추정이 들어가서 Adaptive Moment Estimation인 것. <br>

m_t = gradient의 1차 moment에 대한 추정치 <br>
v_t = 2차 moment에 대한 추정치 <br>

```
m_t = B_1 * m_t-1 + (1-B_1) * g_t
v_t = B_2 * v_t-1 + (1-B_2) * g_t^2
```

m_t와 v_t의 초기값을 0벡터로 주면, 학습초기에 가중치들이 <u>0으로 편향되는 경향</u>이 있음. <br>
특히 decay rate가 작으면, 즉 B_1과 B_2가 1에 가까우면 편향이 더 심해짐. <br>
편향을 잡아주기 위해 bias-corrected를 계산. <br>

```
# bias corrected 식
m_t_hat = m_t / (1-B_1^t)
v_t_hat = v_t / (1-B_2^t)
```

최종 업데이트 식
```
theta_t+1 = theta - (eta / ((v_t_hat)^(1/2) + error)) * m_t_hat
```

[딥러닝 최적화 알고리즘](https://velog.io/@minjung-s/Optimization-Algorithm) <br>
[딥러닝 최적화 알고리즘 알고쓰기, 딥러닝 옵티마이저 총정리](https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html) - 글쓴이가 여태껏 어떤 근거도 없이 adam 쓰다가 문득sgd 써봤는데 성능이 향상된 경험을 하게되면서 정리해봐야겠다고 마음 먹었다고 함. <br>

6. 사망 마커 점수 계산 식의 의미
entropy에 대해......
밑 = 2인 log 사용
entropy : 불확실성
예측하기 힘든 일에서 더 큰 엔트로피. (귀한 정보)
작은 엔트로피 => 높은 확률, 자주 발생 (흔한 정보)

[통계 엔트로피](https://m.blog.naver.com/qbxlvnf11/221535406312) <br>