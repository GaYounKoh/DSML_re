# DSML_re
DSML_restart

```
나중에 220209 txt에 올릴 내용.
변경사항 없을 때, 파일 자체 save안하고 staging & commit 하려고 하면
On branch master Your branch is up to date with 'origin/master'.  nothing to commit, working tree clean
이런 에러 남.
```

```
220210.txt
이미 pull한 상태이거나 혹은 pull할 게 없다면
상태를 변경해주고 pull을 해도 아무 충돌이 일어나지 않음. 아마 push는 물론 staging도 전이어서 그랬던 것 같음.
근데 그거 아무 일도 일어나지 않은 것에 놀라서 그만 스테이징도 안하고 push를 했는데
staging 돼있는게 없어서 아무 일도 안일어남.

staging, commit, push 얘네는 늘 세트인가봄.
```

```
220211
Method1, df_all, ReLU, Sign.ipynb 정리해서 putty 업로드 완료.
```
[데이터구조 간단정리??? 참고자료](https://velog.io/@jha0402/Data-structure-%EA%B0%9C%EB%B0%9C%EC%9E%90%EB%9D%BC%EB%A9%B4-%EA%BC%AD-%EC%95%8C%EC%95%84%EC%95%BC-%ED%95%A0-7%EA%B0%80%EC%A7%80-%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0)


```
220214
판다스로 파이썬 기본 설정
https://mindscale.kr/course/pandas-basic/options/

출력 관련
https://blackhippo.tistory.com/entry/Python-print%EB%AC%B8-%EB%8B%A4%EC%96%91%ED%95%9C-%EC%98%B5%EC%85%98-%EC%A7%80%EC%A0%95%ED%95%98%EA%B8%B0-%ED%83%88%EC%B6%9C%EB%AC%B8%EC%9E%90-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0

```

```
220218
Method1 & 2 allfit 안씀.
Method2_scaling 안씀.

<그래프 그리는 과정>
1. absum 함수를 불러옴.
2. 그 다음 PPL feature list 불러옴.
3. PPL feature의 index를 불러옴.
4. absum 적용.

ref.
[4] Method1 = ReLU x Sign x Entropy, graph까지.ipynb
```

```
220221
axis는 어렵다.
단순하게 생각해서 데이터의 어떤 부분을 사용할 건지, 그랬을 때의 shape은 어때야 하는지를 고려하면서 하면 쉽다.
(axis로 sum 등을 했을 때, 어떤 shape이 나오는지 확인하면서 계산하면 됨.)

ref.
[4] Method1 = ReLU x Sign x Entropy, graph까지.ipynb
```