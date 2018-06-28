ML_Summary
-----
Machine Learning / Andrew Ng / StanfordUniv

1. Introduction
```
우리는 웹 검색 엔진에서도 머신러닝을 평소에도 사용하고있습니다. 이메일의 스팸필터도 마찬가지입니다.
이러한 머신러닝의 최종적인 목표는 인간의 뇌와 유사하게 만드는 것입니다.
```

1.1 이러한 머신러닝이 왜 사용되고 있을까요?
```
기계가 지능을 가지기 원했고, A에서 B로 갈때 어떻게 해야 빠리 갈지 등 기계를 학습시킴으로써
컴퓨터가 할 수 있는 일들이 다양한 분야를 걸쳐 상상이상으로 많습니다. 

그 예로는 
1.1.1 데이터베이스 마이닝입니다. 웹 클릭 기록, 의료 기록, 유전자 분석, 공학분야 등 우리가 수집할 수 있는 데이터 량이 많아지면서 활용성이 높아졌습니다.
1.1.2. 수동적으로 프로그래밍 할 수 없을 때 입니다. 헬리콥터가 비행하도록 하는 프로그래밍은 굉장히 어렵습니다. 이러한 동작을 머신러닝을 통해 자동 프로그래밍 할 수 있습니다. 손 글씨를 읽는 경우도 있습니다. 자연언어처리기법, 컴퓨터비젼도 마찬가지입니다.
1.1.3. 스스로 프로그래밍하는 프로그램, 아마존, 넷플릭스의 추천 서비스가 그 예입니다. 수십만의 사용자를 다루는데 매번 다른 프로그램을 적용하긴 어렵습니다.
```

1.2 머신러닝이란 ?
```
Samuel : machine learning as the field of study that gives computers the ability to learn without being explicitly programmed. 
머신러닝은 컴퓨터에게 배우는 능력을 주는 것이다.
Mitchell : a computer program is said to learn from experience E, with respect to some task T, and some performance measure P, if its performance on T as measured by P improves with experience E.
컴퓨터 프로그램이 작업 T와 관련하여 경험 E로부터 배우고, 성능 측정 P가 있을 때, P로 측정한 성능이 경험 E로 향상되는 것을 말한다.
```

1.2 What is Machine Learning?
```
Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.
Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
Example: playing checkers.
  E = the experience of playing many games of checkers
  T = the task of playing checkers.
  P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:
Supervised learning and Unsupervised learning.
```

1.3 지도학습이란?
```
데이터(올바른 답)을 모두 제공하고, 더 다양한 올바른 답을 얻어올 수 있도록 하는 알고리즘.
regression problem : 연속적인 결과를 예측하는 것.
```
