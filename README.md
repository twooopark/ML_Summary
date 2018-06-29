ML_Summary
-----
Machine Learning / Andrew Ng / StanfordUniv

## WEEK 1
1. 머신 러닝의 개념
```
머신러닝은 컴퓨터에게 배우는 능력을 주는 것 입니다.
컴퓨터 프로그램이 작업 T와 관련하여 경험 E로부터 배우고, 성능 측정 P가 있을 때, P로 측정한 성능이 경험 E로 향상되는 것을 말합니다.
우리는 웹 검색 엔진에서도 머신러닝을 평소에도 사용하고있습니다. 이메일의 스팸필터도 마찬가지입니다.
기계가 지능을 가지기 원했고, A에서 B로 갈때 어떻게 해야 빠리 갈지 등 기계를 학습시킴으로써
컴퓨터가 할 수 있는 일들이 다양한 분야를 걸쳐 상상이상으로 많습니다. 
이러한 머신러닝의 최종적인 목표는 인간의 뇌와 유사하게 만드는 것입니다.
```

1. What is Machine Learning?
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

1.1 활용 예시
 * 데이터베이스 마이닝입니다. 웹 클릭 기록, 의료 기록, 유전자 분석, 공학분야 등 
우리가 수집할 수 있는 데이터 량이 많아지면서 활용성이 높아졌습니다.
 * 수동적으로 프로그래밍 할 수 없을 때 입니다. 헬리콥터가 비행하도록 하는 프로그래밍은 굉장히 어렵습니다. 
이러한 동작을 머신러닝을 통해 자동 프로그래밍 할 수 있습니다. 
손 글씨를 읽는 경우도 있습니다. 자연 언어 처리기법, 컴퓨터 비젼도 마찬가지입니다.
 * 스스로 프로그래밍하는 프로그램, 아마존, 넷플릭스의 추천 서비스가 그 예입니다. 
수십만의 사용자를 다루는데 매번 다른 프로그램을 적용하긴 어렵습니다.


1.2 지도학습의 개념
```
입 출력 데이터를 제공하고 제공된 데이터를 통해, 다양한 다른 입력에 대한 출력 값을 예측 할 수 있도록 하는 알고리즘입니다. 
여기서 출력 값의 형태에 따라 아래와 같이 분류합니다.
regression problem : 연속적인 결과를 예측
classification : {0, 1} 처럼 집합으로 분류된 이산적인 결과를 예측
```

1.2 Supervised Learning
```
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

Example 1:
Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.
We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

Example 2:
(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture
(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.
```

1.2.1 regression problem 
```
  입력(평수 크기)와 출력(가격)은 어떤 관계가 있는 지 궁금했고, 데이터들을 입력했습니다.
  이를 통해 이 관계들이 어떤 형태를 이루는지 그래프로 보여졌습니다.. 
  입력(알고싶은 평수)에 따른 출력 결과(가격)를 알 수 있게 되었습니다.
``` 
<img src="https://github.com/twooopark/ML_Summary/blob/master/1-1_RegressionProblem.JPG" width="600px" height="300px" />

1.2.2 classification
```
  Tumor(종양)의 크기에 따라, Malignant(악성)인지 아닌지 알 수 있을까요? 데이터들을 입력했습니다. 
  사이즈가 작으면 0, 크면 1이라는 결과를 얻었습니다. 
  일정 이상 크기면, 악성 종양일 확률이 높다는 것을 알 수 있게 되었습니다.
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/1-2_Classification.JPG" width="600px" height="300px" />

1.3.2 classification
``` 
 Tumor Size와 Age의 상관관계 1.3.1의 x, y축 2차원 그래프를 사용했습니다. 
 범위에 따른 결과 값(예측에 사용될 값)을 얻을 수 있습니다. 
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/1-3_Classification.JPG" width="600px" height="300px" />




1.3 비지도 학습의 개념
```
우리가 어떤 결과를 얻어야 할지 모를 때, 사용되는 알고리즘. 데이터들의 관계를 통해 결과를 얻습니다.
예시를 들어보면, 구글 검색결과를 보면, 군집화(Clustering) 되어있습니다.
기계는 오로지 다양한 데이터만을 제공받을 뿐입니다. 비지도 학습은 문제의 정확한 답을 제공하지 않기 때문에 비지도 학습이라고 합니다.
활용 예시는 다음과 같습니다. 대규모 클러스터를 조직하는 일, SNS를 통한 인간관계 분석, 고객에 따른 시장 세분화(고객 정보를 가지고) 

현업(실리콘밸리)에서는 octave같은 알고리즘 툴을 사용해서 우선 적용시켜보고, 제대로 원하는 결과가 나온다면
그 후에 우리가 수정해야할 사항들을 C++, JAVA, Python 등으로 수정해서 만듭니다. 처음부터 다 만든다면 아주 많은 시간이 낭비된다고 합니다.

옥타브는 수치해석용 자유 컴퓨터 소프트웨어입니다. 비슷한 프로그램들은 아래 주소에 있습니다.
https://en.wikipedia.org/wiki/List_of_numerical_analysis_software#Numerical_software_packages
```

1.3 Unsupervised Learning
```
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.
We can derive this structure by clustering the data based on relationships among the variables in the data.
With unsupervised learning there is no feedback based on the prediction results.

Example:
Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).
```




