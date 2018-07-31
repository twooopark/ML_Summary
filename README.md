ML_Summary
-----
```
참고한 자료들 
Machine Learning / Andrew Ng / StanfordUniv
https://www.coursera.org/learn/machine-learning/

구글 머신러닝 강좌
https://developers.google.com/machine-learning/crash-course/?utm_source=google-ai&utm_medium=card-image&utm_campaign=training-hub&utm_content=ml-crash-course
```
### 0. 기초 학습
  - 독립변수 : 결과 값에 영향을 주는 변수들
  - 종속변수 : 독립 변수들을 통해 예측되는 결과 값
  - 결측값 : 값이 없는 경우. 컬럼의 평균, 또는 중위값을 넣는다. (평균이 제일 적합함)
  
  
```
### 알고리즘 선정
Photo 과정
passed lot(정상) - 1000개 가량
hold lot(멈춤) 
  - 진성 : 정말 문제 있는 것
  - 가성 : 잘못 판단
  - 한달 한 기계, 진성 30건 가성 100건
  - 기계 hold 사건(X=[진성,가성], A=[30,100], B=[60, 150], C=[50,100])
  - 이 문제는 분류 문제이다. 정상, 가성의 경우를 하나로, 진성의 경우를 또 하나로 (정상, 가성: 0(Label), 진성 : 1(label)
  - 분류 문제를 해결할 만한 알고리즘은 뭐가 있을까 ? Logistic, SVM, Decision Tree, SVDD
    - Logistic : 0, 1 각각 확률이 나옴
    - SVM : 분류할 초평면을 구함 ( 마진을 이용해(Hyper parameter?) 분류가 가장 잘 되는 선형을 구해 분류함)
    - DT :
    - SVDD : 학습할 때 한가지 라벨만 학습하도록 하는 것. 잘 쓰이진 않는다. A와 B가 명확히 구분될 때 쓰임.
    - CNN : CNN도 해봤지만, 데이터가 너무 적어서 오버피팅 발생
  - python: dt.describe R: dt.summary() 로 평균값 최대값 최소값 결측치를 구한다.
  - 결측치를 제외할 때, 행을 지워버리면, 데이터 량이 줄어들어 정확도가 떨어질 수 있다 > 평균, 0 값을 대신 넣어 해결하자.
  
  * 과제 : 데이터 1, 데이터 2, 데이터 3을 산점도로 출력, z-score를 이용해 표준화해서 산점도 3개를 비교하기 쉽게 합쳐서 결과를 출력하라(색상은 다르게) (데이터는 임의로 한다.)
  * 과제 : PCA(Principal Component Analysis)에 대해 알아오시오.(비지도학습)
  
  검증은 어떻게 할 것인가? : 교차검증(CV), c-ford
  
``` 

### 1. 머신 러닝의 개념
  
```
머신러닝은 컴퓨터에게 배우는 능력을 주는 것 입니다.
컴퓨터 프로그램이 작업 T와 관련하여 경험 E로부터 배우고, 성능 측정 P가 있을 때, P로 측정한 성능이 경험 E로 향상되는 것을 말합니다.
우리는 웹 검색 엔진에서도 머신러닝을 평소에도 사용하고있습니다. 이메일의 스팸필터도 마찬가지입니다.
기계가 지능을 가지기 원했고, A에서 B로 갈때 어떻게 해야 빠리 갈지 등 기계를 학습시킴으로써
컴퓨터가 할 수 있는 일들이 다양한 분야를 걸쳐 상상이상으로 많습니다. 
이러한 머신러닝의 최종적인 목표는 인간의 뇌와 유사하게 만드는 것입니다.
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


1.3 Linear regression with one variable

1.3.1 Model representaion
```
x는 입력, y는 출력, h는 가설입니다.
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/2-1_htheta_1.JPG" width="480px" height="270px" />

1.3.2 Cost function
```
제공된 데이터 X들에 대해서 가설의 결과 값H과, 실제 결과 값Y의 차이를 제곱한 값들의 평균 값을 J라 합니다.
J가 최소가 되는 세타0(θ0), 세타1(θ1)을 찾는 것이 우리의 목표입니다. 오차가 제일 작다는 뜻이기 때문입니다.
제곱을 하는 이유는 두 가지입니다.
1. 오차가 음수일 경우 양수로 통일
2. 오차가 클 수록 더 많은 페널티 부여
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/2-1_htheta_2.JPG" width="480px" height="270px" />

```
간단한 데이터, y=x인 선형 그래프로 예시를 들어보자면, 아래 그림과 같습니다.
우리가 확인해보고자 하는 가설 h는 세타0 = 0, 세타1 = 1 일 때 입니다.
모든 X에 대한, 가설H와 결과Y의 오차가 모두 0이기 때문에, J는 0임을 구했습니다.
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/2-1_htheta_3.JPG" width="480px" height="270px" />

```
만일, 세타1 = 0.5라고 한다면, 오차 값이 x는 1일때 0.5, 2일때 1, 3일때 1.5로 J는 1/6*(3.5)..약0.58 가 됩니다.
세타 = 1이면, J는 약 2.3이 되며, J의 그래프가 이차함수의 그래프와 유사한 모양이 됩니다.

```
<img src="https://github.com/twooopark/ML_Summary/blob/master/2-1_htheta_4.JPG" width="480px" height="270px" />



1.3.3 Gradient descent
```
기울기 하강법은 J값을 가지고 J의 최소값을 구하는 방법입니다.
세타0(θ0), 세타1(θ1)에서 시작해서 세타값들을 변경하면서,
convergence(수렴)까지, 즉, J를 최소값이 될 때 까지 줄여나갑니다.
아래와 같이 기울기가 음수이든 양수이든 최소값을 향해 가는 것을 알 수 있습니다.
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/3-1_G_descent_1.JPG" width="480px" height="270px" />


* Learning rate(α)  
```
기울기 하강법에 적용되는 알파값(α:학습률)을 이용해 J값의 변화를 얼마나 할지 조정합니다.
이 학습률(α)은 너무 작으면, 최소값을 찾는데 오래걸리고
너무 크면, 최소값을 지나쳐 탐색하면서, 결국엔 값이 발산할 수 있습니다.
그리고 J가 줄어들며 수렴을 향해가기 때문에, 이 α값은 바꿀 필요가 없습니다.
그럼 이 α는 어떻게 선정해야 할까?....
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/3-1_G_descent_2.JPG" width="480px" height="270px" />

1.3.4 Gradient descent for linear regression
```
기울기 하강법과 선형 회귀 공식입니다.
선형회귀모델에 기울기 하강법을 적용시킴으로써
회귀 모델에서 만들어진 가설 H값의 오차J가 최소가 되는 값을 
기울기 하강법으로 세타를 줄여나가는 방식으로 찾아냄을 알 수 있습니다.
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/3-1_G_descent_3.JPG" width="480px" height="270px" />

1.3.5 multivariate linear regression
```
위에서 설명한 선형 회귀는 단일선형회귀 입니다. 집의 가격(Y)를 구할 때, 집의 크기(X) 1가지의 조건만 고려했기때문입니다.
다중(다변량)선형회귀는 집의 층수, 방 수 등의 여러가지의 조건을 고려하는 선형 회귀입니다차원
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/3-1_G_descent_4.JPG" width="480px" height="270px" />
<img src="https://github.com/twooopark/ML_Summary/blob/master/3-1_G_descent_5.JPG" width="480px" height="270px" />

1.3.6 Feature Scaling
- 이번엔 기울기 하강법을 빠르게 수행할 수 있도록 하려면, 조건 X와 α의 값을 어떻게 조정해야 하는지 알아보겠습니다.
```
만약, 이 cost function에 gradient descent를 적용한다면, gradient는 오랜 시간 동안 앞 뒤로 진동하며, 엄청난 시간이 지나고 나서야 마침내 최소값에 도달할 것입니다. 실제로, 등고선이 극단적인 경우를 생각해보면 즉 엄청 얇은, 얇고 긴 등고선이라면, 
그리고 더욱 극단적으로 과장한다면, gradient descent는 훨씬 더 많은 시간을 소요하며, 구불구불하게 가다가, 오랜 시간이 지나서야 최소값을 찾을 수 있습니다. 이 때, 유용한 방법이 feature를 조절(scale)하는 것입니다. 
```
<img src="https://github.com/twooopark/ML_Summary/blob/master/3-1_G_feature_scaling_1.JPG" width="480px" height="270px" />

1. 범위를 유사하게 합니다.
```
모든 feature들의 값을 조절하여, -3 ~ - 1/3 < X < 1/3 ~ 3 정도로 조절하면, 빠른 결과를 얻을 수 있습니다.
```

2. mean nomalization 하는 이유
```
feature의 평균만큼 빼고, 최대값-최소값으로 나눕니다. feature에 관한 이 식은 정확하게는 아니더라도 대략 저 범위만큼 될 것입니다. 엄밀히 따진다면, 최대값에서 최소값을 뺏을 때, 실제로는 4가 됩니다. 즉, 최대값이 5고, 최소값이 1이면 range는 4가 되지만 여기서 대부분이 근사값이고, feature의 값도 range와 거의 차이가 없기 때문에 괜찮습니다. feature scaling은 너무 정확할 필요는 없습니다. gradient descent가 훨씬 더 빨라지면 되기 때문이죠 
```

3. 적절한 learning rate 찾는 법
```
다 날라갔다...
```




### 머신러닝 분야 정리

  a. 지도학습(Supervised Learning) : 과거, 현재 데이터로 미래를 예측 (라벨링 된 데이터 : 데이터에 대한 답이 주어져 있는 것 : 평가가 되어 있는 것 : 모델링 되어있는 것)
    
    - 분류(Classfication) : Categorial data, 데이터를 카테고리에 따라 분류. Ex) KNN, Naive Bayes, Logistic Regression, SVM, Decision tree
    
    - 회귀(Regression) : Continuous data, 연속된 값을 예측, 패턴이나 경향을 예측할 때 사용 Ex) Linear Regression, Random Forests, Ridge, Lasso
    
    
  b. 비지도학습(Unsupervised Learning) : 과거, 현재 데이터로 미래를 예측 (답이 정해지지 않은 데이터 : 평가되지 않은 데이터 : 라벨링되어 있지 않은 데이터, 로 미래를 예측) 
  
    - 군집(Clustering) : 데이터의 라벨이나 카테고리가 무엇인지 알 수 없는 경우가 많기 때문에 사용된다. Ex) K-means
        
    - Ex) Density estimation, Expection maximization, Pazen window, DBSCAN
    
    - Categorial data >> Hidden Markov Model ...
    
    
  c. 강화학습(Reinforcement Learning)  
  
    -  action에 따른 reward를 통해 학습한다. 이 과정을 반복해, 최상의 결과를 얻는 것이 목표. Ex) 공 튀기기 게임 인공지능)
    
    지도, 비지도와 달리 데이터가 정답이 있는 것이 아니며, 주어진 데이터가 없을 수도 있다. 


  - 차원축소(Dimentionality Reduction) : 특징(feature)이 너무 많기 때문에 학습이 어렵고 더 좋은 특징만 가지고 사용하기 위해 사용. Ex) PCA, 
  
  수치예측은 선형회귀가 짱이다.
  분류예측은 랜덤포레스트가 짱이다.
  
  
  
  
  CNN's Pooling(Subsampling)

DeepLearning.

CNN : convolution neural network, 합성곱 신경 망 : 이미지 레이어들의 특징점들을 이용하여, 이미지를 인식할 수 있도록 하는 알고리즘.
Word2vec : 구글에서 만듬, 단어를 벡터로 바꿈으로써, 단어의 유사도 추출 가능. 













