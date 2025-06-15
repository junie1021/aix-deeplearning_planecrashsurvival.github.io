## aix-deeplearning_planecrashsurvival.github.io
# Title: 비행기 사고 생존율 분석
# Members
문준영 정보시스템공학과 Liquidangel6922@gmail.com 

Task: 모델구조 설계/ 코드 작성

박준희 융합전자공학부 stayfun.junie@gmail.com

Task: 모델에 대한 설명글 작성/ 블로그 작성 

정수민 컴퓨터소프트웨어학과 min_jr13@naver.com

Task: 모델에 대한 설명글 작성/ 영상촬영

# I. Proposal
### - Motivation
항공 안전 연구는 항공 산업의 지속적인 발전과 대중의 신뢰 확보에 있어 핵심적인 요소입니다. 과거 항공 사고로부터 얻은 교훈은 항공기 설계, 운영절차, 규제 기준의 개선을 이끌어 왔으며, 이는 항공여행을 가장 안전한 교통수단 중 하나로 만드는 데 기여하였습니다. 그럼에도 불구하고, 사고 발생 시 피해를 최소화하고 생존 가능성을 높이기 위한 노력은 끊임없이 이루어져야 합니다. 최근 몇 년간 머신러닝(ML) 기술의 발전은 방대한 양의 사고 데이터 속에 숨겨진 복잡한 패턴을 규명하고 이를 통해 보다 정교한 예측 및 분석을 가능하게 하는 새로운 지평을 열었습니다. 또한 근래에 우리나라에서 비행기 사고로 인한 참사가 발생한 바, 시의적절하다고 판단되어 비행기 사고 생존율 분석 모델을 구현하고자 하였습니다.
### - What do you want to see at the end?
본 프로젝트는 제공된 Kaggle의 "항공 사고 데이터베이스 시놉시스(Abiation Accident Database Synopses)" 데이터셋을 활용하여 항공 사고 발생시 생존율을 시뮬레이션하는 머신러닝모델을 개발하는 것을 목표로 합니다. 이 프로젝트는 적절한 머신러닝 알고리즘 선택 및 검증, 그리고 최종적으로 모델 결과를 해석하고 생존율을 시뮬레이션하는 과정을 다룹니다. 항공 사고 데이터 분석은 그 본질상 민감한 정보를 다루며, 결과 해석에 있어 신중함이 요구되므로 이러한 민감성을 고려하여 체계적이고 과학적인 접근방식을 통해 의미 있는 결과를 도출할 수 있도록 하는데 중점을 두었습니다. 
# II. Datasets
https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses


데이터셋은 Kaggle의 Aviation Accident Database & Synopses, up to 2023 를 이용하였습니다. 이 데이터셋은 1962년부터 현재까지의 민간 항공 사고와 미국, 미국 영토 및 속령, 국제 해역에서 발생한 특정 사건들에 대한 정보를 담고 있습니다.


iris_150.csv 는 로지스틱 회귀모델(이진), 로지스틱 회귀모델(OvA), 결정트리모델, K-NN 모델을 실험할 때 사용하였고, house_prices_100.csv 는 다중선형회귀모델, 단순선형회귀모델을 실험할 때 사용하였습니다. 
# III. Methodology
### 가상 환경 생성 및 활성화
``` python3 -m venv venv
source venv/bin/activate
```
### 필요 라이브러리 설치
``` requirements.txt:

pandas
numpy
matplotlib
scikit-learn

pip install -r requirements.txt
```

# IV. Evaluation & Analysis
# V. Conclusion: Discussion
