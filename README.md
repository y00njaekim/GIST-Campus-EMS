# 📙 목차

# 📙 실행 환경 설정

- poetry 를 이용한 환경 설정

```shell
poetry install
```

- poetry 가상 환경 위치 확인

```shell
poetry env info -p
```

- vscode notebook kernel 설정
  - `cmd + shift + p` > `Notebook: Select Notebook Kernel` >`Select Another Kernel` > `Python Environment` > 이전에 확인한 가상 환경과 동일한 항목 선택



# 📙 태양광 발전량 예측 모델

> 오늘의 우리가 내일의 태양광 발전량을 예측하려 할 때 어떤 데이터를 사용할 수 있을까?
>
> 우리가 사용할 수 있는 데이터는 결국, **1. 기상 데이터** **2. 과거의 특정 기상 상황 속 태양광 발전량 History 데이터** 이다.
>
> 우리는 기상청의  **기상정보 예측데이터**를 수집한 후 해당 기상상황에서 태양광 발전기 위치의 **환경감시 데이터 **(수평면의 태양복사에너지, 패널의 태양복사에너지, 주변 온도, 패널 온도)를 예측한다. 이후 **예측 환경감시 데이터**를 통해 해당 패널에서 발전될 **태양광 에너지**을 예측한다.

## 📌 데이터셋 수집

### 1. 기상 데이터

- [기상자료개방포털](https://data.kma.go.kr/cmmn/main.do)에서 예/특보 > 단기예보 데이터셋을 다운로드 한다.
  
- 단기예보는 3시간 주기로 (현재시각 + 6시) ~ (현재시각 + 60시) 가량의 기상 정보를 예측한 데이터셋이다.
  
- csv 파일은 day, hour, forecast, value 로 구성되어 있다. hour(UTC) 은 예측을 수행한 시간이며 hour + forecast 는 예측의 대상 시간이다. 즉, hour 200, forecast 6 일 때, 2시 + 6시 + 9시(UTC->KST) = 17시에 대한 기상 정보를 예측하는 것이다.
  
- 우리는 예측 대상 날짜 **1일 전 8시(UST) 에 예측한 forecast 7 ~ 30 데이터**를 이용하여 해당 날짜의 1시간 단위 기상 값의 데이터셋을 구축한다. `./solar-power/filtered` [전처리 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/solar-power/preprocess.ipynb)
  
- 모든 기상 데이터를 특정 날짜와 특정 시각 (24시간 단위) 에 맞추어 하나의 csv 파일에 병합한다. [병합 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/solar-power/merge1.ipynb) / [병합데이터셋](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/solar-power/merged_dataset.csv)
  

### 2. 발전량 데이터

- 학교에서 제공한 발전량 데이터를 다운로드 한다. `./solar-power/solar-power-report`
  
- 발전량 데이터와 기상 데이터를 특정 날짜와 특정 시각 (24시간 단위) 에 맞추어 하나의 csv 파일에 병합한다. [병합 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/solar-power/merge2.ipynb) / [병합데이터셋]()
  

## 📌 기상데이터 to 환경감시 모델 구현

### 1. 데이터 전처리

- **특성 선택: '시간', '강수확률', '일최저기온', '1시간기온', '하늘상태', '일최고기온', '습도', '풍향', '1시간강수량', '풍속'**
  - **`하늘상태`** 는 맑음(1), 구름많음(3), 흐림(4)으로 구분되기 때문에 범주형 특성의 원-핫 인코딩 진행.
  - **`풍향`** 은 `sin` 을 이용하여 데이터 처리
  - 특성들은 `(0, 1)` 사의 값으로 정규화. **`시간`** 이 목표 변수를 결정하는 데 가장 중요한 특징이라고 판단하여 `(0.5, 1)` 의 정규화 처리
  
- **목표 변수 선택: '수평면', '외기온도', '경사면', '모듈온도'**
  - **`수평면`** 과 **`경사면`** 데이터의 분포 - Long Tail 을 극복하기 위해 Log 스케일의 정규화 처리.
    - (정규화 이전 데이터 분포)
      <img src="https://github.com/y00njaekim/GIST-Campus-EMS/assets/56385667/92b67e35-bdbf-44e1-837e-c4e8a73b870f"/>
    - (정규화 이후 데이터 분포)
      <img src="https://github.com/y00njaekim/GIST-Campus-EMS/assets/56385667/ba19355c-43f9-46b4-9c65-ef535919ab68"/>


### 2. 앙상블 모델 구축

- 앙상블 학습 방법인 Gradient Boosting을 활용하여 예측 모델을 구축

- 하이퍼 파라미터 튜닝 진행
  - GridSearchCV를 사용하여 최적의 하이퍼파라미터 조합을 자동으로 탐색
  - 여러 가지 하이퍼파라미터 값들을 시도하며, 교차 검증을 통해 각 조합의 성능을 평가
  - 최종적으로 가장 성능이 좋은 하이퍼파라미터 조합이 선택
  

### 3. 모델 평가 및 선택

- 평균 제곱 오차 (MSE)를 주요 평가 지표로 사용
- 테스트 데이터를 사용하여 각 목표 변수에 대한 예측 성능을 개별적으로 확인. 이를 통해 어떤 변수에 대한 예측이 더 잘되고, 어떤 변수에 대한 예측이 상대적으로 부족한지를 파악 가능.

## 📌 환경감시 to 태양광 발전량 모델 구현

### 1. 데이터 전처리

- **특성값(인풋) 전처리**
    - 인풋으로 들어가는 4개의 특성 값 sklearn MinMaxScaler 이용해 (0,1) 사잇값으로 정규화
    - 특성마다 값 편차가 매우 상이할 경우 학습 중요도가 달라질 수 있기 때문
- **Train/Test 데이터 선별**
    - 시계열 데이터에 종속되지 않는 데이터 학습을 위해 sklearn train_test_split 이용한 데이터 분별
- **목표 변수 단일화**
    - 발전소마다의 예측값을 내는 방식이 아닌, 한 시간대에 모든 발전소 발전량의 합을 목표 변수로 설정함.

### 2. 특성 조합 선택 및 학습

- LinearRegression 학습을 위해 4개의 특성을 sklearn PolynomialFeatures를 사용해 상관관계를 분석 후 선택, 적절한 특정 조합으로 인풋을 변환시켜줌
- 위의 과정을 거진 학습데이터 셋으로 sklearn LinearRegression 학습을 진행함
- 학습이 잘 진행되었는지 테스트 데이터셋과 score함수를 통해 accuracy에 대한 정량적 평가 진행

# 📙 전기 부하량 예측 모델

> 전기 부하량은 **기상 조건**의 영향을 받으며 **요일과 시간**의 **시계열 반복성**이 나타난다. 이 예측 모델은 주어진 데이터를 기반으로 전기 부하량을 예측하는 모델을 구현한다.

## 📌 데이터셋 수집

###  부하량 데이터 전처리

- 학교에서 제공한 부하량 데이터를 다운로드 한다. `./electrical-load/data-under`, `./electrical-load/data-master` (각각 학사 일보, 석사 일보 데이터)
- 부하량 데이터 중 전기요금의 근원인 **`유효전력`**  데이터를 추출 및 병합한다.
  이전에 추출한 **`기상데이터`** 와 병합하여 입력 특징과 목표 변수에 대한 파일을 생성한다.  [병합 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/electrical-load/preprocess.ipynb) / [병합데이터셋](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/electrical-load/merged_data.csv)

## 📌 전기 부하량 예측 모델 구현

- **`요일`** 을 나타내는 특징에 대해 원 핫 인코딩 처리
- `RandomForestRegressor` 를 사용하여 모델을 구축
- 평가 지표로는 Mean Squared Error (MSE)를 사용



