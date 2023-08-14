# 데이터셋 수집



## 1. 기상 데이터

- [기상자료개방포털](https://data.kma.go.kr/cmmn/main.do)에서 예/특보 > 단기예보 데이터셋을 다운로드 한다.

- 단기예보는 3시간 주기로 (현재시각 + 6시) ~ (현재시각 + 60시) 가량의 기상 정보를 예측한 데이터셋이다.

- csv 파일은 day, hour, forecast, value 로 구성되어 있다. hour(UTC) 은 예측을 수행한 시간이며 hour + forecast 는 예측의 대상 시간이다. 즉, hour 200, forecast 6 일 때, 2시 + 6시 + 9시(UTC->KST) = 17시에 대한 기상 정보를 예측하는 것이다.

- 우리는 예측 대상 날짜 **1일 전 8시(UST) 에 예측한 forecast 7 ~ 30 데이터**를 이용하여 해당 날짜의 1시간 단위 기상 값의 데이터셋을 구축한다. `./filtered`  [전처리 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/preprocess.ipynb)

- 모든 기상 데이터를 특정 날짜와 특정 시각 (24시간 단위) 에 맞추어 하나의 csv 파일에 병합한다. [병합 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/merge1.ipynb) / [병합데이터셋](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/merged_dataset.csv)



## 2. 발전량 데이터

- 학교에서 제공한 발전량 데이터를 다운로드 한다. `./solar-power-report`
- 발전량 데이터와 기상 데이터를 특정 날짜와 특정 시각 (24시간 단위) 에 맞추어 하나의 csv 파일에 병합한다. [병합 코드](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/merge2.ipynb) / [병합데이터셋](https://github.com/y00njaekim/GIST-Campus-EMS/blob/main/merged_result.csv)





# 모델 구현

## 데이터 전처리

- 특성 선택: '강수확률', '일최저기온', '1시간기온', '하늘상태', '일최고기온', '습도', '풍향', '1시간강수량', '풍속'
- 목표 변수 선택: '수평면', '외기온도', '경사면', '모듈온도'
- 범주형 특성 원-핫 인코딩
- 학습 및 테스트 세트 분할

## 개별 모델 선택 및 학습

- 랜덤 포레스트, Gradient Boosting, SVM 선택
- 각 목표 변수에 대해 개별 모델 훈련

## 앙상블 모델 구축

- 개별 모델의 예측 평균
- 오버피팅 방지 및 성능 향상

## 모델 평가 및 선택

- 테스트 데이터 사용
- 평균 제곱 오차 (MSE), 평균 절대 오차 (MAE) 사용
- 최종 모델 선택

- 테스트 데이터 중 샘플 예측 및 실제값 비교
