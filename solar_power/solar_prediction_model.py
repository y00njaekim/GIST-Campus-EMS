import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class FeatureSolarPredictionModel:
    '''
    1. 모델 초기화
    model = FeatureSolarPredictionModel("path_to_your_data.csv")
    2. 데이터 로딩 및 전처리
    model.load_data()
    3. 피쳐 셀렉션
    model.feature_selection()
    4. 모델 학습
    model.train_model()
    5. 예측
    model.predict(input_data)
    6. 테스트 에러 가져오기
    model.get_trained_errors()
    '''
    def __init__(self, data_path, random_state=42, test_size=0.2):
        self.data_path = data_path
        self.random_state = random_state
        self.test_size = test_size
        self.data = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.input_features = ['수평면', '외기온도', '경사면', '모듈온도']
        self.target_variables = ['축구장', '학생회관', '중앙창고', '학사과정', '다산빌딩', '시설관리동', '대학C동', '동물실험동', '중앙도서관', 'LG도서관', '신재생에너지동', '삼성환경동', '중앙연구기기센터', '산업협력관', '기숙사 B동']
        self.poly = None
        self.train_poly = None
        self.test_poly = None
        self.summed_z_train = None
        self.summed_z_test = None
        self.model = None
        self.predictions = None
        self.test_errors = None

    def load_data(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
            
        self.data = pd.read_csv(self.data_path)

        # 전처리
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data[self.input_features] = self.scaler.fit_transform(
            self.data[self.input_features])

        # 데이터 분할
        X = self.data[self.input_features]
        y = self.data[self.target_variables]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        # 모든 발전소별 발전량 하나의 시간대 안에서 통합
        np_z_train = self.y_train.to_numpy()
        np_z_test = self.y_test.to_numpy()
        self.summed_z_train = np_z_train.sum(axis=1)
        self.summed_z_test = np_z_test.sum(axis=1)

    def feature_selection(self):
        # linearregression에 맞는 적절한 조합의 feature 형태들 선별
        self.poly = PolynomialFeatures(include_bias=False)
        self.poly.fit(self.X_train)
        self.train_poly = self.poly.transform(self.X_train)
        self.test_poly = self.poly.transform(self.X_test)

    def train_model(self):
        lr = LinearRegression()
        lr.fit(self.train_poly, self.summed_z_train)
        self.model = lr

    def predict(self, input_data):
        # input_data type = pandas.core.frame.DataFrame
        # input_data.shape = (X, 4)
        nd_input_data = input_data.to_numpy()
        for i in range(len(nd_input_data)):
            condition = nd_input_data[i] < 10
            nd_input_data[i][condition] = 0
        scalered_input_data = self.scaler.fit_transform(nd_input_data)
        poly_input_data = self.poly.transform(scalered_input_data)
        predictions = self.model.predict(poly_input_data)
        return predictions

    def get_trained_errors(self):
        self.predictions = self.model.predict(self.test_poly)
        self.test_errors = mean_squared_error(
            self.summed_z_test, self.predictions)
        return self.test_errors


# model_instance = FeatureSolarPredictionModel("./merged_result.csv")
# model_instance.load_data()
# model_instance.feature_selection()
# model_instance.train_model()
# err = model_instance.get_trained_errors()
# print(err)