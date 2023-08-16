import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


class SolarPredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.scaler = None
        self.scaler_important = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.features_to_scale = ['강수확률', '일최저기온',
                                  '하늘상태', '일최고기온', '습도', '풍향', '풍속']
        self.important_features = ['time']
        self.input_features = self.features_to_scale + self.important_features
        self.target_variables = ['수평면', '외기온도', '경사면', '모듈온도']

    def load_data(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
            
        self.data = pd.read_csv(self.data_path)
        self.data['풍향'] = np.sin(np.deg2rad(self.data['풍향']))

        # 전처리
        self.scaler = MinMaxScaler()
        self.data[self.features_to_scale] = self.scaler.fit_transform(
            self.data[self.features_to_scale])

        self.scaler_important = MinMaxScaler(feature_range=(0.5, 1))
        self.data[self.important_features] = self.scaler_important.fit_transform(
            self.data[self.important_features])

        # 데이터 분할
        test_size = 0.4
        random_state = 42

        X = self.data[self.input_features]
        y = self.data[self.target_variables]
        X_train_temp, self.X_test, y_train_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.5, random_state=random_state)

    def get_data_shape(self):
        return self.data.shape


class SolarPredictionModelExtended(SolarPredictionModel):
    '''
    1. 모델 초기화
    model = SolarPredictionModelExtended("path_to_your_data.csv")
    2. 데이터 로딩 및 전처리
    model.load_data("path_to_your_data.csv")
    3. 하이퍼파라미터 튜닝
    model.hyperparameter_tuning()
    4. 모델 학습
    model.train_model()
    5. 예측
    model.predict(input_data)
    6. 테스트 에러 가져오기
    model.get_trained_errors()
    '''
    def __init__(self, data_path):
        super().__init__(data_path)
        self.best_params = {}
        self.models = {}
        self.predictions = {}
        self.test_errors = {}

    def hyperparameter_tuning(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1],
            'max_features': ['sqrt', 'log2', None]
        }

        for target in self.target_variables:
            grid_search = GridSearchCV(GradientBoostingRegressor(
                random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train, self.y_train[target])
            self.best_params[target] = grid_search.best_params_

    def train_model(self):
        for target in self.target_variables:
            min_val_error = float("inf")
            error_going_up = 0
            gbr = GradientBoostingRegressor(
                **self.best_params[target], random_state=42)
            for n_estimators in range(1, 120):
                gbr.n_estimators = n_estimators
                gbr.fit(self.X_train, self.y_train[target])
                y_pred_val = gbr.predict(self.X_val)
                val_error = mean_squared_error(self.y_val[target], y_pred_val)
                if val_error < min_val_error:
                    min_val_error = val_error
                    error_going_up = 0
                else:
                    error_going_up += 1
                    if error_going_up == 5:
                        break
            self.models[target] = gbr

    def predict(self, input_data):
        # input_data type = pandas.core.frame.DataFrame
        # input_data.shape = (X, 8)
        # return example: {'수평면': array([00.00]), '외기온도': array([00.00]), '경사면': array([00.00]), '모듈온도': array([00.00])}
        for target in self.target_variables:
            self.predictions[target] = self.models[target].predict(input_data)
        return self.predictions

    def get_trained_errors(self):
        for target in self.target_variables:
            self.predictions[target] = self.models[target].predict(self.X_test)
            self.test_errors[target] = mean_squared_error(
                self.y_test[target], self.predictions[target])
        return self.test_errors