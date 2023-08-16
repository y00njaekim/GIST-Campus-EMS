import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


class SolarPredictionModel:
    def __init__(self, data_path, random_state=42, test_size=0.2):
        self.data_path = data_path
        self.random_state = random_state
        self.test_size = test_size
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
        self.target_variables = ['log_수평면', '외기온도', 'log_경사면', '모듈온도']

    def load_data(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path

        self.data = pd.read_csv(self.data_path)

        # 전처리
        self.data['풍향'] = np.sin(np.deg2rad(self.data['풍향']))
        self.data['log_경사면'] = np.log(self.data['경사면'] + 1)
        self.data['log_수평면'] = np.log(self.data['수평면'] + 1)
        self.scaler = MinMaxScaler()
        self.data[self.features_to_scale] = self.scaler.fit_transform(
            self.data[self.features_to_scale])

        self.scaler_important = MinMaxScaler(feature_range=(0.5, 1))
        self.data[self.important_features] = self.scaler_important.fit_transform(
            self.data[self.important_features])

        # 데이터 분할
        X = self.data[self.input_features]
        y = self.data[self.target_variables]
        # GroupShuffleSplit을 사용하여 데이터 분할
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_val_idx, test_idx = next(gss.split(X, y, groups=self.data['day']))
        
        X_train_temp, self.X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_temp, self.y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        
        train_idx, val_idx = next(gss.split(X_train_temp, y_train_temp, groups=self.data['day'].iloc[train_val_idx]))
        
        self.X_train, self.X_val = X_train_temp.iloc[train_idx], X_train_temp.iloc[val_idx]
        self.y_train, self.y_val = y_train_temp.iloc[train_idx], y_train_temp.iloc[val_idx]
        
    def get_data_shape(self):
        return self.data.shape


class SolarPredictionModelExtended(SolarPredictionModel):
    '''
    1. 모델 초기화
    model = SolarPredictionModelExtended("path_to_your_data.csv")
    2. 데이터 로딩 및 전처리
    model.load_data()
    3. 하이퍼파라미터 튜닝
    model.hyperparameter_tuning()
    4. 모델 학습
    model.train_model()
    5. 예측
    model.predict(input_data)
    6. 테스트 에러 가져오기
    model.get_trained_errors()
    '''

    def __init__(self, data_path, random_state=42, test_size=0.2):
        super().__init__(data_path, random_state, test_size)
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
                random_state=self.random_state), param_grid, cv=5, n_jobs=-1, verbose=0)
            grid_search.fit(self.X_train, self.y_train[target])
            self.best_params[target] = grid_search.best_params_

    def train_model(self):
        for target in self.target_variables:
            min_val_error = float("inf")
            error_going_up = 0
            gbr = GradientBoostingRegressor(
                **self.best_params[target], random_state=self.random_state)
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
            if '수평면' in target:
                # self.predictions['수평면'] = self.models[target].predict(input_data)
                self.predictions['수평면'] = np.exp(self.models[target].predict(input_data)) - 1
            elif '경사면' in target:
                # self.predictions['경사면'] = self.models[target].predict(input_data)
                self.predictions['경사면'] = np.exp(self.models[target].predict(input_data)) - 1
            else:
                self.predictions[target] = self.models[target].predict(input_data)
            
        return pd.DataFrame(self.predictions)
    
    def get_trained_errors(self):
        for target in self.target_variables:
            if '수평면' in target:
                self.predictions['수평면'] = np.exp(self.models[target].predict(self.X_test)) - 1
            elif '경사면' in target:
                self.predictions['경사면'] = np.exp(self.models[target].predict(self.X_test)) - 1
            else:
                self.predictions[target] = self.models[target].predict(self.X_test)
                
            if '수평면' in target:
                self.test_errors['수평면'] = mean_squared_error(
                self.y_test[target], self.predictions['수평면'])
            elif '경사면' in target:
                self.test_errors['경사면'] = mean_squared_error(
                self.y_test[target], self.predictions['경사면'])
            else:
                self.test_errors[target] = mean_squared_error(
                    self.y_test[target], self.predictions[target])
        return self.test_errors

# data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merged_result.csv')
# model_instance = SolarPredictionModelExtended(data_path)
# model_instance.load_data()
# model_instance.hyperparameter_tuning()
# model_instance.train_model()
# print(model_instance.get_trained_errors())

# comparison_dfs = {}
# for target in model_instance.target_variables:
#     if '수평면' in target:
#         comparison_dfs['수평면'] = pd.DataFrame({
#             f"Actual 수평면": np.exp(model_instance.y_test[target]) - 1,
#             f"Predicted 수평면": model_instance.predictions['수평면']
#         })
#     elif '경사면' in target:
#         comparison_dfs['경사면'] = pd.DataFrame({
#             f"Actual 경사면": np.exp(model_instance.y_test[target]) - 1,
#             f"Predicted 경사면": model_instance.predictions['경사면']
#         })
#     else:
#         comparison_dfs[target] = pd.DataFrame({
#             f"Actual {target}": model_instance.y_test[target],
#             f"Predicted {target}": model_instance.predictions[target]
#         })

# print(comparison_dfs['외기온도'].head())
# print(comparison_dfs['수평면'].head())
# print(comparison_dfs['모듈온도'].head())
# print(comparison_dfs['경사면'].head())


# ------------------------------
# data_path = "./merged_result.csv"
# model_instance = SolarPredictionModelExtended(data_path)
# model_instance.load_data()
# model_instance.hyperparameter_tuning()
# model_instance.train_model()
# print(model_instance.get_trained_errors())

# comparison_dfs = {}
# for target in model_instance.target_variables:
#     comparison_dfs[target] = pd.DataFrame({
#         f"Actual {target}": model_instance.y_test[target],
#         f"Predicted {target}": model_instance.predictions[target]
#     })

# print(comparison_dfs['외기온도'].head())
# print(comparison_dfs['수평면'].head())
# print(comparison_dfs['모듈온도'].head())
# print(comparison_dfs['경사면'].head())