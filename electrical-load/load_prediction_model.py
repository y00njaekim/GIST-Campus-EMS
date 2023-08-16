import pandas as pd
from sklearn.model_selection import train_test_split

class BuildingEnergyModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
    
    def load_data(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
            
        self.data = pd.read_csv(self.data_path)
        self.data_encoded = pd.get_dummies(self.data, columns=['weekday'])
        self.X = self.data_encoded[['time', '1시간기온', '1시간강수량', '일최고기온', '일최저기온', 
                                    'weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 
                                    'weekday_Sunday', 'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday']]
        self.y = self.data_encoded.loc[:, 'SV-2_석사':'산학협력연구동(E)_학사']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        
    def train_model(self, n_estimators=100):
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, input_data):
        import pandas as pd
        
        assert self.model is not None, "Model is not trained yet. Call the 'train' method first."
        
        return self.model.predict(input_data)
    
    def evaluate(self):
        from sklearn.metrics import mean_squared_error
        
        assert self.model is not None, "Model is not trained yet. Call the 'train' method first."
        
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse


# model_instance = BuildingEnergyModel("./merged_data.csv")
# model_instance.load_data()
# model_instance.train_model()

# y_pred = model_instance.predict(model_instance.X_test)
# y_test_df = pd.DataFrame(model_instance.y_test, columns=model_instance.y.columns)
# y_pred_df = pd.DataFrame(y_pred, columns=model_instance.y.columns)

# print("Actual Data:\n", y_test_df.head(), "\n")
# print("Predicted Data:\n", y_pred_df.head())

# print("Actual Data:")
# print(y_test_df.head())
# print("\nPredicted Data:")
# print(y_pred_df.head())

