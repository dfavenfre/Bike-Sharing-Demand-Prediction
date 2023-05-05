class BuildRegressionModel:
    
    """ build a baseline ml model based on linear regression """
    
    def __init__(self, X, y):
        
        self.X = X
        self.y = y
    
    def shuffle_data(self, shuffle=True, test_size=30):
                
        self.shuffle=shuffle
        self.test_size=test_size
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            shuffle=self.shuffle,
                                                            test_size=self.test_size, 
                                                            random_state=42)      
        return X_train, X_test, y_train, y_test
        
    def baseline_lr(self, X_train, y_train, X_test):
    
        # instantiate lr model
        model = LinearRegression()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred, model
    
    def lgb_obj(trial, hyperparameters):

        param = {}

        for key, value in hyperparameters.items():
            if isinstance(value, Iterable):
                if isinstance(value[0], float):
                    param[key] = trial.suggest_float(key, value[0], value[1])
                else:
                    param[key] = trial.suggest_int(key, value[0], value[1])
            else:
                param[key] = value

        # non-hyperparameter settings
        param["n_jobs"] = -1 # deploy 100% of gpu's computational power 
        param["random_state"] = 42
        

        model = LGBMRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return (mean_squared_error(y_test, y_pred))**(1/2)
    
    
    def xgb_obj(trial, hyperparameters):
        param = {}

        for key, value in hyperparameters.items():
            if isinstance(value, Iterable):
                if isinstance(value[0], float):
                    param[key] = trial.suggest_float(key, value[0], value[1])
                else:
                    param[key] = trial.suggest_int(key, value[0], value[1])
            else:
                param[key] = value

        # non-hyperparameter settings
        param["n_jobs"] = -1 # deploy 100% of gpu's computational power 
        param["random_state"] = 42

        model = xgboost.XGBRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return (mean_squared_error(y_test, y_pred))**(1/2)
    
    def get_regression_result(self, y_test, y_pred):
        
        self.y_test = y_test
        self.y_pred = y_pred
        
        rmse = MSE(y_test, y_pred)**1/2
        
        return print("r^2: {:.3f}".format(r2_score(y_test, y_pred)))        
