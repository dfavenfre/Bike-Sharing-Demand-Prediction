# Bike Rent Demand Prediction Model

Within the project, it is purposed to build a demand prediction model for bike sharing count for Capital Bikeshare program in Washington, D.C.. The data is available at Kaggle's database (https://www.kaggle.com/competitions/bike-sharing-demand/data). 

We've written a class that enables multi-model building, as well as optimization with Optuna for RMSE (root mean squared error). As an example for the whole project, a master notebook is also available within the repository. 

## Data Pre-processing

First, We increased the number of features by dummifying categorical variables, though since the data we have is quite small in terms of size, modelling it this way  easily dropped the model' scores.

As a consideration for the linear model, transforming categorical variables into numeric ones by using either get_dummies of Pandas or Onehot Encoding of Sklearn are more meaningful and thus they provide better score when compared to not transforming them into numeric. 

However, in tree-based models like Decision-Tree, XGBoost, LGBM, RandomForest, etc., not applying these methodologies ensures the tree-based models to understand nonlinear relationships between features and gives better and significant results.

As stated earlier, the data includes way too much categorical features to apply any encoding algorithm (or non-linear relationships might not be caught by tree algorithms)herefore, some of these features, which are under the treshold specified for feature engineering, may be assumed over or underestimated when it comes to feature importance

Consequently, we decided to move forward without applying and encoding for the pre-process. The rest of the was way too perfect, with no missing values or outliers.

## Variable Description
![1](https://user-images.githubusercontent.com/118773869/232551694-b27f3b81-2ddc-4972-8981-4e3f36dbf083.png)

## Preliminary Inferences

![image](https://user-images.githubusercontent.com/118773869/232552049-fe3098c5-3f47-469d-9f48-d002e12e6dee.png)


## Example Usage of BuildRegressionModel.py 

```
# pre-define hyperparameters
hyperparameters_lgb = {
    
    "n_estimators":[200,2000],
    "learning_rate":[0.01,0.10],
    "num_leaves":[20,1000],
    "max_depth":[3,12],
    "min_data_in_leaf":[200,10000],
    "max_bin":[200,300],
    "lambda_l1":[0,80],
    "lambda_l2":[0,100],
    "min_gain_to_split":[0,30]
}

# instantiate modeller
brm = BuildRegressionModel(X,y) # feature and target should be assigned earlier

# shuffle and split data
X_train, X_test, y_train, y_test = brm.shuffle_data() # if no parameter is passed, the defaults are; Shuffle=True, test_size=0.3

# build the optimized XGB model with Optuna
study_xgb = optuna.create_study(direction='minimize',study_name='XGBregression')
study_xgb.optimize(BuildRegressionModel.xgb_obj, n_trials=100) # the optimization is stopped after 100 trials 

model_xgb = xgboost.XGBRegressor(**study_xgb.best_params)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# Optimized XGBoosting Model Results
brm.get_regression_result(y_test, y_pred_xgb)

```
# Optuna Optimization Details

```
    def xgb_obj(trial, hyperparameters=hyperparameters_xgb):
        """
        Description:
        -----------
        Optimize Extreme Gradient Boosting with pre-defined hyperparameters. The tuning is optimized for minimizing the error terms, which is rooted means squared errors (RMSE).  
        
        Parameters:
        -----------
        hyperparameters(dict): Pre-defined dictionary consist of key and value pairs of hyperparameters
        Returns:
        
        -----------
        Verbose lines with RMSE scores 
        """
        
        
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
```


# Hyperparameter fine-tuning with Optuna  

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.  We used Optuna both for finding best parameters and score and for visualising the importance and relations and which values of them useing more and getting better and faster result.

## Extreme Gradient Boosting Model Regression Hyperparameter Importances 

![newplot (1)](https://user-images.githubusercontent.com/118773869/232568264-9875effb-4ab8-4ae1-ae99-accde0ddebfd.png)

![newplot](https://user-images.githubusercontent.com/118773869/232568171-aa5d2b7c-a238-4042-b50f-05c9d31792f6.png)


## Light Gradient Boosting Model Regression Hyperparameter Importances 

![newplot (2)](https://user-images.githubusercontent.com/118773869/232569065-bc85851e-5685-4603-b98c-a7dcbd458fc5.png)

![newplot (3)](https://user-images.githubusercontent.com/118773869/232569086-89ea50e8-8134-43e2-9c25-a8be69a70298.png)

# Controlling for futures' individual contribution using SHAP
SHAP (SHapley Additive Explanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. It measures each inputs' individual contribution to the model. The more red, the more contribution, and vice versa. 

### XGB Model Output
![image](https://user-images.githubusercontent.com/118773869/232570216-f3c04203-f163-4a8a-93cf-fd50348b2546.png)
### LGB Model Output
![image](https://user-images.githubusercontent.com/118773869/232571612-61598cc2-0bee-4efb-bb7a-4c04e16d17a3.png)

# Ensemble Regression Model

Ensemble models combine the decisions from multiple models to improve the overall performance. This can be achieved in various ways. A voting ensemble involves making a prediction that is the average of multiple other regression models.

Stacking is also an ensemble learning technique that uses predictions from multiple models (for example decision tree, knn or svm) to build a new model. This model is used for making predictions on the test set.

# Conclusion 

We've applied several Machine Learning Regression models, including XGBM, LGBM, Stacked and Bagging Ensemble algrothims, and the below table summarizes our findings. Depending on the goal, minimizing rmse or maximixing r-square, the below table should be able to help you out.  

| Model | R-square score | RMSE score|
|-------|----------------|-----------|
| Stacked Ensemble| 0.975| 35.341|
| Bagging Ensemble| 0.960| 45.465|
| Optimized XGB| 0.969| 32.553|  
| Optimized LGB| 0.970| 39.209|
