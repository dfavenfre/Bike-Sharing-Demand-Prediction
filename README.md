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

VISUALIZATION WITH OPTUNA 

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Both we use Optuna for finding best parameters and score and for visualising the importance and relations and which values of them useing more and getting better result.
As result of LightGBM Regression analysis, We can see 


<img width="1105" alt="Screenshot 2023-04-17 at 20 00 37" src="https://user-images.githubusercontent.com/116746888/232557638-bb7107f5-637d-40e5-b477-7de98198e81f.png">




<img width="1098" alt="Screenshot 2023-04-17 at 19 53 32" src="https://user-images.githubusercontent.com/116746888/232556076-f6d48005-522f-42fa-a13a-3c2068a5f51b.png">

<img width="1110" alt="Screenshot 2023-04-17 at 19 53 57" src="https://user-images.githubusercontent.com/116746888/232556147-6a49c3c2-59f0-462d-abf1-5d505516d882.png">

<img width="1102" alt="Screenshot 2023-04-17 at 19 54 24" src="https://user-images.githubusercontent.com/116746888/232556249-90c3bbf0-48b1-4995-b525-47429d1956dc.png">


