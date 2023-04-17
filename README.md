# Bike Rent Demand Prediction Model

Within the project, it is purposed to build a demand prediction model for bike sharing count for Capital Bikeshare program in Washington, D.C.. The data is available at Kaggle's database (https://www.kaggle.com/competitions/bike-sharing-demand/data). 

We've written a class that enables multi-model building, as well as optimization with Optuna for RMSE (root mean squared error). As an example for the whole project, a master notebook is also available within the repository. 

## Data Pre-processing

First, We increased the number of features by dummifying categorical variables, though since the data we have is quite small in terms of size, modelling it this way  easily dropped the model' scores.

As a consideration for the linear model, transforming categorical variables into numeric ones by using either get_dummies of Pandas or Onehot Encoding of Sklearn are more meaningful and thus they provide better score when compared to not transforming them into numeric. 

However, in tree-based models like Decision-Tree, XGBoost, LGBM, RandomForest, etc., not applying these methodologies ensures the tree-based models to understand nonlinear relationships between features and gives better and significant results.

As stated earlier, the data includes way too much categorical features to apply any encoding algorithm (or non-linear relationships might not be caught by tree algorithms)herefore, some of these features, which are under the treshold specified for feature engineering, may be assumed over or underestimated when it comes to feature importance

Consequently, we decided to move forward without applying and encoding for the pre-process. The rest of the was way too perfect, with no missing values or outliers.

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