from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class XGBoost():

    seed = 100
    np.random.seed(seed)
    random.seed(seed)
    
    def tune_regressor(self, X, y):
        """
        Tune hyper-parameters
        Parameters:
        X : array
        y : array

        Returns:
        dictionary
        """
        
        def objective(params):
            clf = XGBRegressor(objective='reg:squarederror',
                               n_estimators=int(params["n_estimators"]),
                               max_depth=int(params["max_depth"]),
                               learning_rate=params["learning_rate"],
                               subsample=params["subsample"],
                               colsample_bytree=params["colsample_bytree"],
                               gamma=params["gamma"],
                               reg_alpha=params["reg_alpha"],
                               reg_lambda=params["reg_lambda"],
                               random_state=XGBoost.seed,
                               eval_metric='logloss',
                               n_jobs=-1)

            score = cross_val_score(clf, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).mean()

            return {"loss": -score,
                    "status": STATUS_OK}
        
        search_space = {"n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
                        "max_depth": hp.quniform("max_depth", 3, 10, 1),
                        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
                        "subsample": hp.uniform("subsample", 0.6, 1.0),
                        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
                        "gamma": hp.uniform("gamma", 0, 5),
                        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
                        "reg_lambda": hp.uniform("reg_lambda", 0, 5)}
        
        trials = Trials()

        best = fmin(fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    rstate=np.random.default_rng(XGBoost.seed))
        
  
        return best
    
    
    def fit_regressor(self, X, y, best):
        """
        Fit final model with optimal hyper-parameters
        Parameters:
        y : array
        y_pred : array
        best: dictionary

        Returns:
        object, dictionary
        """
        
        best_params = {"n_estimators": int(best["n_estimators"]),
                       "max_depth": int(best["max_depth"]),
                       "learning_rate": best["learning_rate"],
                       "subsample": best["subsample"],
                       "colsample_bytree": best["colsample_bytree"],
                       "gamma": best["gamma"],
                       "reg_alpha": best["reg_alpha"],
                       "reg_lambda": best["reg_lambda"],
                       "random_state": XGBoost.seed,
                       "eval_metric": 'logloss',
                       "n_jobs": -1}
        
        # Retrain final model
        tuned_clf = XGBRegressor(objective='reg:squarederror', **best_params)
        tuned_clf.fit(X, y)
        
        return tuned_clf, best_params
    
    def compute_mse(self, y, y_pred):
        """
        Compute MSE.
        Parameters:
        y : array
        y_pred : array

        Returns:
        float
        """
        mse = mean_squared_error(y, y_pred)
        return mse
    
    def compute_mae(self, y, y_pred):
        """
        Compute MAE.
        Parameters:
        y : array
        y_pred : array

        Returns:
        float
        """
        mae = mean_absolute_error(y, y_pred)
        return mae
    
    
    def sorted_feature_importance_indicies(self, clf):
        """
        Retrieves sorted feature importance 
        Parameters:
        clf : object
     
        Returns:
        array
        """

        sorted_indices = np.argsort(clf.feature_importances_)[::-1]
 
        return sorted_indices
    
    
    def compute_prediction_interval(self, X_train, y_train, X_test, y_test, best):


        # Define alpha for 95% prediction interval
        alpha_lower = 0.05  # 5th percentile
        alpha_upper = 0.95  # 95th percentile

        best_params = {"n_estimators": int(best["n_estimators"]),
                       "max_depth": int(best["max_depth"]),
                       "learning_rate": best["learning_rate"],
                       "subsample": best["subsample"],
                       "colsample_bytree": best["colsample_bytree"],
                       "gamma": best["gamma"],
                       "reg_alpha": best["reg_alpha"],
                       "reg_lambda": best["reg_lambda"],
                       "random_state": XGBoost.seed,
                       "eval_metric": 'logloss',
                       "n_jobs": -1}

        # Lower quantile model
        model_lower = XGBRegressor(
            objective='reg:quantileerror',
            **best_params
        )

        # Upper quantile model
        model_upper = XGBRegressor(
            objective='reg:quantileerror',
            **best_params
        )

        # Fit both models
        model_lower.fit(X_train, y_train)
        model_upper.fit(X_train, y_train)

        # Predict intervals
        y_lower = model_lower.predict(X_test)
        y_upper = model_upper.predict(X_test)

        res = {'y_lower': y_lower,
               'y_upper': y_upper}

        return res
