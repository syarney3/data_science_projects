from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class LightGBM():

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

            clf = LGBMRegressor(objective='regression_l1',
                                n_estimators=int(params["n_estimators"]),
                                num_leaves=int(params["num_leaves"]),
                                max_depth=int(params["max_depth"]),
                                learning_rate=params["learning_rate"],
                                subsample=params["subsample"],
                                colsample_bytree=params["colsample_bytree"],
                                reg_alpha=params["reg_alpha"],
                                reg_lambda=params["reg_lambda"],
                                random_state=LightGBM.seed,
                                n_jobs=-1)

            mae = cross_val_score(clf, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).mean()

            return {"loss": -mae,
                    "status": STATUS_OK}
        
        search_space = {"n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
                        "num_leaves": hp.quniform("num_leaves", 20, 150, 1),
                        "max_depth": hp.quniform("max_depth", 3, 15, 1),
                        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
                        "subsample": hp.uniform("subsample", 0.6, 1.0),
                        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
                        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
                        "reg_lambda": hp.uniform("reg_lambda", 0, 5)
                        }
        
        trials = Trials()

        best = fmin(fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    rstate=np.random.default_rng(LightGBM.seed))
        
  
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
                       "num_leaves": int(best["num_leaves"]),
                       "max_depth": int(best["max_depth"]),
                       "learning_rate": best["learning_rate"],
                       "subsample": best["subsample"],
                       "colsample_bytree": best["colsample_bytree"],
                       "reg_alpha": best["reg_alpha"],
                       "reg_lambda": best["reg_lambda"],
                       "random_state": LightGBM.seed,
                       "n_jobs": -1}
        
        # Retrain final model
        tuned_clf = LGBMRegressor(**best_params, verbose=0)
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
    
    
    def cross_validation_classifier(self, clf, X, y):
        
        """
        Performs cross-validation
        Parameters:
        clf : object
        X : array
        y: array

        Returns:
        float
        """
        
        mse = cross_val_score(clf, X, np.ravel(y), cv = 5, scoring='neg_mean_squared_error')
        mse = -mse #convert to positive mse
        return mse