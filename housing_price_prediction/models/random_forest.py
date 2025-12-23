from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class RandomForest():

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
            clf = RandomForestRegressor(n_estimators=int(params["n_estimators"]),
                                        max_depth=int(params["max_depth"]),
                                        min_samples_split=int(params["min_samples_split"]),
                                        min_samples_leaf=int(params["min_samples_leaf"]),
                                        max_features=params["max_features"],
                                        random_state=RandomForest.seed,
                                        n_jobs=-1)

            score = cross_val_score(clf, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1).mean()

            return {"loss": -score,
                    "status": STATUS_OK}
        
        search_space = {"n_estimators": hp.quniform("n_estimators", 100, 600, 50),
                        "max_depth": hp.quniform("max_depth", 3, 20, 1),
                        "min_samples_split": hp.quniform("min_samples_split", 2, 20, 1),
                        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
                        "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
                        "criterion": hp.choice("criterion", ["friedman_mse", "absolute_error", "poisson", "squared_error"])
                        }
        
        trials = Trials()

        best = fmin(fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    rstate=np.random.default_rng(RandomForest.seed))
        
  
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
                       "min_samples_split": int(best["min_samples_split"]),
                       "min_samples_leaf": int(best["min_samples_leaf"]),
                       "max_features": ["sqrt", "log2", None][best["max_features"]],
                       "criterion": ["friedman_mse", "absolute_error", "poisson", "squared_error"][best["criterion"]],
                       "random_state": RandomForest.seed,
                       "n_jobs": -1}
        
        # Retrain final model
        tuned_clf = RandomForestRegressor(**best_params, verbose=0)
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
        # -------------------------------
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
        
        score = cross_val_score(clf, X, np.ravel(y), cv = 5, scoring='roc_auc')
        return score