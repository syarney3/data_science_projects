from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class LinearRegressionModel():

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

            model = LinearRegression(**params)

            score = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="neg_mean_absolute_error"
            ).mean()

            return {
                "loss": -score,  # minimize
                "status": STATUS_OK
            }
        
        search_space = {"fit_intercept": hp.choice("fit_intercept", [True, False]),
                        "positive": hp.choice("positive", [True, False])}
        
        trials = Trials()

        best = fmin(fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials,
                    rstate=np.random.default_rng(LinearRegressionModel.seed))
        
  
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
        
        best_params = {"fit_intercept": [True, False][best["fit_intercept"]],
                       "positive": [True, False][best["positive"]]}
        
        # Retrain final model
        final_model = LinearRegression(**best_params)
        final_model.fit(X, y)
        
        return final_model, best_params
    
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
