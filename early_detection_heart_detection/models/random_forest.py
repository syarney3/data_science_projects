from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class RandomForest():

    seed = 100
    np.random.seed(seed)
    random.seed(seed)
    
    def tune_classifier(self, X, y):
        # TODO: Tune the hyper-parameters.
        # args: X, y
        # return: RandomizedSearchCV
        
        def objective(params):
            clf = RandomForestClassifier(n_estimators=int(params["n_estimators"]),
                                           max_depth=int(params["max_depth"]),
                                           min_samples_split=int(params["min_samples_split"]),
                                           min_samples_leaf=int(params["min_samples_leaf"]),
                                           max_features=params["max_features"],
                                           random_state=RandomForest.seed,
                                           n_jobs=-1)

            score = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()

            return {"loss": -score,
                    "status": STATUS_OK}
        
        search_space = {"n_estimators": hp.quniform("n_estimators", 100, 600, 50),
                        "max_depth": hp.quniform("max_depth", 3, 20, 1),
                        "min_samples_split": hp.quniform("min_samples_split", 2, 20, 1),
                        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
                        "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
                        "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"])
                        }
        
        trials = Trials()

        best = fmin(fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials,
                    rstate=np.random.default_rng(RandomForest.seed))
        
  
        return best
    
    
    def tuned_classifier(self, X, y, best):
        
        best_params = {"n_estimators": int(best["n_estimators"]),
                       "max_depth": int(best["max_depth"]),
                       "min_samples_split": int(best["min_samples_split"]),
                       "min_samples_leaf": int(best["min_samples_leaf"]),
                       "max_features": ["sqrt", "log2", None][best["max_features"]],
                       "criterion": ["gini", "entropy", "log_loss"][best["criterion"]],
                       "random_state": RandomForest.seed,
                       "n_jobs": -1}
        
        # Retrain final model
        tuned_clf = RandomForestClassifier(**best_params)
        tuned_clf.fit(X, y)
        
        return tuned_clf, best_params
    
    def compute_auc(self, y, y_pred):
        """
        Compute ROC-AUC score.
        Parameters:
        y : array
        y_pred : array

        Returns:
        float
        """
        auc = roc_auc_score(y, y_pred)
        return auc
    
    
    def accuracy(self, y, y_pred):
   
        accuracy = accuracy_score(y, np.ravel(y_pred))
    
        return accuracy
    
    
    def feature_importance(self, clf):
        # TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
        # args: model object
        # return: float array
        
        feature_importance = clf.feature_importances_

        return feature_importance
    
    
    def sorted_feature_importance_indicies(self, clf):
        # TODO: Sort them in the ascending order and return the feature numbers[0 to ...].
        # args: model object
        # return: int array

        sorted_indices = np.argsort(clf.feature_importances_)[::-1]
        # -------------------------------
        return sorted_indices
    
    
    def cross_validation_classifier(self, clf, X, y):
        
        # TODO: Perform cross-validation and return metrics
        # args: classifier, predicting vars for training set, target var for training set
        # return: f1-score, auc, and accuracy
        
        accuracy = cross_val_score(clf, X, np.ravel(y), cv = 5, scoring='roc_auc')
        return accuracy