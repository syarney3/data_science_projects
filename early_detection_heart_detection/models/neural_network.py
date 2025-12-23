from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


class NeuralNetwork():

    seed = 100
    np.random.seed(seed)
    random.seed(seed)

    def build_model(self, params, input_shape):

        model = Sequential()
        model.add(Input(shape=input_shape))  # Use Input layer here
        model.add(Dense(params['units'], activation='relu'))
        model.add(Dropout(params['dropout']))
        model.add(Dense(params['units'], activation='relu'))
        model.add(Dropout(params['dropout']))
        model.add(Dense(params['units'], activation='relu'))
        model.add(Dropout(params['dropout']))
        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(loss='binary_crossentropy', 
                      optimizer=optimizer, 
                      metrics=['AUC'])
        
        return model
    
    def tune_classifier(self, X_train, y_train, X_val, y_val):

        """
        Tune hyper-parameters
        Parameters:
        X_train : array
        y_train : array
        X_val: array
        y_val: array

        Returns:
        dictionary
        """

        def objective(params):

            tf.keras.backend.clear_session()
            model = self.build_model(params, input_shape=(X_train.shape[1],))
            model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=params['epochs'],
                                batch_size=params['batch_size'],
                                verbose=0)
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)

            return {'loss': -auc,
                    'status': STATUS_OK
            }
        
        space = {'units': hp.choice('units', [32, 64, 128]),
                 'learning_rate': hp.loguniform('learning_rate', -5, -2),
                 'dropout': hp.uniform('dropout', 0.1, 0.5),
                 'epochs': hp.choice('epochs', [5, 10, 15]),
                 'batch_size': hp.choice('batch_size', [16, 32, 64])}

        trials = Trials()
        best_params = fmin(fn=objective,
                           space=space,
                           algo=tpe.suggest,
                           max_evals=100,
                           trials=trials)

        best_params_mapped = space_eval(space, best_params)
        
        return best_params_mapped
        
        
    def fit_classifier(self, X_train, y_train, best_params):
        """
        Fit final model with optimal hyper-parameters
        Parameters:
        y : array
        y_pred : array
        best: dictionary

        Returns:
        object, dictionary
        """
        
        final_model = self.build_model(best_params, input_shape=(X_train.shape[1],))
        final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)
        
        return final_model
    
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
        """
        Compute Accuracy.
        Parameters:
        y : array
        y_pred : array

        Returns:
        float
        """
   
        accuracy = accuracy_score(y, np.ravel(y_pred))
    
        return accuracy
    
    def compute_f1(self, y, y_pred):
        """
        Compute the F1 score.

        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels

        Returns:
            float: F1 score
        """
        return f1_score(y, y_pred, average='binary')
    
    
 
    