import numpy as np
import pandas as pd

import bentoml
from bentoml.frameworks.lightgbm import LightGBMModelArtifact
from bentoml.adapters import JsonInput


# separate feature spaces for each model (casual and registered users)
features1 = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_casual',
            'rolling_mean_12_hours_casual','season', 'yr', 'mnth',  
            'day_type', 'weathersit', 'CasualHourBins', 'weekday']

features2 = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_registered',
            'rolling_mean_12_hours_registered', 'season', 'yr', 'mnth',  
            'day_type', 'weathersit', 'RegisteredHourBins', 'weekday']

# function to prepair data for prediction
# data is in JSON format
def preprocess(X):
        
    # normalize weather data
    X['temp'] += 8
    X['temp'] /= (39 + 8)
    X['windspeed'] /= 67
    X['hum'] /= 100
    
    # creating 'day_type' feature
    ### 2 - working day
    ### 1 - weekend
    ### 0 - holiday
    if X['holiday'] == 1:
        X['day_type'] = 0
    elif (X['weekday'] == 6) or (X['weekday'] == 0):
        X['day_type'] = 1
    else:
        X['day_type'] = 2
    # don't need 'holiday' feature anymore
    del X['holiday']
    
    # binning hour into 'RegisteredHourBins' feature
    bins = np.array([1.5, 5.5, 6.5, 8.5, 16.5, 18.5, 20.5, 22.5])
    labels = np.arange(len(bins)-1)
    label = pd.cut([X['hr']], bins=bins, labels=labels).fillna(0).astype(int)[0]
    remap_labels = {0: 0, 1: 1, 2: 5, 3: 3, 4: 6, 5: 4, 6: 2}
    X['RegisteredHourBins'] = remap_labels[label]
    
    # binning hour into 'CasualHourBins' feature
    bins = np.array([7.5, 8.5, 10.5, 17.5, 19.5, 21.5])
    labels = np.arange(len(bins)-1)
    label = pd.cut([X['hr']], bins=bins, labels=labels).fillna(0).astype(int)[0]
    remap_labels = {0: 0, 1: 2, 2: 4, 3: 3, 4: 1}
    X['CasualHourBins'] = remap_labels[label]
    
    # predicting future, so year is 1
    X['yr'] = 1
    
    # constructing vectors of the sample
    X_1 = np.array([X[feature] for feature in features1]).reshape(1, -1)
    X_2 = np.array([X[feature] for feature in features2]).reshape(1, -1)
    
    return X_1, X_2

# postprocessing function
# if prediction is negative - it's clipped to zero
# float output  is rounded and converted to integer
def postprocess(prediction):
    return int(np.around(prediction.clip(0), 0)[0])

@bentoml.env(infer_pip_packages=True)

@bentoml.artifacts([LightGBMModelArtifact('model_casual'),
                    LightGBMModelArtifact('model_registered')])
class BikeRentalsPredictionService(bentoml.BentoService):

    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, new_data):
        
        X_casual, X_registered = preprocess(new_data)
     
        casual = postprocess(self.artifacts.model_casual.predict(X_casual))
        registered = postprocess(self.artifacts.model_registered.predict(X_registered))

        return casual + registered