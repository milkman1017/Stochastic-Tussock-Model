import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Run $ wget  "https://docs.google.com/spreadsheets/d/1GfVrWWKMBeOzuNMu31YC-pTHkpYPN9guSKM_pvvei5U/export?format=csv&edit#gid=0" -O "parameterization_data.csv" to get most recent data


def parameterizeSurviveEvent(df):
    counts, bins = np.histogram(df['Normal r0'], bins='auto')

    lin_reg = LinearRegression().fit(bins[:-1].reshape(-1,1), counts)

    return np.round(lin_reg.coef_[0],3)

def parameterizeTillerEvent(df):
    return 0.5

def writeParameters(parameters):
    with open('parameters.txt', 'w') as parameter_file:
        for key, value in zip(parameters.keys(), parameters.values()):
            parameter_file.write(f'{key}={value} \n')

def main():
    parameterization_data = pd.read_csv('parameterization_data.csv')

    parameters = dict()

    parameters['ks'] = parameterizeSurviveEvent(parameterization_data)
    parameters['kr'] = parameterizeTillerEvent(parameterization_data)

    writeParameters(parameters)

if __name__ == '__main__':
    main()