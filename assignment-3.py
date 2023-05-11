# -*- coding: utf-8 -*-
"""
Created on Tue May 09 13:02:15 2023

@author: Adeel Warraich
"""

import pandas as pd
import numpy as np

from sklearn import cluster
import sklearn.metrics as skmet

import matplotlib.pyplot as plt
import cluster_tools as ct


def TransposeDataFrame(df, country):
    """
    this method takes the pandas dataframe
    along with the indicator name and country name 
    and convert into transposed dataframe.
    """
    # df_country = df.loc[df['Country Name'] == country]

    # transpose the dataframe
    df_country = df.transpose()

    # set first row as columns of transpose df
    df_country.columns = df_country.iloc[0]

    # remove non numerical data from dataframe
    df_country = df_country.iloc[4:]
    
    # rename the index column
    df_country.index.name = 'Year'

    # clean the transposed dataframe those are completely empty and then drop column with missing entries
    df_country = df_country.dropna(how='all').dropna(axis=1)

    # make year as integer for the graph to avoid cluttering
    df_country.index = pd.to_numeric(df_country.index)

    return df_country

# read data using pandas and skip reference rows
df_co2 = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5455265.csv', skiprows=4)


# get get data for united kingdom
df_co2_t = TransposeDataFrame(df_co2, 'United Kingdom')
print(df_co2_t)
# get get data for usa
# df_uk = TransposeDataFrame(df_co2, 'United Kingdom')

