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


def TransposeDataFrame(df):
    """
    this method takes the pandas dataframe
    convert into transposed dataframe.
    """
    # transpose the dataframe
    df_t = df.transpose()

    # set first row as columns of transpose df
    df_t.columns = df_t.iloc[0]

    # remove non numerical data from dataframe
    df_t = df_t.iloc[4:]
    
    # rename the index column
    df_t.index.name = 'Year'

    # clean the transposed dataframe those are completely empty and then drop column with missing entries
    df_t = df_t.dropna(how='all').dropna(axis=1)

    # make year as integer for the graph to avoid cluttering
    df_t.index = pd.to_numeric(df_t.index)

    return df_t

# read data using pandas and skip reference rows
df_co2 = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5455265.csv', skiprows=4)

# transpose data frame and clean empty rows
df_co2_t = TransposeDataFrame(df_co2)

# get get data for united kingdom
print(df_co2_t.to_csv('test.csv'))
# get get data for usa
# df_uk = TransposeDataFrame(df_co2, 'United Kingdom')

