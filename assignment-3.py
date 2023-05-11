# -*- coding: utf-8 -*-
"""
Created on Tue May 09 13:02:15 2023

@author: Adeel Warraich
"""

import pandas as pd
import numpy as np

import scipy.optimize as opt
import errors as err

from sklearn import cluster
import sklearn.metrics as skmet

import matplotlib.pyplot as plt
import cluster_tools as ct
from datetime import timedelta, datetime

def PlotScatterGraph(x, y, df, clusters):
    """
    

    Parameters
    ----------
    x : string
        Country 1 name .
    y : string
        country 2 name.
    df : pandas
        dataframe.
    clusters : int
        number of cluster.

    Returns
    -------
    None.
    This is generic method for drawing scatter plot and center points 
    for any two countries
    """
    kmeans = cluster.KMeans(n_clusters=clusters)
    kmeans.fit(df)     

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    plt.figure(figsize=(6.0, 6.0))
    # scatter plot with colours selected using the cluster numbers
    plt.scatter(df[x], df[y], c=labels, cmap="tab10")
    
    # colour map Accent selected to increase contrast between colours
    
    # show cluster centres
    xc = cen[:,0]
    yc = cen[:,1]
    
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"CO2 emissions (metric tons per capita) For {clusters} clusters")
    plt.show()

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
    df_t = df_t.replace(0, pd.np.nan)
    df_t = df_t.dropna(how='all').dropna(axis=1)
    
    # drop columns with entries 0
    # df_t = df_t.loc[:, (df_t != 0)]

    # make year as integer for the graph to avoid cluttering
    df_t.index = pd.to_numeric(df_t.index)

    return df_t



# read data using pandas and skip reference rows
df_co2 = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5455265.csv', skiprows=4)

# transpose data frame and clean empty rows
df_co2_t = TransposeDataFrame(df_co2)

# basic statistics properties
print(df_co2_t.describe())
print()
# print(df_co2_t[df_co2_t.eq(0)].to_csv('ddd.csv'))

top_10 = df_co2_t.sum(axis=0).nlargest(10)

# check the correlation with heatmap using clustor_tools
df_co2_t = df_co2_t.loc[:, top_10.index]
# df_co2_t = df_co2_t.iloc[ : , 10 : 20]

# # print(df_co2_t)
# ct.map_corr(df_co2_t, 9)

# # # scatter plot
# pd.plotting.scatter_matrix(df_co2_t, figsize=(9.0, 9.0))
# # helps to avoid overlap of labels.
# plt.tight_layout()
# plt.show()

# extract columns for normalization 
# use copy method to prevent changes in original dataframe
# df_clus = df_co2_t.iloc[:, 0:2].copy()

df_clus = df_co2_t[['United States', 'Brunei Darussalam']].copy()
df_clus, df_min, df_max = ct.scaler(df_clus)

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_clus)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_clus, labels))


# silhoutte score for 2, 3 and 4 is better let's plot graph for these values
clusters = [2, 3, 4]
for c in clusters:
    PlotScatterGraph("United States", "Brunei Darussalam", df_clus, c)
    
# looking at the plot 3 cluster looks much better than all others 
# let's plot graph graph for 3 on original scale

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_clus)     

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))

# scatter plot with colours selected using the cluster numbers
plt.scatter(df_co2_t["United States"], df_co2_t["Brunei Darussalam"], c=labels, cmap="tab10")

# use backscales to show cluster centres
scen = ct.backscale(cen, df_min, df_max)
xc = cen[:,0]
yc = cen[:,1]

plt.scatter(xc, yc, c="k", marker="d", s=80)
plt.xlabel("United States")
plt.ylabel("Brunei Darussalam")
plt.title("CO2 emissions (metric tons per capita) For 3 clusters")
plt.show()


# Load data
df = pd.read_csv('co2_emissions.csv')
df['year'] = pd.to_datetime(df['year'], format='%Y')

# Define model function
def poly_model(x, a, b, c):
    return a*x**2 + b*x + c

# Fit the model
popt, pcov = opt.curve_fit(poly_model, df.index.values, df['value'])

# Define prediction function
def predict(x):
    return poly_model(x, *popt)

# Get prediction range
n_years = 20
start_date = df['year'].iloc[-1]
end_date = start_date + timedelta(days=365*n_years)
idx = pd.date_range(start=start_date, end=end_date, freq='AS')
pred_range = np.arange(len(df.index), len(df.index) + len(idx))

# Get predictions and error bounds
y_pred = predict(pred_range)
sigma = np.sqrt(np.diag(pcov))
y_upper = err.err_ranges(y_pred, sigma, 0.95)[0]
y_lower = err.err_ranges(y_pred, sigma, 0.95)[1]

# Plot the data and prediction
plt.plot(df['year'], df['value'], 'o', label='Data')
plt.plot(idx, y_pred, '-', label='Prediction')
plt.fill_between(idx, y_lower, y_upper, alpha=0.3, label='95% Confidence Interval')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.legend()
plt.show()
