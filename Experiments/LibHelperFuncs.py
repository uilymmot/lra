# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:48:51 2020

@author: tommy
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import copy
import scipy.spatial.distance
from sklearn.metrics.pairwise import pairwise_distances
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
import subprocess
from sklearn.model_selection import RandomizedSearchCV
import sys
import math
import scipy.spatial.distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.tree import DecisionTreeRegressor
import os
import shutil
import glob
import shap
from scipy.stats import pearsonr


cmap='viridis'
from sklearn.impute import KNNImputer
rstate = 0

# Takes pandas dataframe
# scales data into range [0,1]
def scale_data(dframe):
    data_scaled = copy.deepcopy(dframe)
    for x in data_scaled.columns:
        if not (isinstance(data_scaled[x][0], str)):
            if (max(data_scaled[x] != 0)):
                data_scaled[x] += abs(min(data_scaled[x]))
                data_scaled[x] /= max(data_scaled[x])
    return data_scaled

# takes pandas dataframe
# remove columns with zero variance
def remove_zero_var(dframe):
    data_zero = copy.deepcopy(dframe)
    ind = np.where(np.var(data_zero, axis=0)==0)[0]
    return data_zero.drop(data_zero.columns[ind], axis=1)

# takes pandas dataframe
# replace NAN values with KNN neighbours
def impute_data_knn(frame, n):
    imputer = KNNImputer(n_neighbors=n)
    impdat = imputer.fit_transform(frame)
    return pd.DataFrame(impdat, columns=frame.columns)


def corr_filter(dframe, criteria):
    '''
    A correlation filter on a dataframe that selects feature to filter based on which feature
    has the greater shap value

    Parameters
    ----------
    dframe : pd.dataframe
        A dataframe containing data to filter
    criteria : float
        Lower bound for correlation value

    Returns
    -------
    filt_dat : pd.dataframe
        The reduced/filtered dataframe
    rem_dat : list (tuple)
        A list of tuples containing column-names and shap values of features
        that have been removed.

    '''
    
    filt_dat = copy.deepcopy(dframe)
    rem_dat = set()
    while (True):
        c = filt_dat.corr().abs()
        c = (c.where(np.triu(np.ones(c.shape), k=1).astype(np.bool))
                     .stack()
                     .sort_values(ascending=False))
        if (c[c<1][0] < criteria):
            break

        a = c.index[0][0]
        
        filt_dat = filt_dat.drop(a, axis=1)
        rem_dat.add(a)
            
    return (filt_dat, rem_dat)
    
def shap_corr_filter(dframe, criteria, shap_vals, datacolumns):
    '''
    A correlation filter on a dataframe that selects feature to filter based on which feature
    has the greater shap value

    Parameters
    ----------
    dframe : pd.dataframe
        A dataframe containing data to filter
    criteria : float
        Lower bound for correlation value
    shap_vals : np.ndarray
        1d numpy array containing feature importances
    datacolumns : np.ndarray
        the columns-names of data

    Returns
    -------
    filt_dat : pd.dataframe
        The reduced/filtered dataframe
    rem_dat : list (tuple)
        A list of tuples containing column-names and shap values of features
        that have been removed.

    '''
    removed = set()
    filt_dat = copy.deepcopy(dframe)
    
    c = filt_dat.corr().abs()
    c = (c.where(np.triu(np.ones(c.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
    
    for x in c[c > criteria].index:
        a = x[0]
        b = x[1]

        if (not (a in removed) and not (b in removed)):
            ashap = shap_vals[np.where(datacolumns == a)[0]]
            bshap = shap_vals[np.where(datacolumns == b)[0]]

            if (ashap > bshap):
                removed.add(b)
            else:
                removed.add(a)

    filt_dat = filt_dat.drop(list(removed), axis=1)
    return filt_dat, removed
    
def compute_homogeneous_components(pca_comp, k_percent, normalise=True):
    '''
    given a set of principal components, applies homogeneous constraints according to "Interpretable Dimension Reduction"
    
    @Params:
        pca_comp: the components of the PCA decomposition
        k_percent: desired number of top k components to keep
    @Return:
        new_components: a set of interpretable principal components according
        to homogeneity constraints
    '''
    assert(0 < k_percent <= 1)
    
    new_components = np.zeros(pca_comp.shape)
    for c in range(pca_comp.shape[0]):
        current_component = pca_comp[c]
        abs_curr_component = np.abs(current_component)
        k_value = int(k_percent * abs_curr_component.shape[0])
        if (k_value == 0):
            break
        k_largest = np.argsort(abs_curr_component)[::-1][:k_value]
        maskarr = np.zeros((*abs_curr_component.shape, k_value))
        
        counter = 0
        for i in k_largest:
            temparr = maskarr[:,counter]
            temparr[i] = math.copysign(1, current_component[i])
            counter+=1
            if (counter == k_value):
                break
            maskarr[:,counter] = temparr
        distances = np.apply_along_axis((lambda x : (abs_curr_component @ x)),\
                                        0,  maskarr)
        new_components[c] = maskarr[:,np.argmax(distances)]
        if (normalise):
            new_components[c] = new_components[c]\
                / np.sqrt(np.sum(np.abs(new_components[c])))
    return new_components

def compute_sparse_components(comps, x_shape, nu):
    '''
    given a set of principal components, applies sparse constraints according to "Interpretable Dimension Reduction"

    Parameters
    ----------
    comps : np.ndarray
        the columnwise basis vectors
    x_shape : int
        the number of features in the original unprojected data
    nu : float
        the tuning parameter nu corresonding to weighting penalty

    Returns
    -------
    newcomponents : np.ndarray
        the sparse components

    '''
    p = x_shape
    newcomponents = np.zeros(comps.shape)

    for nc in range(comps.shape[1]):

        dat = comps[:,nc]
        bufferarray = np.repeat(dat, dat.shape[0]).reshape((dat.shape[0], dat.shape[0]))
        sorted_dat = np.argsort(np.abs(dat))
        for i in range(dat.shape[0]):
            bufferarray[i,sorted_dat[:i]] = 0

        def criteria(c1):
            if (np.linalg.norm(c1) == 0):
                return 1
            nonzero = np.where(c1 != 0)[0].shape[0]
            return (np.arccos(c1 @ dat)) / (np.pi / 2) + (nu * nonzero / p)

        sparsityscores = np.abs(np.apply_along_axis(criteria, 1, bufferarray))

        sparseComp = bufferarray[:,np.argmin(sparsityscores)]

        newcomponents[:,nc] = sparseComp
        
    return newcomponents
    
    
def linereg_model_crosseval(X, Y):
    '''
    Evaluates a linear regression model cross validated 5 times on the dataset

    Parameters
    ----------
    X : ndarray
        The array of feature vectors (stored rowwise)
    Y : ndarray
        The array of target labels

    Returns
    -------
    score : float
        The RMSE score obtained by the linear regression model

    '''
    linreg = LinearRegression()
    score = np.mean(cross_val_score(linreg, X, Y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
    return np.abs(np.mean(score))
    
def rf_model_crosseval(X, Y, ntrees):
    '''
    Evaluates a random forest model cross validated 5 times on the dataset

    Parameters
    ----------
    X : ndarray
        The array of feature vectors (stored rowwise)
    Y : ndarray
        The array of target labels

    Returns
    -------
    score : float
        The RMSE score obtained by the random forest model

    '''
    rf = RandomForestRegressor(n_estimators=ntrees, random_state=rstate, n_jobs=-1)
    score = np.mean(cross_val_score(rf, X, Y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
    return np.abs(np.mean(score))

def model_crosseval(X, Y, model):
    '''
    Evaluates a random forest model cross validated 5 times on the dataset

    Parameters
    ----------
    X : ndarray
        The array of feature vectors (stored rowwise)
    Y : ndarray
        The array of target labels

    Returns
    -------
    score : float
        The RMSE score obtained by the random forest model

    '''
    score = np.mean(cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
    return np.abs(np.mean(score))

def linereg_model_eval(X, Y, tsize=0.9, rstate=0):
    '''
    Evaluates a linear regression model on the whole dataset

    Parameters
    ----------
    X : ndarray
        The array of feature vectors (stored rowwise)
    Y : ndarray
        The array of target labels

    Returns
    -------
    score : float
        The RMSE score obtained by the linear regression model

    '''
    linreg = LinearRegression()
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, train_size=tsize, random_state=rstate)
    linreg.fit(X_tr,Y_tr)
    score = np.sqrt(np.mean((linreg.predict(X_te) - Y_te) ** 2))
    return score

def model_eval(X, Y, model, tsize=0.9, rstate=0):
    '''
    Evaluates a model using a train test split

    Parameters
    ----------
    X : ndarray
        The array of feature vectors (stored rowwise)
    Y : ndarray
        The array of target labels

    Returns
    -------
    score : float
        The RMSE score obtained by the linear regression model

    '''
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, train_size=tsize, random_state=rstate)
    model.fit(X_tr,Y_tr)
    score = np.sqrt(np.mean((model.predict(X_te) - Y_te) ** 2))
    return score
    
def rf_model_eval(X, Y, ntrees=1000, tsize=0.9):
    '''
    Evaluates a random forest model on the whole dataset

    Parameters
    ----------
    X : ndarray
        The array of feature vectors (stored rowwise)
    Y : ndarray
        The array of target labels

    Returns
    -------
    score : float
        The RMSE score obtained by the random forest model

    '''
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, train_size=tsize, random_state=rstate)
    rf = RandomForestRegressor(n_estimators=ntrees, random_state=rstate, n_jobs=-1)
    rf.fit(X_tr,Y_tr)
    score = np.sqrt(np.mean((rf.predict(X_te) - Y_te) ** 2))
    return score
    
def pointwise_shap_contributions(components_of_interest, x_shape, current_shap, dr_components):
    '''
    Computes the total weighted shap from each of the features for a particular observation of data
    @Params:
        components_of_interest: the array indices of most contributing shap values
        x_shape: the number of columns in the original unprojected data
        current_shap: the list of shap values for this set of components
        dr_components: the components in the dimension reduction model
    @Return:
        temparr: the average weighted shap contribution of each of the features across the components of interest
    '''
    temparr = np.zeros(x_shape)
    for c in range(len(components_of_interest)):
        current_shap_value = current_shap[components_of_interest[c]]
        
        componentwise_shap = dr_components[c] * current_shap_value
        
        temparr += componentwise_shap
    return temparr
        
def total_mean_shap_contributions(x_shape, shap_values, lower_thres, dr_components):
    '''
    Computes the total weighted shap from each of the features for all observations of a data set
    @Params:
        x_shape: the number of columns in the original unprojected data
        shap_values: all shap values for this dataset
        lower_thres: the threshold to cut off components based on shapley value
        dr_components: the components in the dimension reduction model
    @Return:
        final_shap_array: the average weighted shap contribution of each of the features across the whole dataset
    '''
    final_shap_array = np.zeros(x_shape)
        
    for comp_no in range(shap_values.shape[0]):
        current_shap = shap_values[comp_no]
        
        components_of_interest = np.where(np.abs(current_shap) > lower_thres)[0]
        
        current_shap_contributions = pointwise_shap_contributions(components_of_interest, x_shape, current_shap, dr_components)
        
        final_shap_array += current_shap_contributions
    return final_shap_array

def compute_carried_shap(s_vals, comps, X):
    shap_values_r = np.arange(0, X.shape[0]).reshape(-1, 1)
    carried_shap_vals = np.apply_along_axis((lambda x : s_vals[x].reshape(-1, 1).T @ comps), 1, shap_values_r).reshape(-1, X.shape[1])
    return carried_shap_vals

def mean_carried_shap(s_vals, comps, X):
    t = compute_carried_shap(s_vals, comps, X)
    return np.mean(np.abs(t), axis=0)
    
def evaluate_all_linreg(targetFolder, X, Y):
    '''
    Parameters
    ----------
    targetFolder : str
        Folder containing the dimension reduction components
    X : ndarray
        The feature vectors (stored rowwise)
    Y : ndarray
        The target labels (rowwise)

    Returns
    -------
    temparrscore : array
        An array of cross-validated linear regression scores
    temparrnames : TYPE
        An array of the names of the method corresponding to the score

    '''
    temparrscore = []
    temparrnames = []
    for f in glob.glob(str(targetFolder + "*.csv")):
        dr_comp = np.genfromtxt(f, delimiter=',')
        try:
            reduced_data = (dr_comp @ X.T).T

            crossvalscore = linereg_model_crosseval(reduced_data, Y)
            
            temparrscore.append(crossvalscore)
            temparrnames.append(f)
        except:
            print("Failed on {}".format(f)) # data is stored in incorrect format
    return (temparrscore, temparrnames)
    
def construct_dict_from_dfrow(dfrow, dfcols):
    '''
    Takes a row of a dataframe and turns it into a dictionary based on the column
    headers of the dataframe
    
    Parameters
    ----------
    dfrow : array
        A row of data values from a dataframe
    dfcols : array
        The column headers of the dataframe

    Returns
    -------
    tempdict : dictionary
        a dictionary consisting of the columns and row items in the dataframe row
    '''
    tempdict = dict()
    for x in range(len(dfrow)):
        tempdict[dfcols[x]] = dfrow[x]
    return tempdict
    
def uniform_generate_components(data, target, params, k, rstate):
    '''
    Generates a set of DR components using a parameter list and uniform data

    Parameters
    ----------
    data : np.ndarray
        a set of values X of data
    target : np.ndarray
        a set of labels Y of data
    params : list
        a list of parameter values from see below
    k: int
        The number of components
    rstate : int
        A random state

    Returns
    -------
    temp_array : array
        an array of Dimension Reduction Components

    '''
    passdict = dict()

    for x in params:
        passdict[x] = data

    dat = generate_components(passdict, target, params, k, rstate)

    temp_array = []
    
    for k in dat.keys():
        temp_array.append(dat[k])
        
    return temp_array
    
def generate_components(data, target, paramslist, ncomps, rstate):
    '''
    Generates a set of DR components using a parameter list and data

    Parameters
    ----------
    data : dictionary
        a dictionary of X data
    target : np.ndarray
        a set of labels Y of data
    paramslist : list
        a list of parameter values from see below
    ncomps: int
        the number of components in the DR models
    rstate : int
        A random state

    Returns
    -------
    dr_dict : dict
        a dictionary of Dimension Reduction Components

    '''
    dr_dict = dict()

    if ("pca_standard.csv" in paramslist):
        pca = PCA(n_components=ncomps)
        pca.fit(data['pca_standard.csv'])
        comps = pca.components_
        dr_dict["pca_standard.csv"] = comps

    if ("pca_homogeneous_0.5.csv" in paramslist):
        pca = PCA(n_components=ncomps)
        pca.fit(data['pca_homogeneous_0.5.csv'])
        comps = pca.components_
        homo_comps = compute_homogeneous_components(comps, 0.5)
        dr_dict['pca_homogeneous_0.5.csv'] = homo_comps

    if ("pca_sparse_homogeneous_0.5.csv" in paramslist):
        spca = PCA(n_components=ncomps)
        spca.fit(data['pca_sparse_homogeneous_0.5.csv'])
        scomps = spca.components_
        scomps_homo = compute_sparse_components(scomps, data['pca_sparse_homogeneous_0.5.csv'].shape[1], 0.5)
        scomps_homo = compute_homogeneous_components(scomps_homo, 0.5)
        dr_dict['pca_sparse_homogeneous_0.5.csv'] = scomps_homo
        
    if ("pca_sparse_0.5.csv" in paramslist):
        spca = PCA(n_components=ncomps)
        spca.fit(data['pca_sparse_0.5.csv'])
        scomps = spca.components_
        scomps_homo = compute_sparse_components(scomps, data['pca_sparse_0.5.csv'].shape[1], 0.5)
        dr_dict['pca_sparse_0.5.csv'] = scomps_homo

    if ("nmf_H_matrix_homogeneous_0.5.csv" in paramslist):
        nmf = NMF(n_components=ncomps)
        nmf.fit(data['nmf_H_matrix_homogeneous_0.5.csv'], target)
        nmfcomps = nmf.components_
        nmfcomps_homo = compute_homogeneous_components(nmfcomps, 0.5)
        dr_dict['nmf_H_matrix_homogeneous_0.5.csv'] = nmfcomps_homo

    if ("ISM_squa_homogeneous_0.5.csv" in paramslist):
        line_k = kernels.squared.squared
        sdr_class = sdr(data['ISM_squa_homogeneous_0.5.csv'], target, line_k, q=ncomps)
        sdr_class.train()
        sdr_comps = sdr_class.get_projection_matrix().T
        dr_dict['ISM_squa_homogeneous_0.5.csv'] = sdr_comps

    if ("ISM_poly_homogeneous_0.5.csv" in paramslist):
        line_k = kernels.polynomial.polynomial
        sdr_class = sdr(data['ISM_poly_homogeneous_0.5.csv'], target, line_k, q=ncomps)
        sdr_class.train()
        sdr_comps = sdr_class.get_projection_matrix().T
        dr_dict['ISM_poly_homogeneous_0.5.csv'] = sdr_comps

    if ("ISM_gaus_homogeneous_0.5.csv" in paramslist):
        line_k = kernels.gaussian.gaussian
        sdr_class = sdr(data['ISM_gaus_homogeneous_0.5.csv'], target, line_k, q=ncomps)
        sdr_class.train()
        sdr_comps = sdr_class.get_projection_matrix().T
        dr_dict['ISM_gaus_homogeneous_0.5.csv'] = sdr_comps
    
    return dr_dict
    
def corr_shap(df, shap_vals, colnames, termination_thres, num_elements):
    '''
    Filters a dataframe based on correlation values, informs decision using shap value

    Parameters
    ----------
    df : pd.DataFrame
        a dataframe of data X
    shap_vals : np.ndarray
        the shap values for each of the features
    colnames : np.ndarray
        an array of column names
    termination_thres : float
        the minimum correlation value to filter
    num_elements : int
        the number of elements to filter at this time set

    Returns
    -------
    removed : list
        a list of column names to remove
    thres : float
        the lowest correlation value from the set of columns to remove

    '''
    removed = []
    thres = 1
    
    c = df.corr().abs()
    c = (c.where(np.triu(np.ones(c.shape), k=1).astype(np.bool))
                     .stack()
                     .sort_values(ascending=False))
    c = c[c<1]
    corrs = []
    
    for i in range(num_elements):
        temp = c.index[i]

        a = temp[0]
        b = temp[1]

        if (not (a in removed) and not (b in removed)):
            ashap = shap_vals[np.where(colnames == a)[0]]
            bshap = shap_vals[np.where(colnames == b)[0]]

            if (ashap > bshap):
                removed.append(b)
            else:
                removed.append(a)
                
            corrs.append(c[i])
        
        thres = c[i]
        if (c[i] < termination_thres):
            break
        
    return removed, thres
    
    
def get_indices(columns, values):
    '''
    Takes a np array and a set of value, returns the corresponding indices where 
    values exists in columns

    Parameters
    ----------
    columns : np.ndarray
    values : list

    Returns
    -------
    temp : list

    '''
    temp = []
    
    for f in values:
        temp.append(np.where(columns == f)[0][0])
    return temp
    
def compute_carried_shap(s_vals, comps, X):
    shap_values_r = np.arange(0, X.shape[0]).reshape(-1, 1)
    carried_shap_vals = np.apply_along_axis((lambda x : s_vals[x].reshape(-1, 1).T @ comps), 1, shap_values_r).reshape(-1, X.shape[1])
    return carried_shap_vals

def mean_carried_shap(s_vals, comps, X):
    '''

    Parameters
    ----------
    s_vals : np.ndarray
        The raw shapley values of the lower dimensional data
    comps : np.ndarray
        The array of components used in the PCA transformation
    X : np.ndarray
        The original data X which was projected using comps

    Returns
    -------
    np.ndarray
        An array of feature importances of the original features in X

    '''
    
    sump = np.sqrt(np.mean(comps ** 2, axis=0))
    sump[np.where(sump == 0)[0]] = 1
    sump = sump ** 2
    sump[np.where(sump < 1e-8)[0]] = 1
    
    t = compute_carried_shap(s_vals, comps, X) / sump
    return np.mean(np.abs(t), axis=0) / X.shape[0]
    

    
    
    
    
    
    
    
    
    
    