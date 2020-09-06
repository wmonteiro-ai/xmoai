#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:58:52 2020

@author: wmonteiro92
"""
import numpy as np
import gower

num_objectives = 3 # number of objectives used by xMOAI

def get_difference_target_regression(model, x, y_desired, method='predict'):
    """Calculates the objective 1 (f1), where it attempts to minimize the 
    unsigned difference between y_desired and the value found by the 
    prediction found for the proposed x - i.e. between the target and the 
    value found by the subject. Valid only for regression problems.

    :param model: a machine learning model
    :type model: Object
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param y_desired: the prediction result targeted
    :type y_desired: numpy.array
    :param method: the method responsible of determining the prediction
    :type method: string, defaults to `predict`
    
    :return: two objects. The first are the objective 1 (f1) values and
        the second are the predicted values related to `x` and found by `model` 
        using `method`
    :rtype: np.array (first object) and np.array (second object)
    """
    prediction = getattr(model, method)(x)
    return np.abs(prediction - y_desired), prediction

def get_difference_target_classification_proba(model, x, class_column,
                                               method='predict_proba'):
    """Calculates the objective 1 (f1), where it attempts to maximize the 
    probability of the desired class. Valid only for classification problems 
    with methods returning the probability estimates for each class.

    :param model: a machine learning model
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param class_column: the column index of the prediction class targeted
    :type class_column: Integer
    :param method: the method responsible of determining the prediction
    :type method: string, defaults to `predict_proba`
    
    :return: two objects. The first are the objective 1 (f1) values and
        the second are the predicted values related to `x` and found by `model` 
        using `method`
    :rtype: np.array (first object) and np.array (second object)
    """
    prediction = getattr(model, method)(x)
    return 1 - prediction[:, class_column], prediction

def get_difference_target_classification_simple(model, x, y_desired,
                                                method='predict'):
    """Calculates the objective 1 (f1), where it assigns 1 if the predicted 
    class differs from the desired class and 0 otherwise. Valid only for 
    classification problems with methods returning the predicted class.

    :param model: a machine learning model
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param y_desired: the class targeted
    :type y_desired: object
    :param method: the method responsible of determining the prediction
    :type method: string, defaults to `predict`
    
    :return: two objects. The first are the objective 1 (f1) values and
        the second are the predicted values related to `x` and found by `model` 
        using `method`
    :rtype: np.array (first object) and np.array (second object)
    """
    prediction = getattr(model, method)(x)
    return np.where(prediction==y_desired, 0, 1), prediction

def get_difference_attributes(x, x_original, categorical_columns):
    """Calculates the objective 2 (f2), where it attempts to minimize the 
    difference between the modified and original values through the Gower 
    distance.
    
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param x_original: the original individual
    :type x_original: numpy.array
    :param categorical_columns: the categorical columns used by the dataset
    :type categorical_columns: dict
    
    :return: the Gower distance between x and x_original
    :rtype: np.array
    """
    if categorical_columns==None or len(categorical_columns.keys()) == 0:
        cat_features = np.array([False]*x_original.shape[0])
    else:
        cat_features = np.isin(np.array(range(x_original.shape[0])), 
                               np.array([x for x in categorical_columns.keys()]))
    
    return gower.gower_matrix(data_x=np.nan_to_num(x, nan=-2**32-1), 
                              data_y=np.nan_to_num(x_original.reshape(1, -1), nan=-2**32-1),
                              cat_features=cat_features).flatten()
    
def get_modified_attributes(x, x_original):
    """Calculates the objective 3 (f3), where it attempts to minimize the 
    number of modified attributes (columns).
    
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param x_original: the original individual
    :type x_original: numpy.array
    
    :return: the number of modified attributes for each one of the solutions 
        (rows) provided in x and compared against x_original
    :rtype: np.array
    """
    # f3: minimize the number of modified attributes
    return np.count_nonzero(np.nan_to_num(x_original, nan=-2**32-1) - 
                            np.nan_to_num(x, nan=-2**32-1), axis=1)