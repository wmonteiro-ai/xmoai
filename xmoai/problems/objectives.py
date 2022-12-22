#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:58:52 2020

@author: wmonteiro92
"""
import numpy as np
'''
import gower
'''

num_objectives = 3 # number of objectives used by xMOAI

def convert_single_one_hot_feature_to_label_encoded(x, x_original, col):
    # converting one-hot encoded value of a category to an integer
    val_original = x_original[col].dot(1 << np.arange(len(col) - 1, -1, -1))
    x_original = np.append(x_original, val_original)
    
    if len(np.shape(x)) == 2:
        val = x[:, col].dot(1 << np.arange(len(col) - 1, -1, -1))
        x = np.append(x, np.array([val]).T, axis=1)
    else:
        val = x[col].dot(1 << np.arange(len(col) - 1, -1, -1))
        x = np.append(x, val)
        
    return x_original, x
    
def convert_one_hot_to_label_encoded(x, x_original, categorical_columns_one_hot_encoder, \
                                    cat_features=None):                            
    if categorical_columns_one_hot_encoder is None or \
       len(categorical_columns_one_hot_encoder) == 0:
        if cat_features is None:
            return x, x_original
        else:
            return x, x_original, cat_features
       
    for col in categorical_columns_one_hot_encoder:
        x_original, x = convert_single_one_hot_feature_to_label_encoded(x, x_original, col)
        
        if cat_features is not None:
            cat_features = np.append(cat_features, [True])
    
    # removing the one-hot encoded values and appending the integer instead
    indexes = []
    [indexes.extend(l) for l in categorical_columns_one_hot_encoder]
    
    x = np.delete(x, indexes, axis=len(np.shape(x))-1)
    x_original = np.delete(x_original, indexes)
    
    if cat_features is not None:
        cat_features = np.delete(cat_features, indexes)
        return x, x_original, cat_features
    else:
        return x, x_original

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

def get_difference_attributes(x, x_original, ranges, categorical_columns_label_encoder,
                              categorical_columns_one_hot_encoder):
    """Calculates the objective 2 (f2), where it attempts to minimize the 
    difference between the modified and original values through the Gower 
    distance.
    
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param x_original: the original individual
    :type x_original: numpy.array
    :param categorical_columns_label_encoder: dictionary containing the label-encoded
        categorical columns and their allowed values. The keys are the i-th position 
        of the indexes and the values are the allowed categories. The minimum and maximum
        categories must respect the values in lower_bounds and upper_bounds since this variable
        is called after it in code.
    :type categorical_columns_label_encoder: dict
    :param categorical_columns_one_hot_encoder: list of lists containing the one-hot encoded
        categorical columns. Each list inside this list contains the i-th positions of a given
        one-hot encoded column. Example: if a column was encoded into three columns, 
        the i-th positions of these columns are encoded into a list.
    :type categorical_columns_one_hot_encoder: numpy.array
    
    :return: the Gower distance between x and x_original
    :rtype: np.array
    """
    num_cols = x_original.shape
    num_cols = num_cols[0] if len(num_cols) == 1 else num_cols[1]
    
    if categorical_columns_label_encoder is None or \
       len(categorical_columns_label_encoder.keys()) == 0:
        cat_features = np.array([False]*num_cols) 
    else:
        cat_features = np.isin(np.array(range(num_cols)), 
                               np.array([x for x in categorical_columns_label_encoder.keys()]))
    
    # converting one-hot encoded into label-encoded in order to calculate the gower distance
    x, x_original, cat_features = convert_one_hot_to_label_encoded(x, x_original, \
                                                                   categorical_columns_one_hot_encoder, \
                                                                   cat_features)
    '''
    if categorical_columns_one_hot_encoder is not None or \
       len(categorical_columns_one_hot_encoder) >= 0:
        for col in categorical_columns_one_hot_encoder:
            # converting one-hot encoded value of a category to an integer
            val_original = x_original[col].dot(1 << np.arange(len(col) - 1, -1, -1))
            x_original = np.append(x_original, val_original)
            
            if len(np.shape(x)) == 2:
                val = x[:, col].dot(1 << np.arange(len(col) - 1, -1, -1))
                x = np.append(x, np.array([val]).T, axis=1)
            else:
                val = x[col].dot(1 << np.arange(len(col) - 1, -1, -1))
                x = np.append(x, val)
                    
            cat_features = np.append(cat_features, [True])
        
        # removing the one-hot encoded values and appending the integer instead
        indexes = []
        [indexes.extend(l) for l in categorical_columns_one_hot_encoder]
        
        x = np.delete(x, indexes, axis=len(np.shape(x))-1)
        x_original = np.delete(x_original, indexes)
        cat_features = np.delete(cat_features, indexes)
    '''
    
    x = np.nan_to_num(x, nan=-2**32-1)
    x_original = np.nan_to_num(x_original.reshape(1, -1), nan=-2**32-1)
    
    '''
    norm_values = np.nan_to_num(np.linalg.norm(x, ord=1, axis=0, keepdims=True), nan=1)
    norm_values = np.where(norm_values==0, 1, norm_values)
    
    x = x / norm_values
    x_original = x_original / norm_values

    return np.abs(gower.gower_matrix(data_x=x, 
                                     data_y=x_original,
                                     cat_features=cat_features).flatten())
    '''
    cat_features = np.argwhere(cat_features).flatten()
    
    distances = np.abs(np.repeat(x_original, len(x), axis=0) - x)/ranges
    distances[:, cat_features] = (distances[:, cat_features]!=0).astype(int)
    return np.mean(distances, axis=1)
    
def get_modified_attributes(x, x_original, categorical_columns_one_hot_encoder):
    """Calculates the objective 3 (f3), where it attempts to minimize the 
    number of modified attributes (columns).
    
    :param x: the individual (or individuals) to be evaluated
    :type x: numpy.array
    :param x_original: the original individual
    :type x_original: numpy.array
    :param categorical_columns_one_hot_encoder: list of lists containing the one-hot encoded
        categorical columns. Each list inside this list contains the i-th positions of a given
        one-hot encoded column. Example: if a column was encoded into three columns, 
        the i-th positions of these columns are encoded into a list.
    :type categorical_columns_one_hot_encoder: numpy.array
    
    :return: the number of modified attributes for each one of the solutions 
        (rows) provided in x and compared against x_original
    :rtype: np.array
    """
    num_cols = x_original.shape
    num_cols = num_cols[0] if len(num_cols) == 1 else num_cols[1]
	
    # f3: minimize the number of modified attributes
    x, x_original = convert_one_hot_to_label_encoded(x, x_original, \
                                                     categorical_columns_one_hot_encoder)
    
    return np.count_nonzero(np.nan_to_num(x_original, nan=-2**32-1) - 
                            np.nan_to_num(x, nan=-2**32-1), axis=1)/num_cols