#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 00:06:08 2020

@author: wmonteiro92
"""
import numpy as np

def get_changed_vars_threshold(current_changed_vars, max_changed_vars):
    """Calculates the constraint 1 (g1), where the number of variables 
    (columns) changed must be less than or equal to the limit defined by
    max_changed_vars.

    :param current_changed_vars: the number of variables changed per individual
    :type current_changed_vars: numpy.array
    :param max_changed_vars: the maximum allowed number of changed variables
    :type max_changed_vars: Integer
    
    :return: an array containing the result of the constraint comparison. Each 
        element represents the results for an individual
    :rtype: np.array
    """
    # g1: current_changed_vars needs to be less than or equal to max_changed_var
    return current_changed_vars - max_changed_vars

def is_prediction_in_threshold_regression(y_acceptable_range, prediction):
    """Calculates the constraints 2 (g2) and 3 (g3). Constraint 2 determines 
    if the predicted value is greater than or equal the minimum acceptable 
    value for the output. Constraint 3 determines if the predicted value is 
    lesser than or equal the maximum acceptable value for the output. Valid 
    only for regression problems.

    :param y_acceptable_range: the lower (first value) and upper 
        (second value) limits allowed for the prediction
    :type y_acceptable_range: numpy.array
    :param prediction: the predicted values
    :type prediction: numpy.array
    
    :return: two objects. The first are the constraint 2 (g2) values and
        the second are the constraint 3 (g3) values. Each element of both objects
        represents the results for an individual.
    :rtype: np.array (first object) and np.array (second object)
    """
    # g2 and g3: the predicted value must fall in a certain range
    g2 = y_acceptable_range[0] - prediction
    g3 = prediction - y_acceptable_range[1]
    return g2, g3

def is_prediction_in_threshold_classification_simple(prediction, y_desired):
    """Calculates the constraint 2 (g2). Constraint 2 determines if the
    predicted value is different from the expected output. Valid only for 
    classification problems with methods returning the predicted class.

    :param y_desired: the targeted value
    :type y_desired: object
    :param prediction: the predicted values
    :type prediction: numpy.array
    
    :return: an array containing the result of the constraint comparison. Each 
        element represents the results for an individual
    :rtype: np.array
    """
    g2 = np.where(prediction==y_desired, 0, 1)
    return g2
        
def is_prediction_in_threshold_classification_proba(y_acceptable_range, \
                                              prediction, prob_column):
    """Calculates the constraints 2 (g2) and 3 (g3). Constraint 2 determines 
    if the predicted value is greater than or equal the minimum acceptable 
    value for the output. Constraint 3 determines if the predicted value is 
    lesser than or equal the maximum acceptable value for the output. Valid 
    only for classification problems with methods returning the probability 
    estimates for each class.

    :param y_acceptable_range: the lower (first value) and upper 
        (second value) limits allowed for the probability output
    :type y_acceptable_range: numpy.array
    :param prediction: the predicted probabilities
    :type prediction: numpy.array
    :param prob_column: the i-th column to be observed in the `prediction` 
        parameter
    :type prob_column: Integer
    
    :return: two objects. The first are the constraint 2 (g2) values and
        the second are the constraint 3 (g3) values. Each element of both objects
        represents the results for an individual.
    :rtype: np.array (first object) and np.array (second object)
    """
    # g2 and g3: the predicted value must fall in a certain range
    g2 = y_acceptable_range[0] - prediction[:, prob_column]
    g3 = prediction[:, prob_column] - y_acceptable_range[1]
    return g2, g3