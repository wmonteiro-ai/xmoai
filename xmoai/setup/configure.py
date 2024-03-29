#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:00:34 2020

@author: wellington
"""

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from xmoai.problems.xMOAIProblem import RegressionProblem, \
    ClassificationProblemProbability, ClassificationProblemSimple
from xmoai.problems.xMOAIRepair import xMOAIRepair
from xmoai.problems.objectives import num_objectives

from topsis import topsis

from random import randint

def get_nondominated_solutions(X_current, select_best, res):
    """Retrieve only the non-dominated solutions found by the optimization
    output from all algorithms. Do not use this method directly. Instead, 
    use the methods starting with `generate_counterfactuals_*`.

    :param X_current: the original individual
    :type X_current: numpy.array
    :param select_best: whether we should run TOPSIS to prune the best solutions or not
    :type select_best: bool
    :param res: contains all the counterfactuals found per algorithm.
    :type res: dict
    
    :return: the non-dominated counterfactuals found by xMOAI in regard to
        the three objectives (i.e. the best counterfactuls considering the trade-offs
        between the three objectives) in a tuple. The first parameter has all the
        values found considering the three objectives (Pareto front), the second has
        the individual values (Pareto set) and the third has the algorithm responsible
        of generating each counterfactual.
    :rtype: np.array, np.array, np.array
    """
    F = np.empty([0, num_objectives])
    X = np.empty([0, X_current.shape[0]])
        
    algorithm = np.empty([0, 1])
    for result in res:
        if res[result].F is None:
            continue
            
        F = np.concatenate([F, res[result].F])
        X = np.concatenate([X, res[result].X])

        algorithm = np.concatenate([algorithm, \
                                    np.array([result]*(res[result].F.shape[0])).reshape(-1, 1)])
    
    # selecting only the non-dominated solutions
    dominated_indexes = []
    for index in range(F.shape[0]):
        solution = F[index, :]
        if np.sum(np.all(solution >= F, axis=1)) > 1:
            dominated_indexes.append(index)
 
    F = np.delete(F, dominated_indexes, axis=0)
    X = np.delete(X, dominated_indexes, axis=0)
    algorithm = np.delete(algorithm, dominated_indexes, axis=0)
    
    if not select_best or len(F) > 1:
        return F, pd.DataFrame(X, columns=X_current.index), algorithm
    
    # ranking the best candidates through TOPSIS
    topsis_weights = [1/num_objectives]*num_objectives
    topsis_cost = [0]*num_objectives
    
    decision = topsis(F, topsis_weights, topsis_cost)
    decision.calc()
    
    #topsis_count_cf = int(F.shape[0]/2)
    #best_indexes = np.argsort(decision.C)[-topsis_count_cf:]
    best_indexes = np.argsort(decision.C)
    return F[best_indexes], pd.DataFrame(X[best_indexes], columns=X_current.index), algorithm[best_indexes]

def get_algorithms(X_current, max_changed_vars, \
                   categorical_columns_label_encoder, \
                   categorical_columns_one_hot_encoder, \
                   upper_bounds, lower_bounds, immutable_column_indexes, \
                   integer_columns, pop_size, prob_mating, seed):  
    """Retrieve the list of algorithms to be used in the optimization process.
    Do not use this method directly. Instead, use the methods starting with
    `generate_counterfactuals_*`.

    :param X_current: the original individual
    :type X_current: numpy.array
    :param max_changed_vars: the maximum number of attributes to be
        modified. Default is None, where all variables may be modified.
    :type max_changed_vars: Integer
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
    :param upper_bounds: the maximum values allowed per attribute. It must
        have the same length of x_original. Its values must be different from the
        values informed in lower_bounds. For the categorical columns ordinally
        encoded it represents the category with the minimum value.
    :type upper_bounds: numpy.array
    :param lower_bounds: the minimum values allowed per attribute. It must
        have the same length of x_original. Its values must be different from the
        values informed in upper_bounds. For the categorical columns ordinally
        encoded it represents the category with the maximum value.
    :type lower_bounds: numpy.array
    :param immutable_column_indexes: lists columns that are not allowed to
        be modified.
    :type immutable_column_indexes: numpy.array
    :param integer_columns: lists the columns that allows only integer values.
        It is used by xMOAI in rounding operations.
    :type integer_columns: numpy.array
    :param pop_size: the number of counterfactuals to be generated per algorithm
        and per generation..
    :type pop_size: Integer
    :param prob_mating: the probability of mating by individuals in a given
        generation to be used by the evolutionary algorithms.
    :type prob_mating: Float
    :param seed: the seed to be used by the algorithms.
    :type seed: Integer
    
    :return: the list of algorithms to be used in the optimization as well as
        the upper and lower bounds to be used.
    :rtype: np.array, np.array, np.array
    """
    # fixing cases where the upper and lower bounds are the same value
    # in these cases, these columns are also added to the immutable 
    # column list
    for index in np.where(upper_bounds == lower_bounds)[0]:
        immutable_column_indexes.append(index)
        
        if np.issubdtype(upper_bounds[index], np.integer):
            upper_bounds[index] += 1
        else:
            upper_bounds[index] += 1e-21
    
    immutable_column_indexes = list(set(immutable_column_indexes))
        
    repair = xMOAIRepair(X_current, max_changed_vars, categorical_columns_label_encoder, \
                         categorical_columns_one_hot_encoder, integer_columns, \
                         immutable_column_indexes)
    
    ref_dirs = get_reference_directions("energy", 3, 250, seed=1)
    #ref_dirs = get_reference_directions(
    #    "multi-layer",
    #    get_reference_directions("das-dennis", num_objectives, \
    #                             n_partitions=num_objectives * 4, scaling=1.0),
    #    get_reference_directions("das-dennis", num_objectives, \
    #                             n_partitions=num_objectives * 4, scaling=0.5)
    #)
        
    ref_points = np.zeros((1, num_objectives))
    
    algorithms = {
        #"NSGA-II": NSGA2(pop_size=pop_size, repair=repair),
        #"NSGA-III": NSGA3(pop_size=pop_size*3, ref_dirs=ref_dirs, \
        #                  repair=repair),
        "UNSGA-III": UNSGA3(pop_size=pop_size*3, ref_dirs=ref_dirs, \
                            repair=repair),
        #"RNSGA-II": RNSGA2(pop_size=pop_size*3, \
        #                   ref_points=ref_points, repair=repair),
        #"RNSGA-III": RNSGA3(pop_per_ref_point=pop_size*3, \
        #                   ref_points=ref_points, repair=repair)
    }
    
    return algorithms, upper_bounds, lower_bounds

def generate_counterfactuals_classification_proba(model, X_train, X_current, y_desired, \
                             immutable_column_indexes, y_acceptable_range, \
                             upper_bounds, lower_bounds, \
                             categorical_columns_label_encoder, \
                             categorical_columns_one_hot_encoder, \
                             integer_columns, pop_size=None, max_changed_vars=None, \
                             n_gen=100, seed=None, prob_mating=0.7, \
                             verbose=False, method_name='predict_proba', \
                             select_best=False, n_jobs=1):
    """Generate counterfactuals for a classification problem where the trained
    machine learning model returns only the probabilities predicted for each
    class.

    :param model: the machine learning trained model to be used to
        evaluate the counterfactuals.
    :type model: object
    :param X_train: the train DataFrame used to calculate the data ranges
    :type X_train: pandas.DataFrame
    :param X_current: the original individual
    :type X_current: pandas.Series
    :param y_desired: the desired value to be predicted
    :type y_desired: Object
    :param immutable_column_index: lists columns that are not allowed to
        be modified.
    :type immutable_column_index: numpy.array
    :param upper_bounds: the maximum values allowed per attribute. It must
        have the same length of x_original. For the categorical columns ordinally
        encoded it represents the category with the minimum value.
    :type upper_bounds: numpy.array
    :param lower_bounds: the minimum values allowed per attribute. It must
        have the same length of x_original. For the categorical columns ordinally
        encoded it represents the category with the maximum value.
    :type lower_bounds: numpy.array
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
    :param integer_columns: lists the columns that allows only integer values.
        It is used by xMOAI in rounding operations.
    :type integer_columns: numpy.array
    :param pop_size: the number of counterfactuals to be generated per algorithm
        and per generation. Default is None, where 5 * `number of variables` is
        used.
    :type pop_size: Integer
    :param max_changed_vars: the maximum number of attributes to be
        modified. Default is None, where all variables may be modified.
    :type max_changed_vars: Integer
    :param n_gen: the number of generations to be used per algorithm. Default
        is 100.
    :type n_gen: Integer
    :param seed: the seed to be used by the algorithms. Default is None.
    :type seed: Integer
    :param prob_mating: the probability of mating by individuals in a given
        generation to be used by the evolutionary algorithms. Default is 0.7, 
        corresponding to 70%. Set a value between 0.0 and 1.0.
    :type prob_mating: Float
    :param verbose: sets the verbosity. Default is False.
    :type verbose: Boolean
    :param method_name: the method used by the machine learning model to obtain
        its predictions (e.g. `predict`, `predict_proba`).
    :type method_name: str
    :param select_best: whether we should run TOPSIS to prune the best solutions or not
    :type select_best: bool
    :param n_jobs: sets the number of threads to use. Default is -1, where
        all available threads are used.
    :type n_jobs: Integer
    
    :return: the non-dominated counterfactuals found by xMOAI in regard to
        the three objectives (i.e. the best counterfactuls considering the trade-offs
        between the three objectives) in a tuple. The first parameter has all the
        values found considering the three objectives (Pareto front), the second has
        the individual values (Pareto set) and the third has the algorithm responsible
        of generating each counterfactual.
    :rtype: np.array, np.array, np.array
    """
    n_vars = len(X_current)
    
    if seed is None:
        seed = randint(0, 2**32 - 1)
        
    if pop_size is None:
        pop_size = n_vars * 5
    
    if max_changed_vars is None:
        max_changed_vars = n_vars
    
    # algorithm definition
    res = {}
    for i in range(1, max_changed_vars+1):
        algorithms, upper_bounds, lower_bounds = get_algorithms(X_current, \
                       i, categorical_columns_label_encoder, \
                       categorical_columns_one_hot_encoder, upper_bounds, \
                       lower_bounds, immutable_column_indexes, integer_columns, \
                       pop_size, prob_mating, seed)
        
        # problem definition
        problem = ClassificationProblemProbability(X_current, y_desired, upper_bounds, \
                               lower_bounds, i, y_acceptable_range, \
                               categorical_columns_label_encoder, \
                               categorical_columns_one_hot_encoder, \
                               integer_columns, model, \
                               X_train, method_name, parallelization=('threads', n_jobs))
        
        for algorithm in algorithms:
            if verbose:
                print(f'Generating counterfactuals using {algorithm}.')
            
            res[f'{algorithm}_{i}'] = minimize(problem, algorithms[algorithm], \
                                      ('n_gen', n_gen), seed=seed, \
                                      verbose=verbose, X_current=X_current, \
                                      X_train=X_train, y_desired=y_desired)

    return get_nondominated_solutions(X_current, select_best, res)

def generate_counterfactuals_classification_simple(model, X_train, X_current, y_desired, \
                             immutable_column_indexes, \
                             upper_bounds, lower_bounds, \
                             categorical_columns_label_encoder, \
                             categorical_columns_one_hot_encoder, \
                             integer_columns, \
                             pop_size=None, max_changed_vars=None, \
                             n_gen=100, seed=None, prob_mating=0.7, \
                             verbose=False, method_name='predict', \
                             select_best=False, n_jobs=1):
    """Generate counterfactuals for a classification problem where the trained
    machine learning model returns only the predicted class without the
    probabilities.

    :param model: the machine learning trained model to be used to
        evaluate the counterfactuals.
    :type model: object
    :param X_train: the train DataFrame used to calculate the data ranges
    :type X_train: pandas.DataFrame
    :param X_current: the original individual
    :type X_current: pandas.Series
    :param y_desired: the desired value to be predicted
    :type y_desired: Object
    :param immutable_column_index: lists columns that are not allowed to
        be modified.
    :type immutable_column_index: numpy.array
    :param upper_bounds: the maximum values allowed per attribute. It must
        have the same length of x_original. For the categorical columns ordinally
        encoded it represents the category with the minimum value.
    :type upper_bounds: numpy.array
    :param lower_bounds: the minimum values allowed per attribute. It must
        have the same length of x_original. For the categorical columns ordinally
        encoded it represents the category with the maximum value.
    :type lower_bounds: numpy.array
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
    :param integer_columns: lists the columns that allows only integer values.
        It is used by xMOAI in rounding operations.
    :type integer_columns: numpy.array
    :param pop_size: the number of counterfactuals to be generated per algorithm
        and per generation. Default is None, where 5 * `number of variables` is
        used.
    :type pop_size: Integer
    :param max_changed_vars: the maximum number of attributes to be
        modified. Default is None, where all variables may be modified.
    :type max_changed_vars: Integer
    :param n_gen: the number of generations to be used per algorithm. Default
        is 100.
    :type n_gen: Integer
    :param seed: the seed to be used by the algorithms. Default is None.
    :type seed: Integer
    :param prob_mating: the probability of mating by individuals in a given
        generation to be used by the evolutionary algorithms. Default is 0.7, 
        corresponding to 70%. Set a value between 0.0 and 1.0.
    :type prob_mating: Float
    :param verbose: sets the verbosity. Default is False.
    :type verbose: Boolean
    :param method_name: the method used by the machine learning model to obtain
        its predictions (e.g. `predict`, `predict_proba`).
    :type method_name: str
    :param select_best: whether we should run TOPSIS to prune the best solutions or not
    :type select_best: bool
    :param n_jobs: sets the number of threads to use. Default is -1, where
        all available threads are used.
    :type n_jobs: Integer
    
    :return: the non-dominated counterfactuals found by xMOAI in regard to
        the three objectives (i.e. the best counterfactuls considering the trade-offs
        between the three objectives) in a tuple. The first parameter has all the
        values found considering the three objectives (Pareto front), the second has
        the individual values (Pareto set) and the third has the algorithm responsible
        of generating each counterfactual.
    :rtype: np.array, np.array, np.array
    """
    n_vars = len(X_current)
    
    if seed is None:
        seed = randint(0, 2**32 - 1)
        
    if pop_size is None:
        pop_size = n_vars * 5
    
    if max_changed_vars is None:
        max_changed_vars = n_vars
    
    # algorithm definition
    res = {}
    for i in range(1, max_changed_vars+1):
        algorithms, upper_bounds, lower_bounds = get_algorithms(X_current, \
                       i, categorical_columns_label_encoder, \
                       categorical_columns_one_hot_encoder, upper_bounds, \
                       lower_bounds, immutable_column_indexes, integer_columns, \
                       pop_size, prob_mating, seed)
        
        # problem definition
        problem = ClassificationProblemSimple(X_current, y_desired, upper_bounds, \
                               lower_bounds, i, categorical_columns_label_encoder, \
                               categorical_columns_one_hot_encoder, \
                               integer_columns, model, X_train, method_name, \
                               parallelization=('threads', n_jobs))
        
        for algorithm in algorithms:
            if verbose:
                print(f'Generating counterfactuals using {algorithm}.')
                
            res[f'{algorithm}_{i}'] = minimize(problem, algorithms[algorithm], \
                                      ('n_gen', n_gen), seed=seed, \
                                      verbose=verbose, X_current=X_current, \
                                      X_train=X_train, y_desired=y_desired)
            
    return get_nondominated_solutions(X_current, select_best, res)

def generate_counterfactuals_regression(model, X_train, X_current, y_desired,
                             immutable_column_indexes, y_acceptable_range,
                             upper_bounds, lower_bounds,
                             categorical_columns_label_encoder, 
                             categorical_columns_one_hot_encoder, 
                             integer_columns,
                             pop_size=None, max_changed_vars=None,
                             n_gen=100, seed=None, prob_mating=0.7,
                             verbose=False, method_name='predict',
                             select_best=False, n_jobs=1):
    """Generate counterfactuals for a regression problem.

    :param model: the machine learning trained model to be used to
        evaluate the counterfactuals.
    :type model: object
    :param X_train: the train DataFrame used to calculate the data ranges
    :type X_train: pandas.DataFrame
    :param X_current: the original individual
    :type X_current: pandas.Series
    :param y_desired: the desired value to be predicted
    :type y_desired: Object
    :param immutable_column_index: lists columns that are not allowed to
        be modified.
    :type immutable_column_index: numpy.array
    :param y_acceptable_range: the lower (first value) and upper 
        (second value) limits allowed for the output. A counterfactual
        is considered as being "valid" if it has its output within this range. For
        regression problems it is understood as the predicted value where y_desired
        is inside this range. For classification problems it is understood as 
        the probability of being within the expected class shown in y_desired.
    :type y_acceptable_range: np.array
    :param upper_bounds: the maximum values allowed per attribute. It must
        have the same length of x_original. For the categorical columns ordinally
        encoded it represents the category with the minimum value.
    :type upper_bounds: numpy.array
    :param lower_bounds: the minimum values allowed per attribute. It must
        have the same length of x_original. For the categorical columns ordinally
        encoded it represents the category with the maximum value.
    :type lower_bounds: numpy.array
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
    :param integer_columns: lists the columns that allows only integer values.
        It is used by xMOAI in rounding operations.
    :type integer_columns: numpy.array
    :param pop_size: the number of counterfactuals to be generated per algorithm
        and per generation. Default is None, where 5 * `number of variables` is
        used.
    :type pop_size: Integer
    :param max_changed_vars: the maximum number of attributes to be
        modified. Default is None, where all variables may be modified.
    :type max_changed_vars: Integer
    :param n_gen: the number of generations to be used per algorithm. Default
        is 100.
    :type n_gen: Integer
    :param seed: the seed to be used by the algorithms. Default is None.
    :type seed: Integer
    :param prob_mating: the probability of mating by individuals in a given
        generation to be used by the evolutionary algorithms. Default is 0.7, 
        corresponding to 70%. Set a value between 0.0 and 1.0.
    :type prob_mating: Float
    :param verbose: sets the verbosity. Default is False.
    :type verbose: Boolean
    :param method_name: the method used by the machine learning model to obtain
        its predictions (e.g. `predict`, `predict_proba`).
    :type method_name: str
    :param select_best: whether we should run TOPSIS to prune the best solutions or not
    :type select_best: bool
    :param n_jobs: sets the number of threads to use. Default is -1, where
        all available threads are used.
    :type n_jobs: Integer
 
    :return: the non-dominated counterfactuals found by xMOAI in regard to
        the three objectives (i.e. the best counterfactuls considering the trade-offs
        between the three objectives) in a tuple. The first parameter has all the
        values found considering the three objectives (Pareto front), the second has
        the individual values (Pareto set) and the third has the algorithm responsible
        of generating each counterfactual.
    :rtype: np.array, np.array, np.array
    """
    n_vars = len(X_current)
    
    if seed is None:
        seed = randint(0, 2**32 - 1)
        
    if pop_size is None:
        pop_size = n_vars * 5
    
    if max_changed_vars is None:
        max_changed_vars = n_vars
    
    # algorithm definition
    res = {}
    for i in range(1, max_changed_vars+1):
        algorithms, upper_bounds, lower_bounds = get_algorithms(X_current, \
                       i, categorical_columns_label_encoder, \
                       categorical_columns_one_hot_encoder, upper_bounds, \
                       lower_bounds, immutable_column_indexes, integer_columns, \
                       pop_size, prob_mating, seed)
        
        # problem definition
        problem = RegressionProblem(X_current, y_desired, upper_bounds, \
                               lower_bounds, i, y_acceptable_range, \
                               categorical_columns_label_encoder, \
                               categorical_columns_one_hot_encoder, integer_columns, model, \
                               X_train, method_name, parallelization=('threads', n_jobs))
        
        for algorithm in algorithms:
            if verbose:
                print(f'Generating counterfactuals using {algorithm}.')
            
            res[f'{algorithm}_{i}'] = minimize(problem, algorithms[algorithm], \
                                      ('n_gen', n_gen), seed=seed, \
                                      verbose=verbose, X_current=X_current, \
                                      X_train=X_train, y_desired=y_desired)
    
    return get_nondominated_solutions(X_current, select_best, res)
