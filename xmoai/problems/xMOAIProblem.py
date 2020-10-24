import numpy as np
from pymoo.model.problem import Problem
from xmoai.problems.objectives import *
from xmoai.problems.restrictions import *

#https://pymoo.org/problems/index.html
#https://pymoo.org/problems/custom.html
#https://pymoo.org/misc/constraint_handling.html

class xMOAIProblem(Problem):
    """Defines the multiobjective problem to be solved by xMOAI. The 
    problem may be a regression or a classification problem. This class
    must not be called directly. Instead, please use the methods provided
    in the "configure.py" file.
    """
    
    def __init__(self, X_current, y_desired, upper_bounds, lower_bounds, \
                 max_changed_vars, y_acceptable_range, categorical_columns, \
                 integer_columns, trained_model, method_name, parallelization):
        """Class constructor.
        
        :param x_original: the original individual
        :type x_original: numpy.array
        :param y_desired: the desired value to be predicted
        :type y_desired: Object
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
        :param y_acceptable_range: the lower (first value) and upper 
            (second value) limits allowed for the output. A counterfactual
            is considered as being "valid" if it has its output within this range. For
            regression problems it is understood as the predicted value where y_desired
            is inside this range. For classification problems it is understood as 
            the probability of being within the expected class shown in y_desired.
        :param categorical_columns: dictionary containing the categorical columns
            and their allowed values. The keys are the i-th position of the indexes
            and the values are the allowed categories. The minimum and maximum categories
            must respect the values in lower_bounds and upper_bounds since this variable
            is called after it in code.
        :type categorical_columns: dict
        :param integer_columns: lists the columns that allows only integer values.
            It is used by xMOAI in rounding operations.
        :type integer_columns: numpy.array
        :param trained_model: the machine learning trained model to be used to
            evaluate the counterfactuals.
        :type trained_model: object
        :param method_name: the method used by the machine learning model to obtain
            its predictions (e.g. `predict`, `predict_proba`).
        :type method_name: str
        :param parallelization: parallelization options used by pymoo.
        :type parallelization: Object
        """
        self.X_current = X_current.flatten()
        n_var = self.X_current.shape[0]
        
        self.model = trained_model
        
        self.y_desired = y_desired
        self.y_acceptable_range = y_acceptable_range
        
        self.max_changed_vars = max_changed_vars
        self.categorical_columns = categorical_columns
        self.categorical_indexes = np.array(list(categorical_columns.keys()))
        
        self.integer_columns = integer_columns
        
        if self.categorical_indexes.shape[0] > 0:
            self.numerical_indexes = np.where(np.array(range(n_var)) != self.categorical_indexes)
        else:
            self.numerical_indexes = np.array(range(n_var))
        
        self.method_name = method_name
        
        super().__init__(n_var=n_var, n_obj=num_objectives, n_constr=3, \
                         xl=lower_bounds, xu=upper_bounds, \
                         parallelization=parallelization)

class RegressionProblem(xMOAIProblem):
    """Defines a multiobjective problem to be solved by xMOAI for regression
    problems. This class must not be called directly. Instead, please use 
    the methods provided in the "configure.py" file.
    """
    
    def __init__(self, X_current, y_desired, upper_bounds, lower_bounds, \
                 max_changed_vars, y_acceptable_range, categorical_columns, \
                 integer_columns, trained_model, method_name, parallelization):
        """Class constructor.
        
        
        :param x_original: the original individual
        :type x_original: numpy.array
        :param y_desired: the desired value to be predicted
        :type y_desired: Object
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
        :param y_acceptable_range: the lower (first value) and upper 
            (second value) limits allowed for the output. A counterfactual
            is considered as being "valid" if it has its output within this range. For
            regression problems it is understood as the predicted value where y_desired
            is inside this range. For classification problems it is understood as 
            the probability of being within the expected class shown in y_desired.
        :param categorical_columns: dictionary containing the categorical columns
            and their allowed values. The keys are the i-th position of the indexes
            and the values are the allowed categories. The minimum and maximum categories
            must respect the values in lower_bounds and upper_bounds since this variable
            is called after it in code.
        :type categorical_columns: dict
        :param integer_columns: lists the columns that allows only integer values.
            It is used by xMOAI in rounding operations.
        :type integer_columns: numpy.array
        :param trained_model: the machine learning trained model to be used to
            evaluate the counterfactuals.
        :type trained_model: object
        :param method_name: the method used by the machine learning model to obtain
            its predictions (e.g. `predict`, `predict_proba`).
        :type method_name: str
        :param parallelization: parallelization options used by pymoo.
        :type parallelization: Object
        """
        super().__init__(X_current, y_desired, upper_bounds, lower_bounds, \
                         max_changed_vars, y_acceptable_range, \
                         categorical_columns, integer_columns, \
                         trained_model, method_name, parallelization)
            
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates an individual.
    
        :param x: the individual (or individuals) to be evaluated
        :type x: numpy.array
        :param out: the evaluation output.
        :type out: dict
        """
        f1, prediction = get_difference_target_regression(self.model, x, \
                                              self.y_desired, self.method_name)
        f2 = get_difference_attributes(x, self.X_current, self.categorical_columns)
        f3 = get_modified_attributes(x, self.X_current)
        
        g1 = get_changed_vars_threshold(f3, self.max_changed_vars)
        g2, g3 = is_prediction_in_threshold_regression(self.y_acceptable_range, \
                                                       prediction)
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2, g3])
        
    
class ClassificationProblemProbability(xMOAIProblem):
    """Defines a multiobjective problem to be solved by xMOAI for 
    classification problems where the trained model exposes the probability
    of the classes. This class must not be called directly. Instead, please use 
    the methods provided in the "configure.py" file.
    """
    
    def __init__(self, X_current, class_column, upper_bounds, lower_bounds, \
                 max_changed_vars, y_acceptable_range, categorical_columns, \
                 integer_columns, trained_model, method_name, parallelization):
        """Class constructor.
        
        
        :param x_original: the original individual
        :type x_original: numpy.array
        :param y_desired: the desired value to be predicted
        :type y_desired: Object
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
        :param y_acceptable_range: the lower (first value) and upper 
            (second value) limits allowed for the output. A counterfactual
            is considered as being "valid" if it has its output within this range. For
            regression problems it is understood as the predicted value where y_desired
            is inside this range. For classification problems it is understood as 
            the probability of being within the expected class shown in y_desired.
        :param categorical_columns: dictionary containing the categorical columns
            and their allowed values. The keys are the i-th position of the indexes
            and the values are the allowed categories. The minimum and maximum categories
            must respect the values in lower_bounds and upper_bounds since this variable
            is called after it in code.
        :type categorical_columns: dict
        :param integer_columns: lists the columns that allows only integer values.
            It is used by xMOAI in rounding operations.
        :type integer_columns: numpy.array
        :param trained_model: the machine learning trained model to be used to
            evaluate the counterfactuals.
        :type trained_model: object
        :param method_name: the method used by the machine learning model to obtain
            its predictions (e.g. `predict`, `predict_proba`).
        :type method_name: str
        :param parallelization: parallelization options used by pymoo.
        :type parallelization: Object
        """
        super().__init__(X_current, class_column, upper_bounds, lower_bounds, \
                         max_changed_vars, y_acceptable_range, \
                         categorical_columns, integer_columns, \
                         trained_model, method_name, parallelization)
            
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates an individual.
    
        :param x: the individual (or individuals) to be evaluated
        :type x: numpy.array
        :param out: the evaluation output.
        :type out: dict
        """
        f1, prediction = get_difference_target_classification_proba(self.model, x, \
                                                    self.y_desired, self.method_name)
        f2 = get_difference_attributes(x, self.X_current, self.categorical_columns)
        f3 = get_modified_attributes(x, self.X_current)
        
        g1 = get_changed_vars_threshold(f3, self.max_changed_vars)
        g2, g3 = is_prediction_in_threshold_classification_proba(self.y_acceptable_range, \
                                                                 prediction, self.y_desired)
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2, g3])
        
    
class ClassificationProblemSimple(xMOAIProblem):
    """Defines a multiobjective problem to be solved by xMOAI for 
    classification problems where the trained model does not expose the probability
    of the classes. This class must not be called directly. Instead, please use 
    the methods provided in the "configure.py" file.
    """
    
    def __init__(self, X_current, class_column, upper_bounds, lower_bounds, \
                 max_changed_vars, categorical_columns, \
                 integer_columns, trained_model, method_name, parallelization):
        """Class constructor.
    
        :param x_original: the original individual
        :type x_original: numpy.array
        :param y_desired: the desired value to be predicted
        :type y_desired: Object
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
        :param y_acceptable_range: the lower (first value) and upper 
            (second value) limits allowed for the output. A counterfactual
            is considered as being "valid" if it has its output within this range. For
            regression problems it is understood as the predicted value where y_desired
            is inside this range. For classification problems it is understood as 
            the probability of being within the expected class shown in y_desired.
        :type y_acceptable_range: np.array
        :param categorical_columns: dictionary containing the categorical columns
            and their allowed values. The keys are the i-th position of the indexes
            and the values are the allowed categories. The minimum and maximum categories
            must respect the values in lower_bounds and upper_bounds since this variable
            is called after it in code.
        :type categorical_columns: dict
        :param integer_columns: lists the columns that allows only integer values.
            It is used by xMOAI in rounding operations.
        :type integer_columns: numpy.array
        :param trained_model: the machine learning trained model to be used to
            evaluate the counterfactuals.
        :type trained_model: object
        :param method_name: the method used by the machine learning model to obtain
            its predictions (e.g. `predict`, `predict_proba`).
        :type method_name: str
        :param parallelization: parallelization options used by pymoo.
        :type parallelization: Object
        """
        super().__init__(X_current, class_column, upper_bounds, lower_bounds, \
                         max_changed_vars, None, categorical_columns, \
                         integer_columns, trained_model, method_name, parallelization)
            
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates an individual.
    
        :param x: the individual (or individuals) to be evaluated
        :type x: numpy.array
        :param out: the evaluation output.
        :type out: dict
        """
        f1, prediction = get_difference_target_classification_simple(self.model, x, \
                                                    self.y_desired, self.method_name)
        f2 = get_difference_attributes(x, self.X_current, self.categorical_columns)
        f3 = get_modified_attributes(x, self.X_current)
        
        g1 = get_changed_vars_threshold(f3, self.max_changed_vars)
        g2 = is_prediction_in_threshold_classification_simple(prediction, self.y_desired)
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])