# xMOAI 🗿: Multiobjective Optimization in Explainable Artificial Intelligence

xMOAI is an open-source package implementing Explainable Artificial Intelligence (XAI) using Multiobjective Optimization (MOO). It is capable of generating
a large number of counterfactuals in datasets with several attributes - most of them immutable or very constrained. It supports both regression or classification
problems. For classification problems, it does support both problems with trained machine learning models exposing the predicted class probabilities or only
the predicted class. It was tested throughly with trained models in scikit-learn, XGBoost, LightGBM and Tensorflow. In practice, it works with any model that exposes
an output similar to scikit-learn or Tensorflow `predict` methods.

## Usage

```python
import numpy as np

from xmoai.setup.configure import generate_counterfactuals_classification_proba

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# seed
random_state = 0

# getting a dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=random_state)

# training a machine learning model
clf = RandomForestClassifier(max_depth=2, random_state=random_state)
clf.fit(X_train, y_train)

# getting an individual (X_original), its original prediction (y_original) and
# the desired output (y_desired)
index = 0
X_original = X_test[0,:].reshape(1, -1)
y_original = clf.predict(X_original)
y_original_proba = clf.predict_proba(X_original)
y_desired = 1

print(f'The original prediction was {y_original} with probabilities {y_original_proba}')
print(f'We will attempt to generate counterfactuals where the outcome is {y_desired}.')

# generating counterfactuals
immutable_column_indexes = [2] # let's say we can't change the last column
categorical_columns = {} # there are no categorical columns
integer_columns = [] # there are no columns that only accept integer values
y_acceptable_range = [0.5, 1.0] # we will only accept counterfactuals with the predicted prob. in this range

upper_bounds = np.array(X_train.max(axis=0)*0.8) # this is the maximum allowed number per column
lower_bounds = np.array(X_train.min(axis=0)*0.8) # this is the minimum allowed number per column.
# you may change the bounds depending on the needs specific to the individual being trained.

# running the counterfactual generation algorithm
front, X_generated, algorithms = generate_counterfactuals_classification_proba(clf,
                          X_original, y_desired, immutable_column_indexes,
                          y_acceptable_range, upper_bounds, lower_bounds,
                          categorical_columns, integer_columns, n_gen=20,
                          pop_size=30, max_changed_vars=3, verbose=False, 
                          seed=random_state)
```
## Features

The documentation as well as the code are part of an ongoing research. Currently, it does support:

* Regression problems
* Classification problems (probability or single class as outputs)

On the variables, it does support:

* Decimal and integer variables as values (such as counts, quantities, etc.)
* Ordinally encoded categorical variables (categories encoded as integers)
* Setting the upper and lower bounds per variable
* Setting which columns are immutable
* Setting which categories are bound to be modified (xMOAI is able to understand only the categories 1, 5, 7 and 15 are allowed categories instead of treating it as a numerical range)
* Setting the target desired (for regression problems, you can inform the value you want to have as an output; for classification problems, the desired class)
* Setting the "allowed output range" (for regression problems, you can inform what values are acceptable as outputs instead of a single value. As an example, for a housing prices dataset you may want to find a counterfactual with an output of $100000.00. However, anything between $99000.00 and $105000.00 could also be good prices for your problem. For a classification problem, it is the percentage of certainity of the predicted class considering your problem).

It does not support at the present moment:

* One-hot encoded categories
* Models available in hosted servers (i.e. with a REST API endpoint)
* Multiple allowed intervals for a single attribute (e.g. for a single column, instead of a range of -10 to +20, two ranges of -10 to 0 and +10 to +20).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
