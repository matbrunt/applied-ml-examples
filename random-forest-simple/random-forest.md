

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# modules for scaling, transforming, and wrangling data
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

# cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

# persist model
from sklearn.externals import joblib
```


```python
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';') # source data is ';' seperated
```


```python
# 1599 samples, 12 features
print(data.shape)
```

    (1599, 12)



```python
print(data.describe())
```

           fixed acidity  volatile acidity  citric acid  residual sugar  \
    count    1599.000000       1599.000000  1599.000000     1599.000000   
    mean        8.319637          0.527821     0.270976        2.538806   
    std         1.741096          0.179060     0.194801        1.409928   
    min         4.600000          0.120000     0.000000        0.900000   
    25%         7.100000          0.390000     0.090000        1.900000   
    50%         7.900000          0.520000     0.260000        2.200000   
    75%         9.200000          0.640000     0.420000        2.600000   
    max        15.900000          1.580000     1.000000       15.500000   
    
             chlorides  free sulfur dioxide  total sulfur dioxide      density  \
    count  1599.000000          1599.000000           1599.000000  1599.000000   
    mean      0.087467            15.874922             46.467792     0.996747   
    std       0.047065            10.460157             32.895324     0.001887   
    min       0.012000             1.000000              6.000000     0.990070   
    25%       0.070000             7.000000             22.000000     0.995600   
    50%       0.079000            14.000000             38.000000     0.996750   
    75%       0.090000            21.000000             62.000000     0.997835   
    max       0.611000            72.000000            289.000000     1.003690   
    
                    pH    sulphates      alcohol      quality  
    count  1599.000000  1599.000000  1599.000000  1599.000000  
    mean      3.311113     0.658149    10.422983     5.636023  
    std       0.154386     0.169507     1.065668     0.807569  
    min       2.740000     0.330000     8.400000     3.000000  
    25%       3.210000     0.550000     9.500000     5.000000  
    50%       3.310000     0.620000    10.200000     6.000000  
    75%       3.400000     0.730000    11.100000     6.000000  
    max       4.010000     2.000000    14.900000     8.000000  


Target variable (descriptor) is *quality*, which ranges from 3 to 8, but is mostly clustered around 4.8 to 6.4.

Split data into train and test sets, taking 20% of the data as a test set.

We also stratify the sample by our target variable, which ensures that the training set looks similar to the test set, so the evaluation metrics should be more reliable.


```python
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
```

We need to standardise our features because they are on different scales. This involves centering the feature values around zero with roughly the same variance - we do this for each feature by subtracting its mean, then dividing by its standard deviation.


```python
X_train_scaled = preprocessing.scale(X_train)

# Confirm scaled dataset is centered at zero, with unit variance
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))
```

    [  1.16664562e-16  -3.05550043e-17  -8.47206937e-17  -2.22218213e-17
       2.22218213e-17  -6.38877362e-17  -4.16659149e-18  -2.54439854e-15
      -8.70817622e-16  -4.08325966e-16  -1.17220107e-15]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]


We don't actually do this when building the model because it introduces bias, when running the model against new data we won't have a representation of the data variance.

So instead we use a pipeline with a *transformer*; this allows to calculate the means and std dev from the training set, then apply those same values to the test set. The transformer allows us to 'fit' a preprocessing step using training data the same way we would fit a model.

This makes the model performance estimates more realistic, and allows us to use the preprocessing steps in **cross-validation pipeline**.

We generate a scaler object with the saved means and standard deviations for each feature, then we can apply (transform) the exact same means and standard deviations using the scalar to the test data.


```python
# Save the scaled means and standard deviations for each feature into a scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))
```

    [  1.16664562e-16  -3.05550043e-17  -8.47206937e-17  -2.22218213e-17
       2.22218213e-17  -6.38877362e-17  -4.16659149e-18  -2.54439854e-15
      -8.70817622e-16  -4.08325966e-16  -1.17220107e-15]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]


In practice, we don't need to manually fit the scaler, we can just pass the scaler class to use in the pipeline definition and it will automatically fit against the training data.

Here we set up a **modeling pipeline** that first transforms the data with a StandardScalar(), then fits the model using a Random Forest regressor.


```python
pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
```

*model parameters* are learned directly from the data, i.e. they are regression co-efficients
*hyperparameters* cannot be learned directly from the data, they express more high-level structural information about the model

For example, Random Forest can use either MSE or MAE performance measures, but doesn't know which one to use. It also doesn't know how many trees it should grow against the data. These are both examples of hyperparameters that we must set, to give the model structural information on how to build itself.


```python
# list the hyperparameters we can tune
print(pipeline.get_params())
```

    {'standardscaler__with_std': True, 'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True), 'randomforestregressor__min_weight_fraction_leaf': 0.0, 'standardscaler__copy': True, 'randomforestregressor__min_impurity_decrease': 0.0, 'randomforestregressor__random_state': None, 'randomforestregressor__verbose': 0, 'randomforestregressor__oob_score': False, 'randomforestregressor__n_estimators': 100, 'randomforestregressor__n_jobs': 1, 'randomforestregressor__min_impurity_split': None, 'randomforestregressor': RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False), 'standardscaler__with_mean': True, 'randomforestregressor__min_samples_leaf': 1, 'steps': [('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))], 'memory': None, 'randomforestregressor__min_samples_split': 2, 'randomforestregressor__bootstrap': True, 'randomforestregressor__max_depth': None, 'randomforestregressor__max_leaf_nodes': None, 'randomforestregressor__max_features': 'auto', 'randomforestregressor__warm_start': False, 'randomforestregressor__criterion': 'mse'}


The hyperparameters in a pipeline are prefixed with their class instance, which you need to remember when setting them on the pipeline. For example, the RandomForestRegressor hyperparameters all begin with 'randomforestregressor__'.


```python
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
```

Now, to reduce overfitting we want to take slices of our available data, and use K-fold cross validation to train the model on these different slices, and optimise the hyperparameters.

GridSearchCV performs cross-validation across the entire grid of hyperparameters, e.g. all the possible permutations.


```python
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# fit and tune the model
clf.fit(X_train, y_train)
```




    GridSearchCV(cv=10, error_score='raise',
           estimator=Pipeline(memory=None,
         steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decr...mators=100, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False))]),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'randomforestregressor__max_depth': [None, 5, 3, 1], 'randomforestregressor__max_features': ['auto', 'sqrt', 'log2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)




```python
# look at the best set of parameters discovered using CV
print(clf.best_params_)
```

    {'randomforestregressor__max_depth': None, 'randomforestregressor__max_features': 'sqrt'}


GridSearchCV will automatically refit the model with the best set of hyperparameters against the entire training set, this can be confirmed by checking that `clv.refit` reports `True`.

Now we have our model (`clf`), which we can apply against other sets of data, such as the test set to evaluate the model performance.


```python
def evaluate_performance(model, test_data, actual):
    # make predictions against the test set
    y_pred = model.predict(test_data)

    # print the evaluation metrics
    print('r2: ', r2_score(actual, y_pred))
    print('-'*40)
    print('MSE: ', mean_squared_error(actual, y_pred))
    
evaluate_performance(clf, X_test, y_test)
```

    r2:  0.470718566499
    ----------------------------------------
    MSE:  0.34153125


Now we have a trained model, we can persist it so we don't have to go through all these steps every time we see new data.


```python
# save model to a .pkl file
joblib.dump(clf, 'rf_regressor.pkl')

# load model from a .pkl file
clf2 = joblib.load('rf_regressor.pkl')

evaluate_performance(clf2, X_test, y_test)
```

    r2:  0.470718566499
    ----------------------------------------
    MSE:  0.34153125



```python

```
