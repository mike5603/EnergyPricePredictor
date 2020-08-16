import math
import numpy as np
from sklearn import base

class Composite(base.BaseEstimator, base.RegressorMixin):
  def __init__(self, linear,nonlinear):
    self.linear = linear
    self.nonlinear = nonlinear
    self.fourier = FourierTransformer()

  def fit(self, X, y):
    X2 = self.fourier.fit_transform(X)
    self.linear.fit(X2,y)
    res = np.asarray(y) - np.asarray(self.linear.predict(X2))
    self.nonlinear.fit(X,res)
    return self

  def predict(self, X):
    # make predictions 
    X2 = self.fourier.transform(X)
    pred = np.asarray(self.linear.predict(X2))+np.asarray(self.nonlinear.predict(X))
    return list(pred)

  def score(self, X, y):
    pred = self.predict(X)
    ave_y = np.mean(y)
    y = np.asarray(y)
    r2 = 1-np.sum((y-pred)**2)/np.sum((y-ave_y)**2)
    # custom score implementation
    return r2

class FourierTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        pass
    # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        X['Julian_Day'] = X.index.to_julian_date()
        X['sin1'] = np.sin(2*math.pi*X['Julian_Day'].astype(int)/365.25)
        X['cos1'] = np.cos(2*math.pi*X['Julian_Day'].astype(int)/365.25)
        X['sin2'] = np.sin(2*math.pi*X['Julian_Day'].astype(int)/7)
        X['cos2'] = np.cos(2*math.pi*X['Julian_Day'].astype(int)/7)
        X['sin3'] = np.sin(2*math.pi*X['Julian_Day'].astype(int)/182.625)
        X['cos3'] = np.cos(2*math.pi*X['Julian_Day'].astype(int)/182.625)
        X['sin4'] = np.sin(2*math.pi*X['Julian_Day'].astype(int)/3.5)
        X['cos4'] = np.cos(2*math.pi*X['Julian_Day'].astype(int)/3.5)
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return X[['sin1','cos1','sin2','cos2','sin3','cos3','sin4','cos4']].astype(float)
