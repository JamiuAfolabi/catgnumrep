import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class RedSk2p(BaseEstimator,TransformerMixin):
    """
    Reduces Skewness of the dataset by 
    transforming by np.sqrt
    PARAMETERS
    ----------
    skew_threshold: Threshold for skewness
    drop_column : target column
    bias : sqrt bias
    Returns
    -------
    df : Returns Dataframe
    """
    def __init__(self,drop_column=None,skew_threshold=0.5,bias=0.5):
        self.skew_threshold=skew_threshold
        self.drop_column=drop_column
        self.bias=bias
    def fit(self,X,y=None):
        self.skew=X.skew()
        return self
    def transform(self,X):
        df=X.copy()
        if self.drop_column==None:
            df=np.sqrt(df)
            return df
        for column in self.skew.keys():
            if isinstance(self.drop_column,str):
                bool_ = column != self.drop_column
            elif isinstance(self.drop_column,list):
                bool_= column not in self.drop_column
            else:
                pass
            if (self.skew[column] > self.skew_threshold and bool_):
                df[column]+=self.bias
                df[column]=np.sqrt(df[column])
                print(':::::::::::::::::::: Transformation on {} with Skewness {}'.format(str(column),str(self.skew[column])))
        return df