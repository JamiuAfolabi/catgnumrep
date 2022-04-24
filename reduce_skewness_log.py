
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np

class RedSk1p(BaseEstimator,TransformerMixin):
    """
    Reduces Skewness of the dataset by 
    transforming by np.log1p
    PARAMETERS
    ----------
    skew_threshold: Threshold for skewness
    drop_column : target column
    Returns
    -------
    df : Returns Dataframe
    """
    def __init__(self,drop_column=None,skew_threshold=0.5):
        self.skew_threshold=skew_threshold
        self.drop_column=drop_column
    def fit(self,X,y=None):
        self.skew=X.skew()
        return self
    def transform(self,X):
        df=X.copy()
        if self.drop_column==None:
            df=np.log1p(df)
            return df
        for column in self.skew.keys():
            if isinstance(self.drop_column,str):
                bool_ = column != self.drop_column
            elif isinstance(self.drop_column,list):
                bool_= column not in self.drop_column
            else:
                pass
            if (self.skew[column] > self.skew_threshold and bool_):
                df[column]=np.log1p(df[column])
                
                print(':::::::::::::::::::: Transformation on {} with Skewness {}'.format(str(column),str(self.skew[column])))
        return df