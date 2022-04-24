import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.cluster import KMeans

class Fill_Null(BaseEstimator,TransformerMixin):
    """
    -Dataframe consists of categorical and Numeric Features
    
    Fill missing Value based on the following criteria:
        -If missing value is greater than threshold, fill by fill_value
        else use mode if the mode is greater than threshold_mode
                      else use the mean
    """
    def __init__(self,fill_value=-999,threshold_missing=0.5,threshold_mode=0.5,groupby_column=None):
        """
        Initialize fill values and threshold
        
        PARAMETERS
        ----------
        fill_value: Integer
        threshold_missing: Between (0,1)
        threshold_mode: Between (0,1)
        groupby_column: string or List
        """
        self.fill_value=fill_value
        self.threshold_missing=threshold_missing
        self.threshold_mode=threshold_mode
        self.groupby_column=groupby_column
    def fit(self,X,y=None):
        df=X.copy()
        self.numeric_feature=[pd.api.types.is_numeric_dtype(df[column]) for column in df.columns]
        self.zip_file=zip(df.columns,list(df.isnull().sum()/len(df) > self.threshold_missing),self.numeric_feature)
        return self
    def transform(self,X):
        df=X.copy()
        for feature,bool_,bool_f in self.zip_file:
            if isinstance(self.groupby_column,(list,str)):
                if bool_f:
                    if not bool_ :
                        if df[feature].value_counts(normalize=True).values[0] > self.threshold_mode:
                            df[feature].fillna(df.groupby(self.groupby_column)[feature].transform(lambda grp:grp.mode()[0]),inplace=True)
                        else:
                            df[feature].fillna(df.groupby(self.groupby_column)[feature].transform(lambda grp:grp.mean()),inplace=True)
                    else:
                        df[feature].fillna(self.fill_value,inplace=True)
                else:
                    df[feature].fillna(df.groupby(self.groupby_column)[feature].transform(lambda grp:grp.mode()[0]),inplace=True)
                
                
                
            elif self.groupby_column is None:
                if bool_f:
                    if not bool_ :
                        if df[feature].value_counts(normalize=True).values[0] > self.threshold_mode:
                            df[feature].fillna(df[feature].mode()[0],inplace=True)
                        else:
                            df[feature].fillna(df[feature].mean(),inplace=True)
                    else:
                        df[feature].fillna(self.fill_value,inplace=True)
                else:
                    df[feature].fillna(df[feature].mode()[0],inplace=True)
            else:
                pass
        return df
    def get_feature_names(self,X):
        """
        PARAMETER
        ---------
        X : DataFrame object
        
        Returns
        -------
        numeric features,Categorical_Features
        """
        df=X.copy()
        numeric_feature=[column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
        categoric_feature=[column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
        return numeric_feature,categoric_feature
    



