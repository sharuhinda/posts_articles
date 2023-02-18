import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin # base classes to create custom transformers
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.exceptions import NotFittedError


#============================================================================================

class DFDropColumns(BaseEstimator, TransformerMixin):
    """
    Class to drop columns that are redundant for the model
    """

    def __init__(self, columns=None) -> None: # cols is a list containig column names to drop
        #super().__init__()
        self.columns = columns


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if (self.columns is not None) and (len(self.columns) > 0):
            return X.drop(columns=[col for col in self.columns if col in X.columns])
        return X

#============================================================================================

class DFWoeEncoder(BaseEstimator, TransformerMixin): # columns=None, encode_nans=True, nan_equiv=np.nan
    """
    [TODO] Unknown behavior with np.NaN values
    Implementation of Weight-of-Evidence (WoE) encoder
    WoE is nuanced view towards the relationship between a categorical independent variable and a dependent variable
    The mathematical definition of Weight of Evidence is the natural log of the odds ratio:
    WoE = ln ( %_category_positive_of_all_positives / %_category_negative_of_all_negatives )
    Need to be fitted first, applicable only for binary classification problems
    columns - list of categorical columns to perform encoding (no additional checks)
    encode_nans = True | False - if it's required to encode nan-like values
    nan_equiv = np.NaN - nan-like value (same value applying to all columns)
    X - pandas DataFrame containing 'columns'
    y - target (1 / 0 encoded)
    """

    def __init__(self, columns=None, encode_nans=True, nan_equiv=np.nan) -> None:
        #super().__init__()
        self.columns = columns
        self.encode_nans = encode_nans
        self.nan_equiv = nan_equiv
        self.encoders_ = {}
        self._fitted = False
        #self._logger = logging.getLogger(name='ClassDFWoeEncoder')
        #self._logger.debug('Initialization complete')
        return


    def fit(self, X, y=None):
        #self._logger.debug('`fit` method got control')
        if (y is None) or (self.columns is None):
            raise ValueError('No target values passed')
        self.encoders_ = {}
        #self._logger.debug('\t Internal encoders storage created')
        for col in self.columns:
            if col in X.columns:
                # Here we can output warnings if necessary conditions for WOE not met:
                # 1. each category size >= 5% of sample size
                # 2. each category contains both positive and negative target values
                #self._logger.debug('\t Calling _get_woe to create encoder')
                self.encoders_[col] = self._get_woe(X[col], y, self.encode_nans, self.nan_equiv)
                #self._logger.debug('\t Created encoder for `{}`'.format(col))
        if len(self.encoders_) > 0:
            self._fitted = True
        #self._logger.debug('\t Fit phase complete')
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()
        if not self._fitted:
            raise(NotFittedError)
        for col, map_values in self.encoders_.items():
            X_transformed[col] = X_transformed[col].map(map_values)
        for col in self.columns:
            if col in X_transformed.columns:
                X_transformed[col] = pd.to_numeric(X_transformed[col])
        return X_transformed


    # faster (almost 40% faster) version based on numpy arrays
    def _get_woe(self, X, y, encode_nans, nan_equiv):
        # a = a[a[:, 0].argsort()] # sort by 1st column
        # np.split(a[:,1], np.unique(a[:, 0], return_index=True)[1][1:]) # return grouped values
        #self._logger.debug('\t\t WoE calculator got control')
        num_events_total = y.sum()
        num_nonevents_total = len(y) - num_events_total
        
        # [TODO] Capture np.nan values correctly
        is_numeric = True
        try:
            np.float64(nan_equiv)
        except ValueError:
            is_numeric = False
        if is_numeric and np.isnan(nan_equiv):
            m = X.isna()
        else:
            m = (X == nan_equiv)
        X_vals = X[~m]
        y_vals = y[~m]
        
        X_nans = X[m]
        is_nans_exist = len(X_nans) > 0
        if is_nans_exist:
            y_nans = y[m].to_numpy()
        #self._logger.debug('\t\t Data splitted to `vals` and `nans`')

        grouper = pd.DataFrame({'feature': X, 'target': y}).groupby('feature', as_index=True)

        num_total = grouper['target'].count().to_numpy()
        num_events = grouper['target'].sum().to_numpy()
        num_nonevents = num_total - num_events
        
        #self._logger.debug('\t\t Total statistics calculated')

        cats = list(grouper['target'].sum().index)

        if encode_nans and is_nans_exist:
            num_events = np.append(num_events, y_nans.sum())
            num_nonevents = np.append(num_nonevents, len(y_nans) - y_nans.sum())
            cats.append(nan_equiv)
            #self._logger.debug('\t\t `encode_nans`==True -> added statistics for nans')
        
        events_share = num_events / num_events_total
        nonevents_share = num_nonevents / num_nonevents_total
        
        """
        As we deal with integer values (number of events) we will add small value (0.01) to nominator and denominator
        to exclude division by 0 error and 'inf' value as well. This will not bias results significantly
        because added value is only 1% from minimal value (1.0)
        """
        woe = np.log((events_share + 0.001) / (nonevents_share + 0.001)) # returns np.array
        #iv = (events_share - nonevents_share) * woe
        #self._logger.debug('\t\t Calculations finished. Returning...')

        return {cat: woe for cat, woe in zip(cats, woe)}


    def get_encodings(self):
        pass

#============================================================================================

class DFColumnBinning(BaseEstimator, TransformerMixin): # bins_dict shoud be in form: {'column': bins_param_for_cut_func}, new_names=None
    """
    Used to binning purposes in pipelines
    Wrapper class for pandas 'cut' function
    bins_dict shoud be in form: {'column': bins_param_for_cut_func}
    !!! Replaces original columns with bins if no 'new_names' passed
    """
    def __init__(self, bins_dict=None, new_names=None) -> None:
        #super().__init__()
        if new_names is not None and len(new_names) != len(bins_dict):
            raise('new_names length is not equal to bins_dict')
        self.bins_dict = bins_dict
        self.new_names = new_names
        return


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.bins_dict is None:
            return X
        X_mod = X.copy()
        if self.new_names is None:
            for col, bin_param in self.bins_dict.items():
                X_mod[col] = pd.cut(X_mod[col], bins=bin_param)
        else:
            for new_name, col_param in zip(self.new_names, self.bins_dict.items()):
                X_mod[new_name] = pd.cut(X_mod[col_param[0]], bins=col_param[1])
        return X_mod

#============================================================================================

class DFValuesMapper(BaseEstimator, TransformerMixin): # map_values=None
    """
    Wrapper class for pandas map function  
    map_values shoud be in form: {'column': {dict_of_mapping_values}}
    !!! IMPORTANT !!! mapping dict should contain ALL possible values. In other case missing values will be replaced with NAN
    """
    def __init__(self, map_values=None) -> None:
        self.map_values = map_values
        #super().__init__()
        pass


    """
    As this transformer is not fittable 'fit' is just dummy function for compatibility
    """
    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.map_values is None:
            return X
        X_mod = X.copy()
        for col in self.map_values:
            map_dict = self.map_values.get(col, None)
            if (map_dict is not None) and (col in X_mod.columns):
                X_mod[col] = X_mod[col].map(map_dict)
        return X_mod

#============================================================================================

class DFFuncApplyCols(BaseEstimator, TransformerMixin):
    """
    Wrapper class for using pandas apply function in pipelines
    with ability to apply different functions to different columns
    map_func = {'column': func | [func, args]}, where args is tuple to pass to func
    """
    def __init__(self, map_func=None) -> None:
        #super().__init__()
        self.map_func = map_func
        pass

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.map_func is None:
            return X
        X_mod = X.copy()
        for col, func in self.map_func.items():
            if col in X_mod.columns:
                if type(func) != type(list()):
                    X_mod[col] = X_mod[col].apply(func)
                else:
                    X_mod[col] = X_mod[col].apply(func[0], args=func[1])
        return X_mod

#============================================================================================

class DFCrossFeaturesImputer(BaseEstimator, TransformerMixin):
    """
    For example, I want to fill in missing education level. But I know corresponding job (which is 'housemaid')
    I want to get the most frequent education level for all records with job == 'housemaid'  
    I'm going to pass a dict {'education': 'job'} => fill column 'education' with most frequent values for value in job
    """
    def __init__(self, cross_features=None, strategy='most_frequent', nan_equiv=np.nan) -> None:
        if cross_features is None:
            raise('NoImputeObjectError')
        self.nan_equiv = nan_equiv
        self.strategy = strategy
        self.cross_features = cross_features
        self._imputers = None
        self._is_fitted = False
        #self._logger = logging.getLogger(name='ClassDFCrossFeatureImputer')
        #self._logger.debug('Initialization complete')
        return


    def fit(self, X, y=None):
        if self.cross_features is None:
            #self._logger.debug('No columns passed')
            return self
        self._imputers = {}
        # target_feat is column which contains missing values
        #self._logger.debug('Internal imputers initialized. Beginning loop through column pairs')
        for base_feat, reference_feat in self.cross_features.items():
            #self._logger.debug('\tBeginning training imputer for `{}` using `{}` feature'.format(target_feat, base_feat))
            self._imputers[base_feat] = self._get_reference_values(df=X, base_feat=base_feat, reference_feat=reference_feat, nan_equiv=self.nan_equiv)
        if len(self._imputers) > 0:
            self._is_fitted = True
        return self


    def _get_reference_values(self, df, base_feat, reference_feat, nan_equiv):
        is_numeric = True
        try:
            np.float64(nan_equiv)
        except ValueError:
            is_numeric = False
        if is_numeric and np.isnan(nan_equiv):
            ct = pd.crosstab(df[base_feat], df[reference_feat]) # rows = values of base feature, cols = values of reference feature
        else:
            ct = pd.crosstab(df.loc[(df[base_feat] != nan_equiv), base_feat], df[reference_feat])
        index = ct.index
        columns = ct.columns
        return {columns[i]: index[val] for i, val in enumerate(ct.to_numpy().argmax(axis=0))}


    """
    def _map_values(self, x, target_feat, base_feat, map_values):
        repl_val = map_values.get(x[base_feat], None)
        if repl_val is not None:
            x[target_feat] = repl_val
        return x
    """


    def transform(self, X, y=None):
        if not self._is_fitted:
            raise('NotFittedError')
        try:
            is_nan = np.isnan(self.nan_equiv)
        except TypeError:
            #self._logger.debug('\tCaptured TypeError while checking if `nan_equiv` is np.nan => is_nan = False')
            is_nan = False
        X_mod = X.copy()
        for target_feat, reference_feat in self.cross_features.items():
            map_values = self._imputers[target_feat]
            if is_nan:
                m = X_mod[target_feat].isna()
            else:
                m = X_mod[target_feat]==self.nan_equiv
            X_mod.loc[m, target_feat] = X_mod.loc[m, reference_feat].map(map_values)
        return X_mod

#============================================================================================

class DFOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes each column passed with individual OneHotEncoder
    Returns DataFrame with encoded columns only or full DataFrame with encoded columns appended to the end
    (original columns will be dropped by default until drop_originals == False)
    !!! Incompatible with sklearn.compose.ColumnTransformer !!!
    """
    def __init__(
        self,
        cols_cats,
        drop=None, # maybe worth to set 'if_binary' as default value
        sparse=None,
        dtype=np.float64,
        handle_unknown="error",
        col_overrule_params={},
        return_full_df=True,
        drop_originals=True,
    ) -> None:
        """
        Args:
            cols_cats: dictionary with columns names and categories {'column_name': 'auto' | list of array-like}
            dtype: resulting number type
            drop: 
            sparse: left for compatibility (will always return dense)
            handle_unknown: 'error' | 'ignore'
            col_overrule_params: dict to overrule default parameters for column
            return_full_df: if full data frame should be returned (default) with new encoded columns appended to the end or encoded columns only
            drop_originals: if original columns should be dropped from resulting DataFrame (works only if return_full_df == True)
        """
        self.cols_cats = cols_cats
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.col_overrule_params = col_overrule_params
        self.drop_originals = drop_originals
        self.return_full_df = return_full_df
        pass


    def fit(self, X, y=None):
        """
        Fit a separate OneHotEncoder for each of the columns in the dataframe
        Args:
            X: dataframe
            y: None, ignored. This parameter exists only for compatibility with Pipeline
        Returns
            self
        Raises
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        self.encoders_ = {}
        self.column_names_ = {}

        for c, cat in self.cols_cats.items():
            # Construct the OHE parameters using the arguments
            if cat == 'auto':
                categories = 'auto'
            else:
                categories = [cat]
            enc_params = {
                'categories': categories,
                'drop': self.drop,
                'sparse': False,
                'dtype': self.dtype,
                'handle_unknown': self.handle_unknown,
            }
            # and update it with potential overrule parameters for the current column
            enc_params.update(self.col_overrule_params.get(c, {}))

            # Regardless of how we got the parameters, make sure we always set the
            # sparsity to False
            enc_params["sparse"] = False

            # Now create, fit, and store the onehotencoder for current column c
            enc = OneHotEncoder(**enc_params)
            self.encoders_[c] = enc.fit(X.loc[:, [c]])

            # Get the feature names and replace each x0 with the original column name
            feature_names = enc.get_feature_names_out()
            feature_names = [x.replace("x0", c) for x in feature_names]
            #feature_names = [x.replace("x0_", "") for x in feature_names]
            #feature_names = [f"{c}_{x}" for x in feature_names]
            #feature_names = [f"{c}[{x}]" for x in feature_names]

            self.column_names_[c] = feature_names

        return self

        
    def transform(self, X, y=None):
        """
        Transform X using the one-hot-encoding per column
        Args:
            X: Dataframe that is to be one hot encoded
        Returns:
            Dataframe with onehotencoded data
        Raises
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f'X should be of type dataframe, not {type(X)}')

        if not hasattr(self, 'encoders_'):
            raise NotFittedError(f'{type(self).__name__} is not fitted')

        new_columns = []
        for c, enc in self.encoders_.items():
            transformed_col = enc.transform(X.loc[:, [c]])
            df_col = pd.DataFrame(transformed_col, columns=self.column_names_[c], index=X.index)
            new_columns.append(df_col)

        if self.return_full_df:
            X_transformed = X.copy()
            if self.drop_originals:
                X_transformed.drop(columns=self.encoders_.keys(), inplace=True)
            return pd.concat([X_transformed]+new_columns, axis=1)
        return pd.concat(new_columns, axis=1)

#============================================================================================

class DFOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes each column passed with individual OrdinalEncoder
    Returns DataFrame with encoded columns only or full DataFrame with encoded values in columns passed
    !!! Incompatible with sklearn.compose.ColumnTransformer !!!
    """

    def __init__(self, cols_cats=None, dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, col_overrule_params={}, return_full_df=True) -> None:
        """
        Args:
            cols_cats: dict = {'column_name': 'auto' | list of array-like}
            dtype: resulting number type
            handle_unknown: 'error' | 'use_encoded_value'
            unknown_value: int | np.nan, use if 'handle_unknown' == 'use_encoded_value'
            encoded_missing_value: int | np.nan
            col_overrule_params: dict to overrule default parameters for column
        """
        self.cols_cats = cols_cats
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.col_overrule_params = col_overrule_params
        self.return_full_df = return_full_df
        pass


    def fit(self, X, y=None):
        """
        Fit separate OrdinalEncoder for columns in 'columns' arg 
        Args:
            X: DataFrame
            y: None, ignored. This parameter exists only for compatibility with Pipeline
        Returns
            self
        Raises
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type pd.DataFrame, not {type(X)}")
        
        self.encoders_ = {}
        
        for c, cat in self.cols_cats.items():
            if cat == 'auto':
                categories = 'auto'
            else:
                categories = [cat]
            enc_params = {
                'categories': categories,
                'dtype': self.dtype,
                'handle_unknown': self.handle_unknown,
                'unknown_value': self.unknown_value,
                #'encoded_missing_value': self.encoded_missing_value
            }
            # and update it with potential overrule parameters for the current column
            enc_params.update(self.col_overrule_params.get(c, {}))
            
            enc = OrdinalEncoder(**enc_params)
            
            self.encoders_[c] = enc.fit(X.loc[:, [c]])
        return self


    def transform(self, X, y=None):
        """
        Transform X using the trained OrdinalEncoder per column
        Args:
            X: DataFrame to be encoded
        Returns:
            DataFrame with columns changed to encoded values
        Raises:
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type pd.DataFrame, not {type(X)}")

        if not hasattr(self, 'encoders_'):
            raise NotFittedError(f'{type(self).__name__} is not fitted')

        columns = []
        encoded = []
        for c, enc in self.encoders_.items():
            columns.append(c)
            encoded.append(pd.DataFrame(enc.transform(X.loc[:, [c]]), index=X.index, columns=[c]))
        
        transformed_df = pd.concat(encoded, axis=1)
        
        if self.return_full_df:
            X_transformed = X.copy()
            X_transformed[columns] = transformed_df
            return X_transformed
        else:
            return transformed_df

#============================================================================================

class DFValuesReplacer(BaseEstimator, TransformerMixin): # replaces=None | {'column_name': {value_to_replace: value_to_replace_with}}
    """
    """
    def __init__(self, replaces=None) -> None:
        """
        replaces should be the dict {'column_name': {value_to_replace: value_to_replace_with}}
        """
        self.replaces = replaces # replaces should be in form {'column1': {'value_to_find': 'value_to_replace_with'}, ...}
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.replaces is None or self.replaces == {}:
            return X

        X_transformed = X.copy()
        #for c, val in X_transformed.columns:
        X_transformed = X_transformed.replace(self.replaces)
        return X_transformed

#============================================================================================

if __name__ == "__main__":
    pass