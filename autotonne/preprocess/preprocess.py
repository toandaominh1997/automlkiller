import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
import sklearn.impute
import sklearn.feature_selection
import pyod
import pyod.models.knn
import pyod.models.iforest
import pyod.models.pca
import category_encoders as ce
class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy,
                 categorical_strategy,
                 fill_value_numerical = 0,
                 fill_value_categorical = 'not_available'):
        self.numeric_imputer = sklearn.impute.SimpleImputer(strategy=numeric_strategy, fill_value=fill_value_numerical)
        self.categorical_imputer = sklearn.impute.SimpleImputer(strategy=categorical_strategy, fill_value=fill_value_categorical)

    def fit(self, X, y = None):
        print('FIT SIMPLE IMPUTER')
        X = X.copy()
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        self.time_columns = X.select_dtypes(include=['datetime64[ns]']).columns
        if not self.numeric_columns.empty:
            self.numeric_imputer.fit(X[self.numeric_columns])

        if not self.categorical_columns.empty:
            self.categorical_imputer.fit(X[self.categorical_columns])

        if not self.time_columns.empty:
            self.most_frequent_time = []
            for col in self.time_columns:
                self.most_frequent_time.append(X[col].mode()[0])

        return self
    def transform(self, X, y = None):
        X = X.copy()
        if not self.numeric_columns.empty:
            X[self.numeric_columns] = self.numeric_imputer.transform(X[self.numeric_columns])
        if not self.categorical_columns.empty:
            X[self.categorical_columns] = self.categorical_imputer.transform(X[self.categorical_columns])
        if not self.time_columns.empty:
            for idx, col in enumerate(self.time_columns):
                X[col].fillna(self.most_frequent_time[idx])


        return X
    def fit_transform(self, X, y = None):
        X = X.copy()
        self.fit(X, y)
        return self.transform(X, y)
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols = None, method='onehotencoding'):
        if method == 'backwarddifferenceencoder':
            self.encoder = ce.BackwardDifferenceEncoder(cols=cols, return_df=True)
        elif method == 'baseencoder':
            self.encoder = ce.BaseNEncoder(cols=cols, return_df=True)
        elif method == 'binaryencoder':
            self.encoder = ce.BinaryEncoder(cols=cols, return_df=True)
        elif method == 'catboostencoder':
            self.encoder = ce.CatBoostEncoder(cols=cols, return_df=True)
        elif method == 'countencoder':
            self.encoder = ce.CountEncoder(cols=cols, return_df=True)
        elif method == 'glmmeencoder':
            self.encoder = ce.GLMMEncoder(cols=cols, return_df=True)
        elif method == 'hashingencoder':
            self.encoder = ce.HashingEncoder(cols=cols, return_df=True)
        elif method == 'helmerencoder':
            self.encoder = ce.HelmertEncoder(cols=cols, return_df=True)
        elif method == 'jamessteinencoder':
            self.encoder = ce.JamesSteinEncoder(cols=cols, return_df=True)
        elif method == 'leaveoneoutencoder':
            self.encoder = ce.LeaveOneOutEncoder(cols=cols, return_df=True)
        elif method == 'mestimateencoder':
            self.encoder = ce.MEstimateEncoder(cols=cols, return_df=True)
        elif method == 'onehotencoder':
            self.encoder = ce.OneHotEncoder(cols=cols, use_cat_names=True, return_df=True)
        elif method == 'ordinalencoder':
            self.encoder = ce.OrdinalEncoder(cols=cols, return_df=True)
        elif method == 'sumencoder':
            self.encoder = ce.SumEncoder(cols=cols, return_df=True)
        elif method == 'polynomialencoder':
            self.encoder = ce.PolynomialEncoder(cols=cols, return_df=True)
        elif method == 'targetencoder':
            self.encoder = ce.TargetEncoder(cols=cols, return_df=True)
        elif method == 'woeeencoder':
            self.encoder = ce.WOEEncoder(cols=cols, return_df=True)
    def fit(self, X, y = None):
        self.encoder.fit(X, y)
    def transform(self, X, y = None):
        return self.encoder.transform(X)

class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_discretize):
        self.features_to_discretize = features_to_discretize
        self.binns = []
    def fit(self, X, y = None):
        print('FIT BINNING')
        X = X.copy()
        if len(self.features_to_discretize) > 0:
            for col in self.features_to_discretize:
                hist, bin_edg = np.histogram(X[col], bins = 'sturges')
                self.binns.append(len(hist))
            self.disc = sklearn.preprocessing.KBinsDiscretizer(n_bins=self.binns,
                                                               encode = 'ordinal',
                                                               strategy='kmeans').fit(X[self.features_to_discretize])
        return self
    def transform(self, X, y = None):
        X = X.copy()
        if len(self.features_to_discretize) > 0:
            X[self.features_to_discretize] = self.disc.transform(X[self.features_to_discretize])
        return X

class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns = 'not_available', method = 'zscore'):
        self.method = method
        self.numeric_columns = numeric_columns
    def fit(self, X, y = None):
        print('FIT SCALING')
        X = X.copy()
        if isinstance(self.numeric_columns, str) and self.numeric_columns == 'not_available':
            return self
        if len(self.numeric_columns) == 0:
            self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        if self.method == 'zscore':
            self.scale = sklearn.preprocessing.StandardScaler()
        elif self.method == 'minmax':
            self.scale = sklearn.preprocessing.MinMaxScaler()
        elif self.method == 'yj':
            self.scale = sklearn.preprocessing.PowerTransformer(method = 'yeo-johnson', standardize=True)
        elif self.method == 'quantile':
            self.scale = sklearn.preprocessing.QuantileTransformer(output_distribution= 'normal')
        elif self.method == 'robust':
            self.scale = sklearn.preprocessing.RobustScaler()
        elif self.method =='maxabs':
            self.scale = sklearn.preprocessing.MaxAbsScaler()
        self.scale.fit(X[self.numeric_columns])
        return self
    def transform(self, X, y = None):
        X = X.copy()
        if isinstance(self.numeric_columns, str) and self.numeric_columns == 'not_available':
            return X
        X[self.numeric_columns] = self.scale.transform(X)
        return X
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

class Outlier(BaseEstimator, TransformerMixin):
    def __init__(self, methods = ['knn', 'iforest', 'pca'], contamination = 0.2, random_state = 42,  verbose = True):
        self.contamination = contamination
        self.random_state = random_state
        self.methods = methods
    def fit(self, X, y = None):
        print('FIT OUTLIER')
        X = X.copy()
        self.outlier = []
        if "knn" in self.methods:
            knn = pyod.models.knn.KNN(contamination = self.contamination).fit(X)
            self.outlier.append(knn)
        if 'iforest' in self.methods:
            iforest = pyod.models.iforest.IForest(contamination= self.contamination).fit(X)
            self.outlier.append(iforest)

        if 'pca' in self.methods:
            pca = pyod.models.pca.PCA(contamination=self.contamination).fit(X)
            self.outlier.append(pca)

    def transform(self, X, y = None):
        X = X.copy()
        X['vote_outlier'] = 0
        for out in self.outlier:
            X['vote_outlier'] += out.predict(X.drop(columns=['vote_outlier']))
        print('Remove outlier: {} rows'.format((X['vote_outlier']==len(self.methods)).sum()))
        if y is not None:
            y = y.loc[X['vote_outlier']!=len(self.methods)]
            X = X.loc[X['vote_outlier']!=len(self.methods)].drop(columns=['vote_outlier'])
            return X
        X = X.loc[X['vote_outlier']!=len(self.methods)].drop(columns=['vote_outlier'])
        return X
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

class ReduceCategoricalWithCount(BaseEstimator, TransformerMixin):
    def __init__(self,
                 categorical_columns=[]):
        self.categorical_columns = categorical_columns
    def fit(self, X, y = None):
        print('FIT REDUCE')
        X = X.copy()
        self.data_count = {}
        for col in self.categorical_columns:
            if col not in self.data_count.keys():
                self.data_count[col] = {}
            self.data_count[col] = X[col].value_counts().to_dict()
    def transform(self, X, y = None):
        for col in self.categorical_columns:
            X[col].replace(self.data_count[col].keys(), self.data_count[col].values(), inplace=True)
        return X
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)

class RecursiveFeatureElimination(BaseEstimator, TransformerMixin):
    def __init__(self,
                 estimator = None,
                 step = 1,
                 min_features_to_select = 2,
                 cv = 2):
        if cv == 0:
            self.selector = sklearn.feature_selection.RFE(estimator=estimator, step = step, n_features_to_select=min_features_to_select)
        else:
            self.selector = sklearn.feature_selection.RFECV(estimator=estimator, step = step, min_features_to_select=min_features_to_select, cv = cv, n_jobs=-1)
    def fit(self, X, y = None):
        X = X.copy()
        self.selector.fit(X, y)
        return self
    def transform(self, X, y = None):
        X = X.copy()
        X = X.loc[:, self.selector.support_]
        return X
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)
    def get_support(self, indices = False):
        return self.selector.get_support(indices=False)
    def get_ranking(self):
        return self.selector.ranking_

class ReduceDimensionForSupervised(BaseEstimator, TransformerMixin):
    def __init__(self,
                 method = 'pca_linear',
                 n_components = 0.99,
                 random_state = 42):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
    def fit(self, X, y = None):
        print('FIT REDUCE')
        X = X.copy()
        if self.method == 'pca_linear':
            self.model = sklearn.decomposition.PCA(self.n_components, random_state=self.random_state).fit(X)
        if self.method == 'kernal_pca':
            self.model = sklearn.decomposition.KernelPCA(self.n_components, random_state=self.random_state).fit(X)
        if self.method == 'tsne':
            self.model = sklearn.manifold.TSNE(self.n_components, random_state=self.random_state).fit(X)
        if self.method == 'incremental':
            self.model = sklearn.decomposition.IncrementalPCA(self.n_components, random_state = random_state)
        return self
    def transform(self, X, y = None):
        X = X.copy()
        return self.model.transform(X)
    def fit_transform(self, X, y = None):
        X = X.copy()
        self.fit(X)
        return self.transform(X)

def test(dataset, y):
    print(dataset.info())
    print('Start SimpleImputer ...')
    dataset = SimpleImputer(numeric_strategy='mean', categorical_strategy='most_frequent').fit_transform(dataset)
    print('Done SimpleImputer')
    print('Start Binning ...')
    dataset = Binning(features_to_discretize=['Jumps']).fit_transform(dataset)
    print('Done Binning')
    print('dataset: ', dataset)

    print('Start scaling ...')
    dataset = Scaling(numeric_columns=[], method = 'zscore').fit_transform(dataset)
    print('Done Binning')

    print('Start reducecategorywithcount ...')
    dataset = ReduceCategoricalWithCount(categorical_columns=['Jumps']).fit_transform(dataset)
    print('Done reducecategorywithcount')
    print(dataset)
    print('Start Outlier ...')
    dataset, y = Outlier().fit_transform(dataset, y)
    print('Done Outlier')

    kaka = RecursiveFeatureElimination(estimator=sklearn.ensemble.RandomForestClassifier(), min_features_to_select=3, cv=3).fit(dataset, y)
    print('selector: ',  kaka.get_support())
    print('Start RecursiveFeatureElimination ...')
    dataset = RecursiveFeatureElimination(estimator=sklearn.ensemble.RandomForestClassifier()).fit_transform(dataset, y)
    print('Done RecursiveFeatureElimination')
    print(dataset)
    print('Start ReduceDimensionForSupervised ...')
    dataset = ReduceDimensionForSupervised().fit_transform(dataset)
    print('Done ReduceDimensionForSupervised')
    print('dataset: ', dataset)
if __name__=='__main__':
    X, y = sklearn.datasets.load_linnerud(return_X_y=True, as_frame=True)

    print(y['Waist'])
    test(X, y['Waist'])
    # from sklearn.pipeline import Pipeline

    # pipe = Pipeline(['impute': SimpleImputer()])




