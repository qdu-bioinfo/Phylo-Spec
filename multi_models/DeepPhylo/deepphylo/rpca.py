# code from https://github.com/biocore/DEICODE/blob/master/deicode/rpca.py

import biom
import skbio
import pandas as pd
from typing import Union
from scipy.spatial import distance
from scipy.linalg import svd
from deepphylo.optspace_and_base import OptSpace, _BaseImpute


import numpy as np
from skbio.stats.composition import closure
# need to ignore log of zero warning
np.seterr(all='ignore')

# Configuration file where you can set the parameter default values and
# descriptions. This is used by both the standalone RPCA and QIIME 2 RPCA sides
# of DEICODE.
DEFAULT_RANK = 3
DEFAULT_MSC = 500
DEFAULT_MFC = 10
DEFAULT_MFF = 0
DEFAULT_ITERATIONS = 5

DESC_RANK = ("The underlying low-rank structure."
             " The input can be an integer "
             "(suggested: 1 < rank < 10) [minimum 2]."
             " Note: as the rank increases the runtime"
             " will increase dramatically.")
DESC_MSC = ("Minimum sum cutoff of sample across all features. "
            "The value can be at minimum zero and must be an whole"
            " integer. It is suggested to be greater than or equal"
            " to 500.")
DESC_MFC = ("Minimum sum cutoff of features across all samples. "
            "The value can be at minimum zero and must be an whole"
            " integer")
DESC_MFF = ("Minimum percentage of samples a feature must appear"
            " with a value greater than zero. This value can range"
            " from 0 to 100 with decimal values allowed.")
DESC_ITERATIONS = ("The number of iterations to optimize the solution"
                   " (suggested to be below 100; beware of overfitting)"
                   " [minimum 1]")

def preprocess_table(table, min_sample_count=500, min_feature_count=10, min_feature_frequency=0):
    # get shape of table
    n_features, n_samples = table.shape
    sample_names = {v: k for k, v in enumerate(table.ids(axis='sample'))}
    filtered_samples = [] 
    # filter sample to min seq. depth
    def sample_filter(val, id_, md):
        return sum(val) > min_sample_count
    def sample_filter(val, id_, md):
        if sum(val) <= min_sample_count:
            filtered_samples.append(sample_names[id_])
            return False  
        else:
            return True
    # filter features to min total counts
    def observation_filter(val, id_, md):
        return sum(val) > min_feature_count
    # filter features by N samples presence
    def frequency_filter(val, id_, md):
        if (np.sum(val > 0) / n_samples) <= (min_feature_frequency / 100):
            filtered_samples.append(sample_names[id_])
            return False  
        else:
            return True
    # filter and import table for each filter above
    table = table.filter(observation_filter, axis='observation')
    table = table.filter(frequency_filter, axis='observation')
    table = table.filter(sample_filter, axis='sample')
    table = table.to_dataframe().T
    return table, filtered_samples


def rclr(mat):

    # ensure array is at leadt 2D
    mat = np.atleast_2d(np.array(mat))
    # ensure no missing values
    if (mat < 0).any():
        raise ValueError('Array Contains Negative Values')
    # ensure no undefined values
    if np.count_nonzero(np.isinf(mat)) != 0:
        raise ValueError('Data-mat contains either np.inf or -np.inf')
    # ensure no missing values
    if np.count_nonzero(np.isnan(mat)) != 0:
        raise ValueError('Data-mat contains nans')
    # take the log of the sample centered data
    mat = np.log(closure(mat))
    # generate a mask of missing values
    mask = [True] * mat.shape[0] * mat.shape[1]
    mask = np.array(mat).reshape(mat.shape)
    mask[np.isfinite(mat)] = False
    # sum of rows (features)
    lmat = np.ma.array(mat, mask=mask)
    # perfrom geometric mean
    gm = lmat.mean(axis=-1, keepdims=True)
    # center with the geometric mean
    lmat = (lmat - gm).squeeze().data
    # mask the missing with nan
    lmat[~np.isfinite(mat)] = np.nan
    return lmat

def rpca(table: biom.Table,
         n_components: Union[int, str] = DEFAULT_RANK,
         min_sample_count: int = DEFAULT_MSC,
         min_feature_count: int = DEFAULT_MFC,
         min_feature_frequency: float = DEFAULT_MFF,
         max_iterations: int = DEFAULT_ITERATIONS) -> (
        skbio.OrdinationResults,
        skbio.DistanceMatrix):
    """Runs RPCA with an rclr preprocessing step.

       This code will be run by both the standalone and QIIME 2 versions of
       DEICODE.
    """
    # get shape of table
    n_features, n_samples = table.shape

    # filter sample to min seq. depth
    def sample_filter(val, id_, md):
        return sum(val) > min_sample_count

    # filter features to min total counts
    def observation_filter(val, id_, md):
        return sum(val) > min_feature_count

    # filter features by N samples presence
    def frequency_filter(val, id_, md):
        return (np.sum(val > 0) / n_samples) > (min_feature_frequency / 100)

    # filter and import table for each filter above
    table = table.filter(observation_filter, axis='observation')
    table = table.filter(frequency_filter, axis='observation')
    table = table.filter(sample_filter, axis='sample')
    table = table.to_dataframe().T
    # check the table after filtering
    if len(table.index) != len(set(table.index)):
        raise ValueError('Data-table contains duplicate indices')
    if len(table.columns) != len(set(table.columns)):
        raise ValueError('Data-table contains duplicate columns')
    # Robust-clt (rclr) preprocessing and OptSpace (RPCA)
    opt = MatrixCompletion(n_components=n_components,
                           max_iterations=max_iterations).fit(rclr(table))
    # get new n-comp when applicable
    n_components = opt.s.shape[0]
    # get PC column labels for the skbio OrdinationResults
    rename_cols = ['PC' + str(i + 1) for i in range(n_components)]
    # get completed matrix for centering
    X = opt.sample_weights @ opt.s @ opt.feature_weights.T
    # center again around zero after completion
    X = X - X.mean(axis=0)
    X = X - X.mean(axis=1).reshape(-1, 1)
    # re-factor the data
    u, s, v = svd(X)
    # only take n-components
    u = u[:, :n_components]
    v = v.T[:, :n_components]
    # calc. the new variance using projection
    p = s**2 / np.sum(s**2)
    p = p[:n_components]
    s = s[:n_components]
    # save the loadings
    feature_loading = pd.DataFrame(v, index=table.columns,
                                   columns=rename_cols)
    sample_loading = pd.DataFrame(u, index=table.index,
                                  columns=rename_cols)
    # % var explained
    proportion_explained = pd.Series(p, index=rename_cols)
    # get eigenvalues
    eigvals = pd.Series(s, index=rename_cols)

    # if the n_components is two add PC3 of zeros
    # this is referenced as in issue in
    # <https://github.com/biocore/emperor/commit
    # /a93f029548c421cb0ba365b4294f7a5a6b0209ce>
    # discussed in DEICODE -- PR#29
    if n_components == 2:
        feature_loading['PC3'] = [0] * len(feature_loading.index)
        sample_loading['PC3'] = [0] * len(sample_loading.index)
        eigvals.loc['PC3'] = 0
        proportion_explained.loc['PC3'] = 0

    # save ordination results
    short_method_name = 'rpca_biplot'
    long_method_name = '(Robust Aitchison) RPCA Biplot'
    ord_res = skbio.OrdinationResults(
        short_method_name,
        long_method_name,
        eigvals.copy(),
        samples=sample_loading.copy(),
        features=feature_loading.copy(),
        proportion_explained=proportion_explained.copy())
    # save distance matrix
    dist_res = skbio.stats.distance.DistanceMatrix(
        opt.distance, ids=sample_loading.index)

    return ord_res, dist_res

def rpca_table(table: pd.DataFrame,
         n_components: Union[int, str] = DEFAULT_RANK,
         max_iterations: int = DEFAULT_ITERATIONS) -> (
        skbio.OrdinationResults,
        skbio.DistanceMatrix):
    """
        Runs RPCA with an rclr preprocessing step.
        This code will be run by both the standalone and QIIME 2 versions of DEICODE.
    """
    # check the table after filtering
    if len(table.index) != len(set(table.index)):
        raise ValueError('Data-table contains duplicate indices')
    if len(table.columns) != len(set(table.columns)):
        raise ValueError('Data-table contains duplicate columns')
    # Robust-clt (rclr) preprocessing and OptSpace (RPCA)
    opt = MatrixCompletion(n_components=n_components,
                           max_iterations=max_iterations).fit(table.values)
    # get new n-comp when applicable
    n_components = opt.s.shape[0]
    # get PC column labels for the skbio OrdinationResults
    rename_cols = ['PC' + str(i + 1) for i in range(n_components)]
    # get completed matrix for centering
    X = opt.sample_weights @ opt.s @ opt.feature_weights.T
    # center again around zero after completion
    X = X - X.mean(axis=0)
    X = X - X.mean(axis=1).reshape(-1, 1)
    # re-factor the data
    u, s, v = svd(X)
    # only take n-components
    u = u[:, :n_components]
    v = v.T[:, :n_components]
    # calc. the new variance using projection
    p = s**2 / np.sum(s**2)
    p = p[:n_components]
    s = s[:n_components]
    # save the loadings
    feature_loading = pd.DataFrame(v, index=table.columns,
                                   columns=rename_cols)
    sample_loading = pd.DataFrame(u, index=table.index,
                                  columns=rename_cols)
    # % var explained
    proportion_explained = pd.Series(p, index=rename_cols)
    # get eigenvalues
    eigvals = pd.Series(s, index=rename_cols)

    # if the n_components is two add PC3 of zeros
    # this is referenced as in issue in
    # <https://github.com/biocore/emperor/commit
    # /a93f029548c421cb0ba365b4294f7a5a6b0209ce>
    # discussed in DEICODE -- PR#29
    if n_components == 2:
        feature_loading['PC3'] = [0] * len(feature_loading.index)
        sample_loading['PC3'] = [0] * len(sample_loading.index)
        eigvals.loc['PC3'] = 0
        proportion_explained.loc['PC3'] = 0

    # # save ordination results
    # short_method_name = 'rpca_biplot'
    # long_method_name = '(Robust Aitchison) RPCA Biplot'
    # ord_res = skbio.OrdinationResults(
    #     short_method_name,
    #     long_method_name,
    #     eigvals.copy(),
    #     samples=sample_loading.copy(),
    #     features=feature_loading.copy(),
    #     proportion_explained=proportion_explained.copy())
    # # save distance matrix
    # dist_res = skbio.stats.distance.DistanceMatrix(
    #     opt.distance, ids=sample_loading.index)

    return feature_loading, sample_loading, eigvals, proportion_explained, opt.distance

class MatrixCompletion(_BaseImpute):

    def __init__(self, n_components=2, max_iterations=5, tol=1e-5):
        """

        This form of matrix completion uses OptSpace (1). Furthermore,
        here we directly interpret the loadings generated from matrix
        completion as a dimensionality reduction.

        Parameters
        ----------

        X: numpy.ndarray - a rclr preprocessed matrix of shape (M,N)
        N = Features (i.e. OTUs, metabolites)
        M = Samples

        n_components: int, optional : Default is 2
        The underlying rank of the default set
        to 2 as the default to prevent overfitting.

        max_iterations: int, optional : Default is 5
        The number of convex iterations to optimize the solution
        If iteration is not specified, then the default iteration is 5.
        Which redcues to a satisfactory error threshold.

        tol: float, optional : Default is 1e-5
        Error reduction break, if the error reduced is
        less than this value it will return the solution

        Returns
        -------
        U: numpy.ndarray - "Sample Loadings" or the unitary matrix
        having left singular vectors as columns. Of shape (M,n_components)

        s: numpy.ndarray - The singular values,
        sorted in non-increasing order. Of shape (n_components,n_components).

        V: numpy.ndarray - "Feature Loadings" or Unitary matrix
        having right singular vectors as rows. Of shape (N,n_components)

        solution: numpy.ndarray - (U*S*V.transpose()) of shape (M,N)

        distance: numpy.ndarray - Distance between each
        pair of the two collections of inputs. Of shape (M,M)

        Raises
        ------
        ValueError

        ValueError
            `ValueError: n_components must be at least 2`.

        ValueError
            `ValueError: max_iterations must be at least 1`.

        ValueError
            `ValueError: Data-table contains either np.inf or -np.inf`.

        ValueError
            `ValueError: The n_components must be less
             than the minimum shape of the input table`.

        References
        ----------
        .. [1] Keshavan RH, Oh S, Montanari A. 2009. Matrix completion
                from a few entries (2009_ IEEE International
                Symposium on Information Theory

        Examples
        --------
        TODO

        """

        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol

        return

    def fit(self, X):
        """
        Fit the model to X_sparse
        """

        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self

    def _fit(self):

        # make copy for imputation, check type
        X_sparse = self.X_sparse
        n, m = X_sparse.shape

        # make sure the data is sparse (otherwise why?)
        if np.count_nonzero(np.isinf(X_sparse)) != 0:
            raise ValueError('Contains either np.inf or -np.inf')

        # test n-iter
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        # check the settings for n_components
        if isinstance(self.n_components, str) and \
           self.n_components.lower() == 'auto':
            # estimate the rank of the matrix
            self.n_components = 'auto'
        # check hardset values
        elif isinstance(self.n_components, int):
            if self.n_components > (min(n, m) - 1):
                raise ValueError("n-components must be at most"
                                 " 1 minus the min. shape of the"
                                 " input matrix.")
            if self.n_components < 2:
                raise ValueError("n-components must "
                                 "be at least 2")
        # otherwise rase an error.
        else:
            raise ValueError("n-components must be "
                             "an interger or 'auto'")

        # return solved matrix
        self.U, self.s, self.V = OptSpace(n_components=self.n_components,
                                          max_iterations=self.max_iterations,
                                          tol=self.tol).solve(X_sparse)
        # save the solution (of the imputation)
        self.solution = self.U.dot(self.s).dot(self.V.T)
        self.eigenvalues = np.diag(self.s)
        self.explained_variance_ratio = list(
            self.eigenvalues / self.eigenvalues.sum())
        self.distance = distance.cdist(self.U, self.U)
        self.feature_weights = self.V
        self.sample_weights = self.U

    def fit_transform(self, X):
        """
        Returns the final SVD of

        U: numpy.ndarray - "Sample Loadings" or the
        unitary matrix having left singular
        vectors as columns. Of shape (M,n_components)

        s: numpy.ndarray - The singular values,
        sorted in non-increasing order. Of shape (n_components,n_components).

        V: numpy.ndarray - "Feature Loadings" or Unitary matrix
        having right singular vectors as rows. Of shape (N,n_components)

        """
        X_sparse = X.copy().astype(np.float64)
        self.X_sparse = X_sparse
        self._fit()
        return self.sample_weights, self.s, self.feature_weights
