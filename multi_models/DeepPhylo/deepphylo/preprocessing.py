# code from https://github.com/biocore/DEICODE/blob/master/deicode/preprocessing.py

from skbio.diversity._util import _vectorize_counts_and_tree
import numpy as np


def fast_unifrac(table, tree):
    """
    A wrapper to return a vectorized Fast UniFrac
    algorithm. The nodes up the tree are summed
    and exposed as vectors in the matrix. The
    closed matrix is then multipled by the
    branch lengths to phylogenically
    weight the data.

    Parameters
    ----------
    table : biom.Table
       A biom table of counts.
    tree: skbio.TreeNode
       Tree containing the features in the table.
    Returns
    -------
    counts_by_node: array_like, np.float64
       A matrix of counts with internal nodes
       vectorized.
    tree_index: dict
        A housekeeping dictionary.
    branch_lengths: array_like, np.float64
        An array of branch lengths.
    fids: list
        A list of feature IDs matched to tree_index['id'].
    otu_ids: list
        A list of the original table OTU IDs (tips).
    Examples
    --------
    TODO

    """

    # original table
    bt_array = table.matrix_data.toarray()
    otu_ids = table.ids('observation')
    # expand the vectorized table
    counts_by_node, tree_index, branch_lengths \
        = _vectorize_counts_and_tree(bt_array.T, otu_ids, tree)
    # check branch lengths
    if sum(branch_lengths) == 0:
        raise ValueError('All tree branch lengths are zero. '
                         'This will result in a table of zero features.')
    # drop zero sum features (non-optional for CTF/RPCA)
    keep_zero = counts_by_node.sum(0) > 0
    # drop zero branch_lengths (no point to keep it)
    node_branch_zero = branch_lengths.sum(0) > 0
    # combine filters
    keep_node = (keep_zero & node_branch_zero)
    # subset the table (if need, otherwise ignored)
    counts_by_node = counts_by_node[:, keep_node]
    branch_lengths = branch_lengths[keep_node]
    fids = ['n' + i for i in list(tree_index['id'][keep_node].astype(str))]
    tree_index['keep'] = {i: v for i, v in enumerate(keep_node)}
    # re-label tree to return with labels
    tree_relabel = {tid_: tree_index['id_index'][int(tid_[1:])]
                    for tid_ in fids}
    # re-name nodes to match vectorized table
    otu_ids_set = set(otu_ids)
    for new_id, node_ in tree_relabel.items():
        if node_.name in otu_ids_set:
            # replace table name (leaf - nondup)
            fids[fids.index(new_id)] = node_.name
        else:
            # replace tree name (internal)
            node_.name = new_id

    return counts_by_node, tree_index, branch_lengths, fids, otu_ids

def matrix_rclr(mat, branch_lengths=None):
    """
    Robust clr transform helper function.
    This function is built for mode 2 tensors,
    also known as matrices.

    Raises
    ------
    ValueError
       Raises an error if any values are negative.
    ValueError
       Raises an error if the matrix has more than 2 dimension.

    References
    ----------
    .. [1] V. Pawlowsky-Glahn, J. J. Egozcue,
           R. Tolosana-Delgado (2015),
           Modeling and Analysis of
           Compositional Data, Wiley,
           Chichester, UK

    .. [2] C. Martino et al., A Novel Sparse
           Compositional Technique Reveals
           Microbial Perturbations. mSystems.
           4 (2019), doi:10.1128/mSystems.00016-19.

    Examples
    --------
    TODO

    """
    # ensure array is at least 2D
    mat = np.atleast_2d(np.array(mat))
    # ensure array not more than 2D
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    # ensure no neg values
    if (mat < 0).any():
        raise ValueError('Array Contains Negative Values')
    # ensure no undefined values
    if np.count_nonzero(np.isinf(mat)) != 0:
        raise ValueError('Data-matrix contains either np.inf or -np.inf')
    # ensure no missing values
    if np.count_nonzero(np.isnan(mat)) != 0:
        raise ValueError('Data-matrix contains nans')
    # take the log of the sample centered data
    if branch_lengths is not None:
        mat = np.log(matrix_closure(matrix_closure(mat) * branch_lengths))
    else:
        mat = np.log(matrix_closure(mat))
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

def matrix_closure(mat):
    """
    Simillar to the skbio.stats.composition.closure function.
    Performs closure to ensure that all elements add up to 1.
    However, this function allows for zero rows. This results
    in rows that may contain missing (NaN) vlues. These
    all zero rows may occur as a product of a tensor slice and
    is dealt later with the tensor restructuring and factorization.

    Parameters
    ----------
    mat : array_like
       a matrix of proportions where
       rows = compositions
       columns = components
    Returns
    -------
    array_like, np.float64
       A matrix of proportions where all of the values
       are nonzero and each composition (row) adds up to 1
    Examples
    --------
    >>> import numpy as np
    >>> from gemelli.preprocessing import matrix_closure
    >>> X = np.array([[2, 2, 6], [0, 0, 0]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ nan,  nan,  nan]])

    """

    mat = np.atleast_2d(mat)
    mat = mat / mat.sum(axis=1, keepdims=True)

    return mat.squeeze()
