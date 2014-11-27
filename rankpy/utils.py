import os
import numpy
import scipy.sparse

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def pickle(obj, filepath, protocol=-1):
    '''
    Pickle the object into the specified file.

    Parameters:
    -----------
    obj: object
        The object that should be serialized.

    filepath:
        The location of the resulting pickle file.
    '''
    with open(filepath, 'wb') as fout:
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(filepath):
    '''
    Unpicle the object serialized in the specified file.

    Parameters:
    -----------
    filepath:
        The location of the file to unpickle.
    '''
    with open(filepath) as fin:
        return _pickle.load(fin)


def save_spmatrix(filename, X, compress=False, tempdir=None):
    """
    Serializes X into file using numpy .npz format
    with an optional compression.
    """
    is_csc = False
    if scipy.sparse.isspmatrix_csc(X):
        is_csc = True
    elif not scipy.sparse.isspmatrix_csr(X):
        raise TypeError('X is not a sparse matrix.')

    is_csc = numpy.asarray(is_csc)

    npz = dict(data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape, is_csc=is_csc)
    save_fcn = numpy.savez_compressed if compress else numpy.savez

    old_tmpdir = os.environ.get('TMPDIR')
    os.environ['TMPDIR'] = os.getcwd() if tempdir is None else tempdir

    try:
        save_fcn(filename, **npz)
    finally:
        if old_tmpdir is None:
            del os.environ['TMPDIR']
        else:
            os.environ['TMPDIR'] = old_tmpdir


def load_spmatrix(filename):
    """
    Load a sparse matrix X from the specified .npz file.
    """
    npz = numpy.load(filename)

    if npz['is_csc']:
        return scipy.sparse.csc_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
    else:
        return scipy.sparse.csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
