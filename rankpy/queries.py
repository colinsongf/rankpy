# -*- coding: utf-8 -*-
#
# This file is part of RankPy.
#
# RankPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy as np
import scipy.sparse as sp

from operator import itemgetter
from itertools import chain, izip
from .utils import pickle, unpickle
from collections import defaultdict


logger = logging.getLogger(__name__)


class Bunch(dict):
    '''
    Container object for convenient data transfer: dictionary-like
	object, which exposes its keys as attributes.
    '''
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class Query(object):
    '''
    Data structure representing a single query. It can be either
    stand-alone query or a query that is a part of a dataset, in
    which case the query is just a view into the fields of the
    particular query and any changes will be effectively reflected
    inside the dataset.

    Parameters:
    -----------
    qid: int
        The query (unique) identifier.

    relevance_scores: array, shape = (n_documents,)
        The relevance scores of the associated documents.

    feature_vectors: array, shape = (n_documents, n_features)
        The vector space representations of the documents associated with the query.

    base: Queries object, optional (default is None)
        The set of queries, which this query comes from. If not None, all the
        (non-primitive) internal parts of this object are views into this queries
        data structure, so any changes will be reflected accordingly.
    '''
    def __init__(self, qid, relevance_scores, feature_vectors, base=None):
        self.qid = qid
        self.max_score = relevance_scores.max()
        self.relevance_scores = relevance_scores
        self.feature_vectors = feature_vectors
        self.base = base


    def document_count(self):
        '''
        Return the number of documents for the query.
        '''
        return self.feature_vectors.shape[0]


    def get_feature_vectors(self):
        '''
        Return the feature vectors of the documents for the query.
        '''
        return self.feature_vectors


    def highest_relevance(self):
        '''
        Return the maximum relevance score of a document
        associated with the query.
        '''
        return self.max_score


    def __str__(self):
        return 'Query (qid: %d, documents: %d)' % (self.qid, self.relevance_scores.shape[0])


class Queries(object):
    '''
    Data structure representing queries used for training learning to
	rank algorithms. It is created from query-document feature vectors,
	corresponding relevance scores, and index mapping from queries to
	associated query-document feature vectors.

    Parameters:
    -----------
    feature_vectors: array, shape = (# of query-document pairs, # of features)
        The feature vectors for query-document pairs.

    relevance_scores: array, shape = (# of query-document pairs,)
        The relevance scores correspoding to the feature vectors.

    query_indptr: array
        The query index pointer into the feature_vectors and relevance_scores array,
		i.e. the document feature vectors, feature_vectors[query_indptr[i]:query_indptr[i + 1]],
		and the corresponding relevance scores, relevance_scores[query_indptr[i]:query_indptr[i + 1]], 
		are the feature vectors and relevance scores for the i-th query documents.

    max_score: int, optional (default is None)
        The maximum relevance score value. If None, the value is derived
        from the specified relevance scores.

    has_sorted_relevances: bool, optional (default is False)
        If True, it indicates that the relevance scores are sorted in decreasing
        order. Use to surpass sorting when it has been already done.

    feature_indices: array, shape = (# of features,), optional (default is None)
        The set of original feature indices (relevant only for reporting).
    '''
    def __init__(self, feature_vectors, relevance_scores, query_indptr, max_score=None, has_sorted_relevances=False, query_ids=None, feature_indices=None):
        self.feature_vectors = np.asanyarray(feature_vectors, dtype=np.float32)
        self.relevance_scores = np.asanyarray(relevance_scores, dtype=np.intc).ravel()
        self.query_indptr = np.asanyarray(query_indptr, dtype=np.intc).ravel()

        min_relevance = self.relevance_scores.min()

        if min_relevance > 0:
            self.relevance_scores -= min_relevance

        self.n_queries = self.query_indptr.shape[0] - 1
        self.n_feature_vectors = self.feature_vectors.shape[0]

        if query_ids is not None:
            self.query_ids = np.asanyarray(query_ids).ravel()
            assert self.query_ids.shape[0] == self.n_queries, 'the number of queries (%d) != the number of query ids (%d)' \
                                                               % (self.n_queries, self.query_ids.shape[0])
        if feature_indices is not None:
            self.feature_indices = np.asanyarray(feature_indices).ravel()
            assert self.feature_vectors.shape[1] == feature_indices.shape[0], 'the number of features (%d) != the number of feature indices (%d)' \
                                                                               % (self.feature_vectors.shape[1], feature_indices.shape[0])
        # Make sure shapes are consistent.
        assert self.n_feature_vectors == self.relevance_scores.shape[0], 'the number of documents does not equal the number of relevance scores'
        assert self.n_feature_vectors == self.query_indptr[-1], 'the query index pointer is not correct (number of indexed items is not the same as the number of documents)'
        
        if max_score is None:
            max_score = self.relevance_scores.max()

        self.max_score = max_score

        if not has_sorted_relevances:
            # Find the indices to make the documents sorted according to their relevance scores. We create
            # one huge array of indices to tacke the potentially big beast (feature_vectors) at once 
            # (since fancy indexing makes a copy it can be more advantageous to do it once -- but it needs
            # 2x as much memory, of course).
            relevance_scores_sorted_indices = np.fromiter(chain(*[self.query_indptr[i] + np.argsort(self.relevance_scores[self.query_indptr[i]:self.query_indptr[i + 1]])[::-1] for i in range(self.n_queries)]), dtype=np.intc, count=self.n_feature_vectors)
            self.feature_vectors = self.feature_vectors[relevance_scores_sorted_indices, :]
            self.relevance_scores = self.relevance_scores[relevance_scores_sorted_indices]

        # Store for each query where in its document list the relevance scores changes.
        self.query_relevance_strides = np.empty((self.n_queries, self.max_score + 1), dtype=np.intc)
        self.query_relevance_strides.fill(-1)

        # Compute relevance score strides for each query (the scores need to be sorted first), i.e.
        # query_relevance_strides[i,j] denotes the smallest index of a document that is less relevant
        # than the relevance score j in the document list of the i-th query.
        for i in range(self.n_queries):
            query_relevance_scores = self.relevance_scores[self.query_indptr[i]:self.query_indptr[i + 1]]
            query_relevance_starts = np.where(np.diff(query_relevance_scores))[0]
            # Special cases: all documents have the same relevance or there is no irrelevant document.
            if query_relevance_starts.size == 0 or query_relevance_scores[-1] > 0:
                query_relevance_starts = np.append(query_relevance_starts, query_relevance_scores.size - 1)
            query_relevance_starts += self.query_indptr[i]
            self.query_relevance_strides[i, self.relevance_scores[query_relevance_starts]] = query_relevance_starts + 1

        # Just to make use of the space (which might be found useful later).
        self.query_relevance_strides[:,0] = self.query_indptr[1:]


    def __str__(self):
        return 'Queries (%d queries, %d documents, %d max. relevance)' % (self.n_queries, self.n_feature_vectors, self.max_score)


    @staticmethod
    def load_from_text(filepaths, dtype=np.float32, max_score=None, has_sorted_relevances=False):
        '''
        Load queries in the svmlight format from the specified file(s).

        SVMlight format example (one line):

            5[\s]qid:8[\s]103:1.0[\s]110:-1.0[\s]111:-1.0[\s]...[\s]981:1.0 982:1.0 # comment[\n]

        Parameters:
        -----------
        filepath: string or list of strings
            The location of the dataset file(s).

        dtype: data-type, optional (default is np.float32)
            The desired data-type for the document feature vectors. Here,
            the default value (np.float32) is chosen for mere optimization
            purposes.

        max_score: int, optional (default is None)
            The maximum relevance score value. If None, the value is derived
            from the relevance scores in the file.

        has_sorted_relevances: bool, optional (default is False)
            If True, it indicates that the relevance scores of the queries in the file
            are sorted in decreasing order.
        '''
        # Arrays used to build CSR matrix of query-document vectors.
        data, indices, indptr = [], [], [0]

        # Relevance score, query ID, query hash, and document hash.
        relevances = []

        query_ids = []
        query_indptr = [0]
        prev_qid = None

        # If only single filepath is given, not a list.
        if isinstance(filepaths, basestring):
            filepaths = [filepaths]

        n_feature_vectors = 0
        
        for filepath in filepaths:
            lineno = 0 # Used just to report invalid lines (if any).
    
            logger.info('Reading queries from %s.' % filepath)

            with open(filepath, 'rb') as ifile:
                # Loop through every line containing query-document pair.
                for pair in ifile:
                    lineno += 1
                    try:
                        comment_start = pair.find('#')

                        # Remove the line comment first.
                        if comment_start >= 0:
                            pair = pair[:comment_start]

                        pair = pair.strip()

                        # Skip comments and empty lines.
                        if not pair:
                            continue

                        # Sadly, the distinct items on the line are not properly separated
                        # by a single delimiter. We split using all whitespaces here.
                        items = pair.split()

                        # Relevance is the first number on the line.
                        relevances.append(int(items[0]))

                        # Query ID follows the second item on the line, which is 'qid:'.
                        qid = int(items[1].split(':')[1])

                        if qid != prev_qid:
                            query_ids.append(qid)
                            query_indptr.append(query_indptr[-1] + 1)
                            prev_qid = qid
                        else:
                            query_indptr[-1] += 1

                        # Load the feature vector into CSR arrays.
                        for fidx, fval in map(lambda s: s.split(':'), items[2:]):
                            data.append(dtype(fval))
                            indices.append(int(fidx))
                        indptr.append(len(indices))

                        n_feature_vectors += 1

                        if n_feature_vectors % 10000 == 0:
                            logger.info('Read %d queries and %d documents so far.' \
                                        % (len(query_indptr) - 1, n_feature_vectors))
                    except:
                        # Ill-formated line (it should not happen). Print line number
                        print 'Ill-formated line: %d' % lineno
                        raise

                logger.info('Read %d queries and %d documents in total.' \
                            % (len(query_indptr) - 1, n_feature_vectors))

        # Remap the features into 0:(# unique feature indices) range.
        feature_indices = np.unique(indices)
        indices = np.searchsorted(feature_indices, indices)

        feature_vectors = sp.csr_matrix((data, indices, indptr), dtype=dtype,
                                        shape=(n_feature_vectors, len(feature_indices)))

        # Free the copies of the feature_vectors in non-Numpy arrays (if any), this
        # is important in order not to waste memory for the transfer of the
        # feature vectors to dense format (default option).
        if feature_vectors.data is not data: del data
        if feature_vectors.indices is not indices: del indices
        if feature_vectors.indptr is not indptr: del indptr

        feature_vectors = feature_vectors.toarray()

        # Create and return a Queries object.
        return Queries(feature_vectors, relevances, query_indptr, max_score=max_score,
                       has_sorted_relevances=has_sorted_relevances, query_ids=query_ids,
                       feature_indices=feature_indices)


    def save_as_text(self, filepath, shuffle=False):
        '''
        Save queries into the specified file in svmlight format.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        # Inflate the query_ids array such that each id covers the corresponding feature vectors.        
        query_ids = np.fromiter(chain(*[[qid] * cnt for qid, cnt in zip(self.query_ids, np.diff(self.query_indptr))]), dtype=int)
        relevance_scores = self.relevance_scores
        feature_vectors = self.feature_vectors

        if shuffle:
            shuffle_indices = np.random.permutation(self.document_count())
            reshuffle_indices = np.argsort(query_ids[shuffle_indices])
            document_shuffle_indices = np.arange(self.document_count(), dtype=np.intc)[shuffle_indices[reshuffle_indices]]
            query_ids = query_ids[document_shuffle_indices]
            relevance_scores = relevance_scores[document_shuffle_indices]
            feature_vectors = feature_vectors[document_shuffle_indices]

        with open(filepath, 'w') as ofile:
            for score, qid, feature_vector in izip(relevance_scores, query_ids, feature_vectors):
                ofile.write('%d' % score)
                ofile.write(' qid:%d' % qid)
                for feature in izip(self.feature_indices, feature_vector):
                    output = ' %d:%.12f' % feature
                    ofile.write(output.rstrip('0').rstrip('.'))
                ofile.write('\n')


    @classmethod
    def load(cls, filepath, mmap=None):
        '''
        Load the previously saved Queries object from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a Queries object will be loaded.

        mmap: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}, optional (default is None)
            If not None, then memory-map the feature vectors, using
            the given mode (see `numpy.memmap` for a details).
        '''
        logger.info('Loading queries from %s.' % filepath)
        queries = unpickle(filepath)
        queries.feature_vectors = np.load(filepath + '.feature_vectors.npy', mmap_mode=mmap)
        # Recover the query indices pointer array from the relevance strides array.
        setattr(queries, 'query_indptr', np.empty(queries.query_relevance_strides.shape[0] + 1, dtype=np.intc))
        queries.query_indptr[0]  = 0
        queries.query_indptr[1:] = queries.query_relevance_strides[:,0]
        # Recover the relevance scores from the relevance strides array.
        setattr(queries, 'relevance_scores', np.empty(queries.feature_vectors.shape[0], dtype=np.intc))
        iQD = 0
        for iQ in range(queries.n_queries):
            for score in range(queries.max_score, -1, -1):
                while iQD < queries.query_relevance_strides[iQ, score]:
                    queries.relevance_scores[iQD] = score
                    iQD += 1
        logger.info('Loaded %d queries with %d documents in total.' % (queries.query_count(), queries.document_count()))
        return queries


    def save(self, filepath):
        '''
        Save this Queries object into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        np.save(filepath + '.feature_vectors.npy', np.ascontiguousarray(self.feature_vectors))
        removed_attributes = {}
        # Delete all numpy arrays that can be restored from the 2 arrays above.
        for attribute in ('query_indptr', 'relevance_scores', 'feature_vectors'):
            removed_attributes[attribute] = getattr(self, attribute)
            delattr(self, attribute)
        # Pickle...
        pickle(self, filepath)
        # ... and restore the object's properties.
        for attribute, value in removed_attributes.iteritems():
            setattr(self, attribute, value)


    def __getitem__(self, i):
        if i < 0 or i > self.query_count():
            raise IndexError()
        s = self.query_indptr[i]
        e = self.query_indptr[i + 1]
        return Query(qid, self.relevance_scores[s:e], self.feature_vectors[s:e,:], base=self)


    def get_query(self, qid):
        '''
        Return the query with the given id.
        '''
        i = np.where(self.query_ids == qid)[0]
        if i is None:
            raise KeyError('no query exist with the specified id: %d' % qid)
        i = i[0]
        s = self.query_indptr[i]
        e = self.query_indptr[i + 1]
        return Query(qid, self.relevance_scores[s:e], self.feature_vectors[s:e,:], base=self)


    def document_count(self, qid=None):
        '''
        Return the number of documents for the query. If qid is None
        than the total number of "documents" is returned.
        '''
        if qid is None:
            return self.n_feature_vectors
        else:
            return self.query_indptr[qid + 1] - self.query_indptr[qid]


    def get_feature_vectors(self, qid):
        '''
        Return the feature vectors of the documents for the specified query.
        '''
        return self.feature_vectors[self.query_indptr[qid]:self.query_indptr[qid + 1]]


    def query_count(self):
        '''
        Return the number of queries in this Queries.
        '''
        return self.n_queries


    def highest_relevance(self):
        '''
        Return the maximum relevance score of a document.
        '''
        return self.max_score


    def longest_document_list(self):
        '''
        Return the maximum number of documents query can have.
        '''
        return np.diff(self.query_indptr).max()


def train_test_split(queries, train_size=None, test_size=0.2):
    '''
    Split the specified set of queries into training and test sets.
    The portion of queries that ends in the training or test set
    is determined by train_size and test_size parameters, respectively.
    The train_size parameter takes precedense (if specified)

    Parameters:
    -----------
    queries: Queries object
        The set of queries that should be partitioned to a training and test set.

    train_size: int or float, optional (default is None)
        If float, denotes the portion of (randomly chosen) queries that will
        become part of the training set. If int, the precise number of queries
        will be put into the training set. The complement will make the test set.

    test_size: int or float, optional (default is 0.2)
        If float, denotes the portion of (randomly chosen) queriess that will
        become part of the test set. If int, the precise number of samples
        will be put into the test set. The complement will make the training set.
    '''
    n_queries = queries.query_indptr.size - 1

    if train_size is not None:
        if isinstance(train_size, float):
            if train_size >= 1.0 or train_size <= 0.0:
                raise ValueError('the value of train_size must be in [0.0; 1.0] range')
            train_size = int(train_size * n_queries)
        elif train_size >= n_queries:
            raise ValueError('the specified train_size (%d) must be less than the number of queries'
                             ' queries (%d)' % (train_size, n_queries))
        elif train_size < 1:
            raise ValueError('the train_size must be at least 1 (%d was given)' % train_size)
        test_size = n_queries - train_size
    elif test_size is not None:
        if isinstance(test_size, float):
            if test_size >= 1.0 or test_size <= 0.0:
                raise ValueError('the value of test_size must be in [0.0; 1.0] range')
            test_size = int(test_size * n_queries)
        elif test_size >= n_queries:
            raise ValueError('the specified test_size (%d) must be less than the number of queries'
                             ' queries (%d)' % (test_size, n_queries))
        elif test_size < 1:
            raise ValueError('the test_size must be at least 1 (%d was given)' % test_size)
        train_size = n_queries - test_size
    else:
        raise ValueError('train_size and test_size cannot be both None!')

    test_queries_indices = np.arange(n_queries, dtype=np.intc)
    np.random.shuffle(test_queries_indices)

    train_queries_indices = test_queries_indices[test_size:]
    test_queries_indices  = test_queries_indices[:test_size]

    n_query_documents = np.diff(queries.query_indptr)

    test_query_indptr = np.concatenate([[0], n_query_documents[test_queries_indices]])
    train_query_indptr = np.concatenate([[0], n_query_documents[train_queries_indices]])

    np.cumsum(test_query_indptr, out=test_query_indptr)
    np.cumsum(train_query_indptr, out=train_query_indptr)

    assert test_query_indptr[-1] + train_query_indptr[-1] == queries.feature_vectors.shape[0]

    test_feature_vectors = np.empty((test_query_indptr[-1], queries.feature_vectors.shape[1]), dtype=queries.feature_vectors.dtype)
    train_feature_vectors = np.empty((train_query_indptr[-1], queries.feature_vectors.shape[1]), dtype=queries.feature_vectors.dtype)

    test_relevance_scores = np.empty(test_query_indptr[-1], dtype=queries.relevance_scores.dtype)
    train_relevance_scores = np.empty(train_query_indptr[-1], dtype=queries.relevance_scores.dtype)

    for i in range(len(test_query_indptr) - 1):
        test_feature_vectors[test_query_indptr[i]:test_query_indptr[i + 1]] = queries.feature_vectors[queries.query_indptr[test_queries_indices[i]]:queries.query_indptr[test_queries_indices[i] + 1]]
        test_relevance_scores[test_query_indptr[i]:test_query_indptr[i + 1]] = queries.relevance_scores[queries.query_indptr[test_queries_indices[i]]:queries.query_indptr[test_queries_indices[i] + 1]]

    for i in range(len(train_query_indptr) - 1):
        train_feature_vectors[train_query_indptr[i]:train_query_indptr[i + 1]] = queries.feature_vectors[queries.query_indptr[train_queries_indices[i]]:queries.query_indptr[train_queries_indices[i] + 1]]
        train_relevance_scores[train_query_indptr[i]:train_query_indptr[i + 1]] = queries.relevance_scores[queries.query_indptr[train_queries_indices[i]]:queries.query_indptr[train_queries_indices[i] + 1]]

    feature_indices = None
    if queries.feature_indices is not None:
        feature_indices = queries.feature_indices

    train_query_ids = None
    test_query_ids = None

    if queries.query_ids is not None:
        train_query_ids = queries.query_ids[train_queries_indices].copy()
        test_query_ids = queries.query_ids[test_queries_indices].copy()
        
    test_queries = Queries(test_feature_vectors, test_relevance_scores, test_query_indptr, queries.max_score, True, query_ids=test_query_ids, feature_indices=feature_indices)
    train_queries = Queries(train_feature_vectors, train_relevance_scores, train_query_indptr, queries.max_score, True, query_ids=train_query_ids, feature_indices=feature_indices)

    return train_queries, test_queries


def shuffle_split_queries(queries, n_folds=5):
    query_document_count  = np.diff(queries.query_indptr)

    if np.any(query_document_count < n_folds):
        raise ValueError('queries contain a document with less documents'\
                         ' than the wanted number of folds')

    # Magic that makes the whole thing work (hopefully in every case!).
    fold_document_indices = [[np.array([], dtype=np.intp)] * n_folds]
    fold_document_counts = []

    for qid, n_documents in enumerate(query_document_count):
        fold_document_indices.append(np.array_split(queries.query_indptr[qid] + np.random.permutation(n_documents), n_folds))
        fold_document_counts.append([document_indices.shape[0] for document_indices in fold_document_indices[-1]])

    # Using numpy arrays in Fortran-contiguous order for fancy and efficient column indexing.
    fold_document_indices = np.array(fold_document_indices, dtype=object, order='F')
    fold_document_counts = np.array(fold_document_counts, dtype=np.intc, order='F')

    # This will result in a list of arrays (one per fold) holding indices of documents for each query.
    fold_document_indices = [np.concatenate(fold_document_indices[:, i]) for i in range(n_folds)]

    # This will result in a list of array (one per fold) holding number of documents per query.
    fold_document_counts = [fold_document_counts[:, i] for i in range(n_folds)]

    cross_validation_queries = []

    for valid_fold in range(n_folds):
        valid_fold_document_indices = fold_document_indices[valid_fold]
        valid_fold_document_counts = fold_document_counts[valid_fold]

        valid_feature_vectors = queries.feature_vectors[valid_fold_document_indices, :]
        valid_relevance_scores = queries.relevance_scores[valid_fold_document_indices]
        valid_query_indptr = np.r_[0, valid_fold_document_counts].cumsum()

        valid_queries = Queries(valid_feature_vectors, valid_relevance_scores, valid_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

        # Make a shallow copy of the lists ...
        train_folds_document_indices = list(fold_document_indices)
        train_folds_document_counts = list(fold_document_counts)
        # ... then remove the valid fold ...
        del train_folds_document_indices[valid_fold]
        del train_folds_document_counts[valid_fold]
        # ... and finally concatenate them together
        train_folds_document_indices = np.concatenate(train_folds_document_indices)
        train_folds_document_counts = sum(train_folds_document_counts)

        train_feature_vectors = queries.feature_vectors[train_folds_document_indices, :]
        train_relevance_scores = queries.relevance_scores[train_folds_document_indices]
        train_query_indptr = np.r_[0, train_folds_document_counts].cumsum()
        
        train_queries = Queries(train_feature_vectors, train_relevance_scores, train_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

        cross_validation_queries.append((train_queries, valid_queries))

    return cross_validation_queries


def concatenate(queries):
    '''
    Concatenates the given list of queries (rankpy.queries.Queries) into a single
    queries Queries object.
    '''
    feature_vectors = np.concatenate([q.feature_vectors for q in queries])
    relevance_scores = np.concatenate([q.relevance_scores for q in queries])
    query_indptr = np.concatenate([np.diff(q.query_indptr) for q in queries]).cumsum()
    query_indptr = np.r_[0, query_indptr]
    max_score = max([q.max_score for q in queries])
    query_ids = np.concatenate([q.query_ids for q in queries])

    assert len(query_ids) == len(np.unique(query_ids)), 'some of the queries is in more than one collection'

    try:
        feature_indices = np.concatenate([q.feature_indices.reshape(1, -1) for q in queries]) - queries[0].feature_indices
    except:
        raise ValueError('feature indices of some queries does not correspond')

    assert not np.any(feature_indices), 'feature indices of some queries does not correspond'

    feature_indices = queries[0].feature_indices
    
    return Queries(feature_vectors, relevance_scores, query_indptr, max_score, False, query_ids, feature_indices)
