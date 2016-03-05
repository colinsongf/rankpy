# This file is part of RankPy.
#
# RankPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RankPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from warnings import warn

from ._metrics import MeanPrecision as MPN
from ._metrics import WinnerTakesAll as WTA
from ._metrics import ClickthroughRate as CTR
from ._metrics import MeanReciprocalRank as MRR
from ._metrics import MeanAveragePrecision as MAP
from ._metrics import ExpectedReciprocalRank as ERR
from ._metrics import DiscountedCumulativeGain as DCG

from sklearn.utils import check_random_state


class MeanPrecision(object):
    '''
    Mean Precision metric.

    Parameters
    ----------
    cutoff: int, optional (default=-1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        Not used.

    max_documents : int, optional (default=1024):
        Not used.

    queries : list of Queries instances
        Not used.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, random_state=None):
        self.cutoff = cutoff
        self.max_relevance = 0
        self.max_documents = 0
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = MPN(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return MPN(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return MeanPrecision(cutoff=self.cutoff,
                             max_relevance=self.max_relevance,
                             max_documents=self.max_documents,
                             normalized=self.normalized,
                             random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'MP')
        else:
            return (normalized + 'MP@%d') % self.metric_.cutoff


class MeanAveragePrecision(object):
    '''
    Mean Average Precision metric.

    Parameters
    ----------
    cutoff: int, optional (default=-1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        Not used.

    max_documents : int, optional (default=1024):
        Not used.

    queries : list of Queries instances
        Not used.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, random_state=None):
        # Get the maximum relevance score and maximum number of documents
        # per query from the specified set(s) of queries...
        if queries is not None:
            max_relevance_ = max([qs.max_relevance_score() for qs in queries])
            max_documents_ = max([qs.max_document_count() for qs in queries])

            if max_relevance is None:
                max_relevance = max_relevance_
            elif max_relevance < max_relevance_:
                raise ValueError('the specified maximum relevance score is '
                                 'smaller than the maximum found in the given '
                                 'queries: %d < %d'
                                 % (max_relevance, max_relevance_))

            if max_documents is None:
                max_documents = max_documents_
            elif max_documents < max_documents_:
                raise ValueError('the specified maximum document list length '
                                 'is smaller than the maximum found in the '
                                 'given queries: %d < %d'
                                 % (max_documents, max_documents_))

        # ... or use the parameters given. None values indicate that explicit
        # values were not given which may lead to unexpected results and
        # to runtime errors, hence a user warnings are issued.
        if max_relevance is None:
            max_relevance = 4
            warn('Maximum relevance label was not explicitly specified '
                 '(using default value 4). This should be avoided in '
                 'order not to encounter runtime error!')

        if max_documents is None:
            max_documents = 1024
            warn('Maximum number of documents per query was not '
                 'explicitly specified (using default value 1024). '
                 'This should be avoided in order not to encounter '
                 'runtime error!')

        self.cutoff = cutoff
        self.max_relevance = max_relevance
        self.max_documents = max_documents
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = MAP(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return MAP(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return MeanAveragePrecision(cutoff=self.cutoff,
                                    max_relevance=self.max_relevance,
                                    max_documents=self.max_documents,
                                    normalized=self.normalized,
                                    random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'MAP')
        else:
            return (normalized + 'MAP@%d') % self.metric_.cutoff


class MeanReciprocalRank(object):
    '''
    Mean Reciprocal Rank metric.

    Parameters
    ----------
    cutoff: int, optional (default=-1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        Not used.

    max_documents : int, optional (default=1024):
        Not used.

    queries : list of Queries instances
        Not used.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, random_state=None):
        self.cutoff = cutoff
        self.max_relevance = 0
        self.max_documents = 0
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = MRR(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return MRR(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return MeanReciprocalRank(cutoff=self.cutoff,
                                  max_relevance=self.max_relevance,
                                  max_documents=self.max_documents,
                                  normalized=self.normalized,
                                  random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'MRR')
        else:
            return (normalized + 'MRR@%d') % self.metric_.cutoff


class WinnerTakesAll(object):
    '''
    Generalized Winner Takes All metric. The metric value is 1 if the most
    relevant document of the query is at top `k` positions, where `k` is
    the cutoff.

    Parameters
    ----------
    cutoff : int, optional (default=1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        Not used.

    max_documents : int, optional (default=1024):
        Not used.

    queries : list of Queries instances
        Not used.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, random_state=None):
        self.cutoff = cutoff
        self.max_relevance = 0
        self.max_documents = 0
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = WTA(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return WTA(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return WinnerTakesAll(cutoff=self.cutoff,
                              max_relevance=self.max_relevance,
                              max_documents=self.max_documents,
                              normalized=self.normalized,
                              random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'WTA')
        else:
            return (normalized + 'WTA@%d') % self.metric_.cutoff


class DiscountedCumulativeGain(object):
    '''
    Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of DCG metric
    for faster computation. Optionally, you can specify `max_relevance` and
    `max_documents`, but they should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine :).
      2) You will run into "segmentation fault" :(.

    Parameters
    ----------
    cutoff: int, optional (default=-1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        The maximum relevance score a document can have. This must be
        set for caching purposes. If the evaluated document list contain
        a document with higher relevance than the number specified,
        IndexError will be raised.

    max_documents : int, optional (default=1024):
        The maximum number of documents a query can be associated with. This
        must be set for caching purposes. If the evaluated document list is
        bigger than the specified number, IndexError will be raised.

    queries : list of Queries instances
        The collections of queries that are known to be evaluated by this
        metric instance. These are used to compute `max_relevance` and
        `max_documents`.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, random_state=None):
        # Get the maximum relevance score and maximum number of documents
        # per query from the specified set(s) of queries...
        if queries is not None:
            max_relevance_ = max([qs.max_relevance_score() for qs in queries])
            max_documents_ = max([qs.max_document_count() for qs in queries])

            if max_relevance is None:
                max_relevance = max_relevance_
            elif max_relevance < max_relevance_:
                raise ValueError('the specified maximum relevance score is '
                                 'smaller than the maximum found in the given '
                                 'queries: %d < %d'
                                 % (max_relevance, max_relevance_))

            if max_documents is None:
                max_documents = max_documents_
            elif max_documents < max_documents_:
                raise ValueError('the specified maximum document list length '
                                 'is smaller than the maximum found in the '
                                 'given queries: %d < %d'
                                 % (max_documents, max_documents_))

        # ... or use the parameters given. None values indicate that explicit
        # values were not given which may lead to unexpected results and
        # to runtime errors, hence a user warnings are issued.
        if max_relevance is None:
            max_relevance = 4
            warn('Maximum relevance label was not explicitly specified '
                 '(using default value 4). This should be avoided in '
                 'order not to encounter runtime error!')

        if max_documents is None:
            max_documents = 1024
            warn('Maximum number of documents per query was not '
                 'explicitly specified (using default value 1024). '
                 'This should be avoided in order not to encounter '
                 'runtime error!')

        self.cutoff = cutoff
        self.max_relevance = max_relevance
        self.max_documents = max_documents
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = DCG(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return DCG(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return DiscountedCumulativeGain(cutoff=self.cutoff,
                                        max_relevance=self.max_relevance,
                                        max_documents=self.max_documents,
                                        normalized=self.normalized,
                                        random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'DCG')
        else:
            return (normalized + 'DCG@%d') % self.metric_.cutoff


class ExpectedReciprocalRank(object):
    '''
    Normalized Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for internal arrays used by ERR metric
    for faster computation. Optionally, you can specify `max_relevance` and
    `max_documents`, but they should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine :).
      2) You will run into "segmentation fault" :(.

    Parameters
    ----------
    cutoff: int, optional (default=-1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        The maximum relevance score a document can have. This must be
        set for caching purposes. If the evaluated document list contain
        a document with higher relevance than the number specified,
        IndexError will be raised.

    max_documents : int, optional (default=1024):
        The maximum number of documents a query can be associated with. This
        must be set for caching purposes. If the evaluated document list is
        bigger than the specified number, IndexError will be raised.

    queries : list of Queries instances
        The collections of queries that are known to be evaluated by this
        metric instance. These are used to compute `max_relevance` and
        `max_documents`.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, random_state=None):
        # Get the maximum relevance score and maximum number of documents
        # per query from the specified set(s) of queries...
        if queries is not None:
            max_relevance_ = max([qs.max_relevance_score() for qs in queries])
            max_documents_ = max([qs.max_document_count() for qs in queries])

            if max_relevance is None:
                max_relevance = max_relevance_
            elif max_relevance < max_relevance_:
                raise ValueError('the specified maximum relevance score is '
                                 'smaller than the maximum found in the given '
                                 'queries: %d < %d'
                                 % (max_relevance, max_relevance_))

            if max_documents is None:
                max_documents = max_documents_
            elif max_documents < max_documents_:
                raise ValueError('the specified maximum document list length '
                                 'is smaller than the maximum found in the '
                                 'given queries: %d < %d'
                                 % (max_documents, max_documents_))

        # ... or use the parameters given. None values indicate that explicit
        # values were not given which may lead to unexpected results and
        # to runtime errors, hence a user warnings are issued.
        if max_relevance is None:
            max_relevance = 4
            warn('Maximum relevance label was not explicitly specified '
                 '(using default value 4). This should be avoided in '
                 'order not to encounter runtime error!')

        if max_documents is None:
            max_documents = 1024
            warn('Maximum number of documents per query was not '
                 'explicitly specified (using default value 1024). '
                 'This should be avoided in order not to encounter '
                 'runtime error!')

        self.cutoff = cutoff
        self.max_relevance = max_relevance
        self.max_documents = max_documents
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = ERR(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return ERR(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return ExpectedReciprocalRank(cutoff=self.cutoff,
                                      max_relevance=self.max_relevance,
                                      max_documents=self.max_documents,
                                      normalized=self.normalized,
                                      random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'ERR')
        else:
            return (normalized + 'ERR@%d') % self.metric_.cutoff


class ClickthroughRate(object):
    '''
    Clickthrough Rate (CTR) metric.

    Parameters
    ----------
    cutoff: int, optional (default=-1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance : int, optional (default=4)
        Not used.

    max_documents : int, optional (default=1024):
        Not used.

    queries : list of Queries instances
        Not used.

    random_state : int or RandomState instance
        Random number generator or a seed for initialization of a generator
        which is used for breaking ties in sorting documents with equal
        ranking scores.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, normalized=False, click_proba=None,
                 stop_proba=None, abandon_proba=0.0, relative=False,
                 sample=False, n_impressions=1000, random_state=None):
        self.cutoff = cutoff
        self.max_relevance = 0
        self.max_documents = 0
        self.normalized = normalized
        self.random_state = check_random_state(random_state)

        if click_proba is None:
            raise ValueError('click probabilities cannot be None')

        if stop_proba is None:
            raise ValueError('stop probabilities cannot be None')

        # Create the metric cython backend.
        self.metric_ = CTR(self.cutoff, self.max_relevance, self.max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

        # Initialize the underlying click model.
        self.metric_.initialize_click_model(click_proba,
                                            stop_proba,
                                            abandon_proba,
                                            relative=relative,
                                            sample=sample,
                                            n_impressions=n_impressions)

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            metric_ = CTR(self.cutoff, self.max_relevance, self.max_documents,
                          self.random_state.randint(1, np.iinfo('i').max))

            metric_.initialize_click_model(self.metric_.click_model.click_proba,
                                           self.metric_.click_model.stop_proba,
                                           self.metric_.click_model.abandon_proba,
                                           relative=self.metric_.relative,
                                           sample=self.metric_.sample,
                                           n_impressions=self.metric_.n_impressions)
            return metric_
        else:
            return self.metric_

    def copy(self, reseed=True):
        '''
        Returns a copy of this object. If `reseed` is True, the internal
        random number generator of the new metric object will be seeded
        with a randomly generated number.
        '''
        if reseed:
            random_state = self.random_state.randint(1, np.iinfo('i').max)
        else:
            random_state = self.random_state

        return ClickthroughRate(cutoff=self.cutoff,
                                max_relevance=self.max_relevance,
                                max_documents=self.max_documents,
                                normalized=self.normalized,
                                click_proba=self.metric_.click_proba,
                                stop_proba=self.metric_.stop_proba,
                                abandon_proba=self.metric_.abandon_proba,
                                relative=self.metric_.relative,
                                sample=self.metric_.sample,
                                n_impressions=self.metric_.n_impressions,
                                random_state=random_state)

    def evaluate(self, ranking=None, labels=None, ranked_labels=None,
                 scale=None, weight=1.0):
        '''
        Evaluate the metric on the specified ranking.

        The function input can be either ranked list of relevance labels
        (`ranked_labels`), or it can be in the form of ranked list of documents
        (`ranking`) and corresponding relevance scores (`labels`), from which
        the ranked document relevance labels are created.

        Parameters:
        -----------
        ranking : array of int, shape = [n_documents]
            Specify list of ranked documents.

        labels : array: shape = [n_documents]
            Specify relevance score for each document.

        ranked_labels : array, shape = [n_documents]
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale : float, optional (default=None)
            Optional argument for speeding up computation of normalized
            value of the metric for the given query. If None is passed
            and the metric is set to normalize, then the value will be
            computed from the document relevance scores, otherwise, it
            is be ignored.

        weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if self.normalized:
            if scale is None:
                if ranked_labels is not None:
                    scale = ranked_labels
                elif labels is not None:
                    scale = labels

                scale = self.metric_.evaluate(-np.sort(-scale), 1.0, 1.0)
        else:
            # Make sure the metric setting is respected.
            scale = 1.0

        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale, weight)

        if ranking is None:
            raise ValueError('missing list of documents (ranking)')

        if labels is None:
            raise ValueError('missing relevance labels')

        if ranking.shape[0] != labels.shape[0]:
            raise ValueError('the number of documents does not match '
                             'the number of relevance labels: %d != %d'
                             % (ranking.shape[0], labels.shape[0]))

        return self.metric_.evaluate_ranking(ranking, labels, scale, weight)

    def evaluate_queries(self, queries, scores, scales=None,
                         query_weights=None, document_weights=None,
                         out=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`).
        The documents are sorted by corresponding ranking scores (`scores`)
        and the metric is then computed as a (weighted) average of the metric
        values evaluated on each query document list.

        The ties in ranking scores are broken randomly.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the metric is evaluated.

        scores : array of floats, shape = [n_documents]
            The ranking scores for each document in the queries.

        scales : array of floats, shape = [n_queries], or None
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if queries.document_count() != len(scores):
            raise ValueError('the number of documents does not match '
                             'the number of ranking scores: %d != %d'
                             % (queries.document_count(), len(scores)))

        if query_weights is not None and len(queries) != len(query_weights):
            raise ValueError('the number of queries does not match '
                             'the number of weights: %d != %d'
                             % (len(queries), len(query_weights)))

        if out is not None and len(queries) > len(out):
            raise ValueError('the number of queries is larger than the size '
                             'of the output array: %d > %d'
                             % (len(queries), len(out)))

        if self.normalized:
            if scales is None:
                scales = self.compute_scaling(queries, query_weights=None,
                                              document_weights=document_weights)
            elif len(queries) != len(scales):
                raise ValueError('the number of queries does not match '
                                 'the size of the scales array: %d > %d'
                                 % (len(queries), len(scales)))
        else:
            # Make sure the metric setting is respected.
            scale = None

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        return self.metric_.evaluate_queries(queries.query_indptr,
                                             queries.relevance_scores,
                                             scores, scales, query_weights,
                                             document_weights,
                                             out)

    def compute_scaling(self, queries, query_weights=None,
                        document_weights=None, relevance_scores=None,
                        out=None):
        '''
        Returns the ideal metric value for each query.

        Parameters
        ----------
        queries : Queries instance
            The set of queries for which the scaling is computed.

        query_weights : array of floats, shape = [n_queries], or None
            The weight of each query for which the metric is evaluated.
            It is used just as an indicator of queries that should be
            ignored. The ignored queries have 0 weights.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used just as an indicator of
            documents that should be ignored. The ignored documents have
            0 weight.

        relevance_scores: array of ints or None (default=None)
            The relevance scores that should be used instead of the
            relevance scores inside queries. Note, this argument is
            experimental!

        out : array of floats, shape = [n_queries], or None
            An optional output array for the scale values.
        '''
        if out is None:
            out = np.empty(queries.query_count(), dtype='float64')

        if len(queries) > len(out):
            raise ValueError('the number of queries is greater than '
                             'the size of the output array: %d > %d'
                             % (len(queries), len(out)))

        out.fill(1.0)

        if not self.normalized:
            return out

        if query_weights is not None:
            if len(queries) != len(query_weights):
                raise ValueError('the number of queries does not match '
                                 'the number of weights: %d != %d'
                                 % (len(queries), len(query_weights)))
            else:
                query_weights = (query_weights != 0.).astype('float64')

        if (document_weights is not None and
            queries.document_count() != len(document_weights)):
            raise ValueError('the number of query documents does not match '
                             'the number of document weights: %d != %d'
                             % (queries.document_count(),
                                len(document_weights)))

        if relevance_scores is None:
            relevance_scores = queries.relevance_scores

        if queries.document_count() != len(relevance_scores):
            raise ValueError('the number of documents does not match '
                             'the number of relevance scores: %d != %d'
                             % (queries.document_count(),
                                len(relevance_scores)))

        ranking_scores = queries.relevance_scores.astype('float64')

        self.metric_.evaluate_queries(queries.query_indptr,
                                      queries.relevance_scores,
                                      ranking_scores, None, query_weights,
                                      document_weights,
                                      out)
        return out

    def compute_deltas(self, document_ranks, relevance_scores,
                       scale=None, out=None):
        '''
        Compute the changes in the metric caused by swapping pairs of
        documents `i` and `j` (`i` < `j`).

        The relevance and rank of the document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        document_ranks: array of ints, shape = [n_documents]
            The ranks of the documents.

        relevance_scores: array of ints, shape = [n_documents]
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The precomputed ideal metric value of the query.

        out: array of floats, shape = [n_documents, n_documents], or None,
             optional (default=None)
            An output array, which size is expected to be square matrix.
            Its upper triangular part will be filled with delta values.
        '''
        if (np.unique(document_ranks) != np.arange(len(document_ranks))).all():
            raise ValueError('rank array does not make a valid ranking')

        if len(document_ranks) != len(relevance_scores):
            raise ValueError('the number of ranks does not match the number '
                             'of relevance scores: %d != %d'
                              % (len(document_ranks), len(relevance_scores)))

        n_documents = len(document_ranks)

        if out is None:
            out = np.empty((n_documents, n_documents), dtype='float64')

        out.fill(0.0)

        if (getattr(out, 'dtype', None) != np.float64 or out.ndim != 2 or
            not out.flags['C_CONTIGUOUS']):
            raise ValueError('output array must be contiguous 1-d array '
                             'of doubles (np.float64)')

        if out.shape != (n_documents, n_documents):
            raise ValueError('output array has wrong shape: %r != %r'
                             % (out.shape, (n_documents, n_documents)))

        if self.normalized:
            if scale is None:
                scale = self.metric_.evaluate(-np.sort(-relevance_scores),
                                              1.0, 1.0)
        else:
            scale=1.0

        for i in range(n_documents - 1):
            self.metric_.delta(i, i + 1, document_ranks, relevance_scores,
                               None, n_documents, scale, out[i, (i + 1):])
        return out

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        normalized = 'n' if self.normalized else ''
        if self.metric_.cutoff < 0:
            return (normalized + 'CTR')
        else:
            return (normalized + 'CTR@%d') % self.metric_.cutoff


class MetricFactory(object):
    name2metric = {'WTA': WinnerTakesAll,
                   'MPN': MeanPrecision,
                   'CTR': ClickthroughRate,
                   'MRR': MeanReciprocalRank,
                   'MAP': MeanAveragePrecision,
                   'ERR': ExpectedReciprocalRank,
                   'DCG': DiscountedCumulativeGain }

    @staticmethod
    def __new__(cls, metric_name_kwargs, queries=None, max_relevance=None,
                max_documents=None, normalized=False, random_state=None):
        if isinstance(metric_name_kwargs, basestring):
            name = metric_name_kwargs
            kwargs = {}
        elif isinstance(metric_name_kwargs, (list, tuple)):
            name, kwargs = metric_name_kwargs
        else:
            return metric_name_kwargs

        if not isinstance(kwargs, dict):
            raise ValueError('valid 2-tuple metric specification must have '
                             'a dict in the second position')

        name, _, cutoff = name.partition('@')

        # Using shortcut specification for normalized metric.
        if name[0] == 'n':
            name = name[1:]
            normalized = True

        cutoff = int(cutoff) if len(cutoff) > 0 else -1

        try:
            return cls.name2metric[name](cutoff,
                                         queries=queries,
                                         max_relevance=max_relevance,
                                         max_documents=max_documents,
                                         normalized=normalized,
                                         random_state=random_state,
                                         **kwargs)
        except KeyError:
            raise ValueError('unknown metric: %s' % name)