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

from ._metrics import DiscountedCumulativeGain as DCG
from ._utils import relevance_argsort_v1


class DiscountedCumulativeGain(object):
    ''' 
    Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of DCG metric
    for faster computation. Optionally, you can specify `max_relevance` and
    `max_documents`, but they should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into Segmentation Fault.

    Arguments:
    ----------
    cutoff: int, optional (default is -1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance: int, optional (default is 4)
        The maximum relevance score a document can have. This must be
        set for caching purposes. If the evaluated document list contain
        a document with higher relevance than the number specified,
        IndexError will be raised.

    max_documents: int, optional (default is 8192):
        The maximum number of documents a query can be associated with. This
        must be set for caching purposes. If the evaluated document list is
        bigger than the specified number, IndexError will be raised.

    queries: list of rankpy.queries.Queries
        The collections of queries that are known to be evaluated by this metric.
        These are used to compute `max_relevance` and `max_documents`.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None, queries=None):
        # Get the maximum relevance score and maximum number of documents
        # per a query from the specified set(s) of queries...
        if queries is not None:
            max_relevance = max([qs.highest_relevance() for qs in queries])
            max_documents = max([qs.longest_document_list() for qs in queries])
        else:
            # or use the parameters given. None values indicate that explicit
            # values were not given which may lead to unexpected results and
            # to runtime errors, hence a user warnings are issued.
            if max_relevance is None:
                max_relevance = 8
                warn('Maximum relevance label was not explicitly specified ' \
                     '(using default value 8). This should be avoided in order ' \
                     'not to encounter runtime error (SegFault)!')

            if max_documents is None:
                max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified ' \
                     '(using default value 8192). This should be avoided in order not to ' \
                     'encounter runtime error (SegFault)!')

        # Create the metric cython backend.
        self.metric_ = DCG(cutoff, max_relevance, max_documents)


    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scale=None):
        ''' 
        Evaluate the DCG metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale: float, optional (default is None)
            Ignored.
        '''
        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, 1.0)
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            return self.metric_.evaluate_ranking(ranking, labels, 1.0)


    def evaluate_queries(self, queries, scores, scale=None, out=None):
        ''' 
        Evaluate the DCG metric on the specified set of queries (`queries`). The documents
        are sorted by corresponding ranking scores (`scores`) and the metric is then
        computed as the average of the metric values evaluated on each query document
        list. The ties in ranking are broken probabilistically.
        
        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The set of queries for which the metric will be computed.

        scores: array, shape=(n_documents,)
            The ranking scores for each document in the queries.

        scale: array, shape=(n_queries,), or None
            Ignored.

        out: array, shape=(n_documents,), or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        if queries.document_count() != scores.shape[0]:
            raise ValueError('number of documents != number of scores (%d, %d)' \
                             % (queries.document_count(), scores.shape[0]))

        if out is not None and queries.query_count() != out.shape[0]:
            raise ValueError('number of queries != size of output array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        return self.metric_.evaluate_queries(queries.query_indptr, queries.relevance_scores, scores, None, out)


    def compute_delta(self, i, offset, document_ranks, relevance_scores, scale=None, out=None):
        ''' 
        Compute the change in the DCG metric after swapping document 'i' with
        each document in the document list starting at 'offset'.

        The relevance and rank of the document 'i' is 'relevance_scores[i]' and
        'document_ranks[i]', respectively.

        Similarly, 'relevance_scores[j]' and 'document_ranks[j]' for each j starting
        from 'offset' and ending at the end of the list denote the relevance score
        and rank of the document that will be swapped with document 'i'.

        Parameters:
        -----------
        i: int
            The index (zero-based) of the document that will appear in every pair
             of documents that will be swapped.

        offset: int
            The offset pointer to the start of the documents that will be swapped.

        document_ranks: array
            The ranks of the documents.

        relevance_scores: array
            The relevance scores of the documents.

        out: array, optional (default is None)
            The output array. The array size is expected to be at least equal to
            the number of documents shorter by the ofset.

        scale: float or None, optional (default is None)
            Ignored.
        '''
        n_documents = len(document_ranks)

        if out is None:
            out = np.empty(n_documents - offset, dtype=np.float64)

        if out.shape[0] < n_documents - offset:
            raise ValueError('output array is too small (%d < %d)' \
                             % (out.shape[0], n_documents - offset))

        if document_ranks.shape[0] != relevance_scores.shape[0]:
            raise ValueError('document ranks size != relevance scores (%d != %d)' \
                              % (document_ranks.shape[0], relevance_scores.shape[0]))

        self.metric_.delta(i, offset, document_ranks, relevance_scores, 1.0, out)

        return out


    def compute_scale(self, queries, relevance_scores=None):
        ''' 
        Since DCG is not normalized, return None.
        '''
        return None


    def __str__(self):
        ''' 
        Return the textual description of the metric.
        '''
        return 'DCG' if self.metric_.cutoff < 0 else 'DCG@%d' % self.metric_.cutoff


class NormalizedDiscountedCumulativeGain(object):
    ''' 
    Normalized Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of NDCG metric
    for faster computation.

    Optionally, you can specify `max_relevance` and `max_documents`, but they
    should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into Segmentation Fault.

    Arguments:
    ----------
    cutoff: int, optional (default is -1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance: int, optional (default is 4)
        The maximum relevance score a document can have. This must be
        set for caching purposes. If the evaluated document list contain
        a document with higher relevance than the number specified,
        IndexError will be raised.

    max_documents: int, optional (default is 8192):
        The maximum number of documents a query can be associated with. This
        must be set for caching purposes. If the evaluated document list is
        bigger than the specified number, IndexError will be raised.

    queries: list of rankpy.queries.Queries
        The collections of queries that are known to be evaluated by this metric.
        These are used to compute `max_relevance` and `max_documents`.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None, queries=None):
        # Get the maximum relevance score and maximum number of documents
        # per a query from the specified set(s) of queries...
        if queries is not None:
            max_relevance = max([qs.highest_relevance() for qs in queries])
            max_documents = max([qs.longest_document_list() for qs in queries])
        else:
            # or use the parameters given. None values indicate that explicit
            # values were not given which may lead to unexpected results and
            # to runtime errors, hence a user warnings are issued.
            if max_relevance is None:
                max_relevance = 8
                warn('Maximum relevance label was not explicitly specified ' \
                     '(using default value 8). This should be avoided in order ' \
                     'not to encounter runtime error (SegFault)!')

            if max_documents is None:
                max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified ' \
                     '(using default value 8192). This should be avoided in order not to ' \
                     'encounter runtime error (SegFault)!')

        # Create the metric cython backend.
        self.metric_ = DCG(cutoff, max_relevance, max_documents)


    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scale=None):
        ''' 
        Evaluate NDCG metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scale: float, optional (default is None)
            The ideal DCG value on the given documents. If None is given
            it will be computed from the document relevance scores.
        '''
        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scale or self.metric_.evaluate(np.ascontiguousarray(np.sort(ranked_labels)[::-1]), 1.0))
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            return self.metric_.evaluate_ranking(ranking, labels, scale or self.metric_.evaluate(np.ascontiguousarray(np.sort(labels)[::-1]), 1.0))


    def evaluate_queries(self, queries, scores, scale=None, out=None):
        ''' 
        Evaluate the NDCG metric on the specified set of queries (`queries`). The documents
        are sorted by corresponding ranking scores (`scores`) and the metric is then
        computed as the average of the metric values evaluated on each query document
        list.
        
        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The set of queries for which the metric will be computed.

        scores: array, shape=(n_documents,)
            The ranking scores for each document in the queries.

        scale: array, shape=(n_queries,) or None, optional (default is None)
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        out: array, shape=(n_documents,), or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        if queries.document_count() != scores.shape[0]:
            raise ValueError('number of documents != number of scores (%d, %d)' \
                             % (queries.document_count(), scores.shape[0]))
        if scale is None:
            scale = np.empty(queries.query_count(), dtype=np.float64)
            self.metric_.evaluate_queries_ideal(queries.query_indptr, queries.relevance_scores, scale)

        if queries.query_count() != scale.shape[0]:
            raise ValueError('number of queries != number of scaling factors (%d != %d)' \
                             % (queries.query_count(), scale.shape[0]))

        if out is not None and queries.query_count() != out.shape[0]:
            raise ValueError('number of queries != size of output array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        return self.metric_.evaluate_queries(queries.query_indptr, queries.relevance_scores, scores, scale, out)


    def compute_delta(self, i, offset, document_ranks, relevance_scores, scale=None, out=None):
        ''' 
        Compute the change in the NDCG metric after swapping document 'i' with
        each document in the document list starting at 'offset'.

        The relevance and rank of the document 'i' is 'relevance_scores[i]' and
        'document_ranks[i]', respectively.

        Similarly, 'relevance_scores[j]' and 'document_ranks[j]' for each j starting
        from 'offset' and ending at the end of the list denote the relevance score
        and rank of the document that will be swapped with document 'i'.

        Parameters:
        -----------
        i: int
            The index (zero-based) of the document that will appear in every pair
             of documents that will be swapped.

        offset: int
            The offset pointer to the start of the documents that will be swapped.

        document_ranks: array
            The ranks of the documents.

        relevance_scores: array
            The relevance scores of the documents.

        scale: float or None, optional (default is None)
            The ideal DCG value for the query the documents are associated with.
            If None is given, the scale will be computed from the relevance scores.

        out: array, optional (default is None)
            The output array. The array size is expected to be at least equal to
            the number of documents shorter by the ofset.
        '''
        n_documents = len(document_ranks)

        if out is None:
            out = np.empty(n_documents - offset, dtype=np.float64)

        if out.shape[0] < n_documents - offset:
            raise ValueError('output array is too small (%d < %d)' \
                             % (out.shape[0], n_documents - offset))

        if document_ranks.shape[0] != relevance_scores.shape[0]:
            raise ValueError('document ranks size != relevance scores (%d != %d)' \
                             % (document_ranks.shape[0], relevance_scores.shape[0]))

        if scale is None:
            scale = self.metric_.evaluate(np.ascontiguousarray(np.sort(relevance_scores)[::-1]), 1.0)

        self.metric_.delta(i, offset, document_ranks, relevance_scores, scale, out)

        return out


    def compute_scale(self, queries, relevance_scores=None):
        ''' 
        Return the ideal DCG value for each query. Optionally, external
        relevance assessments can be used instead of the relevances
        present in the queries.
        '''
        ideal_values = np.empty(queries.query_count(), dtype=np.float64)
        if relevance_scores is not None:
            if queries.document_count() != relevance_scores.shape[0]:
                raise ValueError('number of documents and relevance scores do not match')
            # Need to sort the relevance labels first.
            indices = np.empty(relevance_scores.shape[0], dtype=np.intc)
            relevance_argsort_v1(relevance_scores, indices, relevance_scores.shape[0])
            # Creates a copy.
            relevance_scores = relevance_scores[indices]
        else:
            # Assuming these are sorted.
            relevance_scores = queries.relevance_scores

        self.metric_.evaluate_queries_ideal(queries.query_indptr, relevance_scores, ideal_values)
        return ideal_values


    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'NDCG' if self.metric_.cutoff < 0 else 'NDCG@%d' % self.metric_.cutoff
