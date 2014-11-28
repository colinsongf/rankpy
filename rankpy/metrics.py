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

import numpy as np

from warnings import warn

from .metrics_inner import compute_delta_dcg
from .metrics_inner import compute_dcg_metric
from .metrics_inner import compute_ndcg_metric
from .metrics_inner import compute_ideal_dcg_metric_per_query


class AbstractMetric(object):
    '''
    The base class for information retrieval kind of evaluation metric.

    Every evaluation metric should implement this class interface.

    Arguments:
    ----------
    cutoff: int, optional (default is -1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation. If 0 is given
        ValueError is raised.
    '''
    def __init__(self, cutoff=-1):
        self.cutoff = cutoff
        if cutoff == 0:
            raise ValueError('cutoff has to be positive integer'
                             ' or (-1) but 0 was given')


    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scale=None):
        '''
        Evaluate the metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of document ranking (`ranking`) and corresponding relevance scores (`labels`),
        from which the ranked document relevance labels are computed.

        This method has to be implemented by each metric object.

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
            If the metric value are scaled (e.g. its values need to be scaled
            by the ideal metric value), then it may be computationally more convenient
            to precalculate the normalization factors in advance (by using `compute_scale`
            method) and use its output values for this parameter. Specific metric 
            implementations can exploit this parameter to speed up their evaluation.
        '''
        raise NotImplementedError()


    def evaluate_queries(self, queries, scores, scale=None):
        '''
        Evaluate the metric on the specified set of queries (`queries`). The documents
        are sorted by corresponding ranking scores (`scores`) and the metric is then
        computed as the average of the metric values evaluated on each query document
        list.

        If the metric is scaled, `scale` argument can be used.
        
        This method has to be implemented by each metric object.

        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The set of queries for which the metric will be computed.

        scores: array, shape=(n_queries, )
            The ranking scores for each document in the queries.

        scale: array, shape=(n_queries,)
            The scale factor for each query, e.g. ideal metric value for DCG metric,
            which allows to compute NDCG metric in the end.
        '''
        raise NotImplementedError()


    def compute_delta(self, i, offset, document_ranks, relevance_scores, scale=None, out=None):
        '''
        Compute the change in the metric value of the ranked document list after swapping
        document `i` with each document in the document list starting at `offset`.

        The relevance and rank of the document `i` is `relevance_scores[i]` and
        `document_ranks[i]`, respectively.

        Similarly, `relevance_scores[j]` and `document_ranks[j]` for each j starting
        from `offset` and ending at the end of the list denote the relevance score
        and rank of the document that will be swapped with document `i`.

        This method has to be implemented by each metric object.

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

        scale: float, optional (default is None)
            The ideal metric value for the specified documents. If None is given
            the ideal metric value must be computed from the document ranks and
            corresponding relevance scores.

        out: array, optional (default is None)
            The output array. The array size is expected to be at least equal to
            the number of documents shorter by the ofset.
        '''
        raise NotImplementedError()


    def compute_scale(self, queries):
        '''
        Return the ideal metric value for each query in the specifed
        set of queries. These values can be used as `scale` parameter in
        `evaluation` methods to speed up computation of metrics, which
        need to be normalized.

        This method should be implemented by each evaluation metric
        that is normalized. Other should always return None
        '''
        raise NotImplementedError()


class DiscountedCumulativeGain(AbstractMetric):
    '''
    Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of DCG metric
    for faster computation. Optionally, you can specify `max_relevance` and
    `max_documents`, but they should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into IndexError or similar RuntimeError and you
         will be at least given a hint about what is going on.        

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
        super(DiscountedCumulativeGain, self).__init__(cutoff)
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
                max_relevance = 4
                warn('Maximum relevance label was not explicitly specified '\
                     '(using default value 4). This should be avoided in order '\
                     'not to encounter runtime error (IndexError)!')

            if max_documents is None:
                max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified '\
                     '(using default value 8192). This should be avoided in order not to '\
                     'encounter runtime error (IndexError)!')

        self.gain_cache = 2**np.arange(max_relevance + 1, dtype=np.float64) - 1
        self.discount_cache = np.log2(np.arange(2, max_documents + 2, dtype=np.float64))


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
        if ranked_labels is None:
            try:
                ranked_labels = labels.take(ranking)
            except:
                raise ValueError('cannot evaluate the ranking because some of the arguments:'\
                                 '`ranking` and/or `labels` were not specified')

        cutoff = ranked_labels.size if self.cutoff < 0 else self.cutoff
        cutoff = min(ranked_labels.size, cutoff)

        return (self.gain_cache.take(ranked_labels[:cutoff]) / self.discount_cache[:cutoff]).sum()


    def evaluate_queries(self, queries, scores, scale=None):
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
        '''
        return compute_dcg_metric(scores, queries.relevance_scores, queries.query_indptr,
                                  self.gain_cache, self.discount_cache, self.cutoff)


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
            The resulting deltas will be scaled by this number. If ideal DCG is given,
            for example, this will effectively result in computing delta for NDCG metric.
        '''
        n_documents = len(document_ranks)

        if out is None:
            out = np.empty(n_documents - offset, dtype=np.float64)

        if scale is None:
            scale = 1.0

        # To avoid returning the view (see below)
        original_out = out

        # To avoid unnecessary broadcasting beyond indices that matter.
        out = out[:(n_documents - offset)]

        # The metric cutoff rank.
        cutoff = n_documents if self.cutoff < 0 else self.cutoff

        compute_delta_dcg(self.gain_cache, self.discount_cache, cutoff, i, offset, document_ranks, relevance_scores, scale, out)

        np.absolute(out, out=out)

        return original_out


    def compute_scale(self, queries):
        '''
        Since DCG is not normalized, return None.
        '''
        return None


    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'DCG' if self.cutoff < 0 else 'DCG@%d' % self.cutoff


class NormalizedDiscountedCumulativeGain(DiscountedCumulativeGain):
    '''
    Normalized Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of NDCG metric
    for faster computation.

    Optionally, you can specify `max_relevance` and `max_documents`, but they
    should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into IndexError or similar RuntimeError and you
         will be at least given a hint about what is going on.

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
        super(NormalizedDiscountedCumulativeGain, self).__init__(cutoff, max_relevance, max_documents, queries)


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
        # Use the ideal DCG score (if given).
        ideal_dcg = scale
        if ranked_labels is not None:    
            dcg = super(NormalizedDiscountedCumulativeGain, self).evaluate(ranked_labels=ranked_labels)
            if ideal_dcg is None:
                ideal_dcg = super(NormalizedDiscountedCumulativeGain, self).evaluate(ranked_labels=np.sort(ranked_labels)[::-1])
        else:
            dcg = super(NormalizedDiscountedCumulativeGain, self).evaluate(ranking=ranking, labels=labels)
            if ideal_dcg is None:
                ideal_dcg = super(NormalizedDiscountedCumulativeGain, self).evaluate(ranked_labels=np.sort(labels)[::-1])

        return dcg if ideal_dcg == 0.0 else dcg / ideal_dcg


    def evaluate_queries(self, queries, scores, scale=None):
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
        '''
        return compute_ndcg_metric(scores, queries.relevance_scores, queries.query_indptr, self.gain_cache, self.discount_cache, self.cutoff, scale)


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
        # If the ideal DCG has not been given...
        if scale is None:
            # ... it needs to be calculated:
            scale = super(NormalizedDiscountedCumulativeGain, self).evaluate(ranked_labels=np.sort(relevance_scores)[::-1])
        return super(NormalizedDiscountedCumulativeGain, self).compute_delta(i, offset, document_ranks, relevance_scores, scale, out)


    def compute_scale(self, queries):
        '''
        Return the ideal DCG value for each query.
        '''
        return compute_ideal_dcg_metric_per_query(queries.relevance_scores, queries.query_indptr,
                                                  self.gain_cache, self.discount_cache, self.cutoff)


    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'NDCG' if self.cutoff < 0 else 'NDCG@%d' % self.cutoff


class ExpectedReciprocalRank(AbstractMetric):
    '''
    Expected Reciprocal Rank as described in [1].

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of NDCG metric
    for faster computation.

    Optionally, you can specify `max_relevance` but it will depend on queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into IndexError or similar RuntimeError and you
         will be at least given a hint about what is going on.

    [1] Olivier Chapelle et. al. Expected Reciprocal Rank for Graded Relevance, CIKM'2009

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

    queries: list of rankpy.queries.Queries
        The collections of queries that are known to be evaluated by this metric.
        These are used to compute `max_relevance` and `max_documents`.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, queries=None):
        super(ExpectedReciprocalRank, self).__init__(cutoff)
        # Get the maximum relevance score from the specified set(s) of queries...
        if queries is not None:
            max_relevance = max([qs.highest_relevance() for qs in queries])
        else:
            # or use the parameter given. None value indicate that explicit
            # value was not given which may lead to to runtime errors, hence
            # a user warnings are issued.
            if max_relevance is None:
                max_relevance = 4
                warn('Maximum relevance label was not explicitly specified '\
                     '(using default value 4). This should be avoided in order '\
                     'not to encounter runtime error (IndexError)!')
        self.R = (2.0**np.arange(max_relevance + 1, dtype=np.float64) - 1) / 2.0**max_relevance


    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scale=None):
        '''
        Evaluate ERR metric on the specified ranked list of document relevance scores.

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
        if ranked_labels is None:
            if ranking is not None and labels is not None:
                ranked_labels = labels.take(ranking)
            else:
                raise ValueError('cannot evaluate the ranking because provided arguments were empty')

        if cutoff < 0:
            cutoff = ranked_labels.size

        R = self.R.take(ranked_labels[:cutoff])
        P = np.ones_like(R)
        P[1:] -= R[:-1]
        np.cumprod(P, out=P)
        R /= np.arange(1, cutoff + 1, dtype=np.float64)

        return np.dot(R, P)


    def evaluate_queries(self, queries, scores, scale=None):
        '''
        Evaluate the ERR metric on the specified set of queries (`queries`). The documents
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
            The ideal DCG values for each query.
        '''
        raise NotImplementedError()


    def compute_delta(self, i, offset, document_ranks, relevance_scores, scale=None, out=None):
        '''
        Compute the change in the ERR metric after swapping document 'i' with
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

        scale: float, optional (default is None)
            The resulting deltas will be scaled by this factor.
        '''
        raise NotImplementedError()


    def compute_scale(self, queries):
        '''
        Since ERR is not normalized, return None.
        '''
        return None


    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'ERR' if self.cutoff < 0 else 'ERR@%d' % self.cutoff
