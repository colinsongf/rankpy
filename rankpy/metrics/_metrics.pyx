# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
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
# along with RankPy. If not, see <http://www.gnu.org/licenses/>.


from cython cimport view

from cpython cimport Py_INCREF, PyObject

from libc.stdlib cimport malloc, calloc, realloc, free
from libc.string cimport memset
from libc.math cimport log2

from ._utils cimport ranksort_queries_c, ranksort_relevance_scores_queries_c
from ..models.users.users_inner cimport CascadeUserModel

import numpy as np
cimport numpy as np
np.import_array()


# =============================================================================
# Types, constants, inline and helper functions and structures
# =============================================================================

cdef inline INT_t imin(INT_t a, INT_t b) nogil:
    return b if b < a else a

cdef inline INT_t imax(INT_t a, INT_t b) nogil:
    return b if b > a else a

cdef inline DOUBLE_t fabs(DOUBLE_t a) nogil:
    return -a if a < 0 else a

cdef inline INT_t filter_weightless_documents_from_ranking(INT_t *ranking,
                                                           DOUBLE_t *weights,
                                                           INT_t n_documents) nogil:
    '''
    Collapses the indices in `ranking` such that

        ``for all i in range(N): weights[ranking[i]] > 0``

    where N (the return value) is the number of documents with non-zero weight.
    '''
    cdef INT_t i, j = 0

    for i in range(n_documents):
        if weights[ranking[i]] > 0.0:
            ranking[j] = ranking[i]
            j += 1

    return j

cdef struct ERRDeltaInfo:
    DOUBLE_t *e
    DOUBLE_t *p
    INT_t    *r
    INT_t     c

cdef struct WTADeltaInfo:
    INT_t r1
    INT_t r2

cdef struct MAPDeltaInfo:
    DOUBLE_t *v
    INT_t     c
    DOUBLE_t  k
    INT_t     cutoff

cdef struct MRRDeltaInfo:
    INT_t rr1
    INT_t rr2

cdef struct CTRDeltaInfo:
    INT_t *r
    INT_t  c

# =============================================================================
# Metric
# =============================================================================

cdef class Metric:
    '''
    The interface for an information retrieval evaluation metric.
    '''
    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        All these values should allow the metric object to pre-allocate
        and precompute `something`, which may help it to evaluate
        the metric for queries faster.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance : int
            The maximum relevance score a document can have.

        maximum_documents : int
            The maximum number of documents a query can have.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        self.cutoff = cutoff
        self.seed = seed

        if cutoff == 0:
            raise ValueError('cutoff cannot be 0')

        if seed == 0:
            raise ValueError('seed cannot be 0')

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        pass

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        pass

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        pass

    cpdef delta(self,
                INT_t i,
                INT_t offset,
                INT_t[::1] document_ranks,
                INT_t[::1] relevance_scores,
                DOUBLE_t[::1] document_weights,
                INT_t nnz_documents,
                DOUBLE_t scale_value,
                DOUBLE_t[::1] out):
        '''
        Compute the change in the metric caused by swapping document `i`
        with every document `i + 1`, `i + 2`, ...

        The relevance score and rank of document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters
        ----------
        i : int
            The index of the document that is swapped with others.

        offset: int
            The offset pointer to the start of the documents to be swapped.

        document_ranks : array of ints, shape = [n_documents]
            The position in the ranking of all query documents.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance score of all query documents.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        nnz_documents : int
            The number of documents with non-zero weight.

        scale_value : float
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        out : array of floats, shape = [len(document_ranks) - i - 1]
            The output array. The array size is expected to be at least as big
            as the the number of pairs of documents that will be swapped, which
            is ``len(document_ranks) - i - 1``.
        '''
        cdef DOUBLE_t * weights = NULL

        with nogil:
            if document_weights is not None:
                weights = &document_weights[0]

            self.delta_c(i, offset, document_ranks.shape[0],
                         &document_ranks[0], &relevance_scores[0],
                         weights, nnz_documents, scale_value, &out[0])

    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=NULL) nogil:
        return NULL

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of self.delta(...) method.
        '''
        pass

    cdef void finalize_delta_c(self,
                               void *info) nogil:
        pass

# =============================================================================
# Discounted Cumulative Gain Metric
# =============================================================================

cdef class DiscountedCumulativeGain(Metric):
    '''
    Discounted Cumulative Gain (DCG) metric.
    '''

    cdef DOUBLE_t *gain_cache
    cdef INT_t     maximum_relevance

    cdef DOUBLE_t *discount_cache
    cdef INT_t     maximum_documents

    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the DCG metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance : int
            The maximum relevance score a document can have.

        maximum_documents : int
            The maximum number of documents a query can have.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        cdef INT_t i
        cdef DOUBLE_t gain

        self.gain_cache = NULL
        self.discount_cache = NULL

        if maximum_relevance <= 0:
            maximum_relevance = 4

        if maximum_documents <= 0:
            maximum_documents = 1024

        self.maximum_relevance = maximum_relevance
        self.maximum_documents = maximum_documents

        self.gain_cache = <DOUBLE_t*> calloc(self.maximum_relevance + 1,
                                             sizeof(DOUBLE_t))

        if self.gain_cache == NULL:
            raise MemoryError()

        self.discount_cache = <DOUBLE_t*> calloc(self.maximum_documents,
                                                 sizeof(DOUBLE_t))

        if self.discount_cache == NULL:
            raise MemoryError()

        gain = 1.0
        for i in range(maximum_relevance + 1):
            self.gain_cache[i] = gain - 1.0
            gain *= 2

        for i in range(maximum_documents):
            self.discount_cache[i] = log2(2.0 + i)

    def __dealloc__(self):
        '''
        Clean up the cached gain and discount values.
        '''
        free(self.gain_cache)
        free(self.discount_cache)

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (DiscountedCumulativeGain, (self.cutoff, self.maximum_relevance,
                                           self.maximum_documents, self.seed))

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the DCG metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            if self.cutoff < 0:
                cutoff = ranking.shape[0]
            else:
                cutoff = imin(self.cutoff, ranking.shape[0])

            result = 0.0

            for i in range(cutoff):
                result += (self.gain_cache[relevance_scores[ranking[i]]] /
                           self.discount_cache[i])

            result = query_weight * result / scale_value

        return result

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the DCG metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            if self.cutoff < 0:
                cutoff = ranked_relevance_scores.shape[0]
            else:
                cutoff = imin(self.cutoff, ranked_relevance_scores.shape[0])

            result = 0.0

            for i in range(cutoff):
                result += (self.gain_cache[ranked_relevance_scores[i]] /
                           self.discount_cache[i])

            result = query_weight * result / scale_value

        return result

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the DCG metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, k, n_queries, n_documents, rc
            INT_t *rankings = NULL
            DOUBLE_t result, qresult, qweight, query_weights_sum
            INT_t *relevance_scores_ptr = NULL
            DOUBLE_t *document_weights_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight = 0.0, 1.0

                # Override the default query weight.
                if query_weights is not None:
                    qweight = query_weights[i]

                # Queries with zero weight are ignored.
                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                # The number of queries of the current query.
                n_documents = query_indptr[i + 1] - query_indptr[i]

                if self.cutoff < 0:
                    cutoff = n_documents
                else:
                    cutoff = imin(self.cutoff, n_documents)

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = (&relevance_scores[0] + query_indptr[i])

                if document_weights is None:
                    # This computation is faster if all documents should be
                    # used in evaluation of the metric.
                    for j in range(cutoff):
                        qresult += (self.gain_cache[relevance_scores_ptr[rankings[j]]] /
                                    self.discount_cache[j])
                else:
                    # This branch evaluates the metric while ignoring
                    # documents with zero weight.
                    document_weights_ptr = (&document_weights[0] +
                                            query_indptr[i])

                    k = 0
                    for j in range(n_documents):
                        if document_weights_ptr[rankings[j]] != 0.0:
                            qresult += (self.gain_cache[relevance_scores_ptr[rankings[j]]] /
                                        self.discount_cache[k])
                            k += 1
                            if k == cutoff:
                                break

                    # When all documents have 0 weight, it is as if there
                    # was no query at all.
                    if k == 0:
                        qweight = 0.0

                rankings -= query_indptr[i] - query_indptr[0]

                qresult *= qweight
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of Metric.delta(...) method.
        '''
        cdef:
            INT_t j, cutoff
            DOUBLE_t i_relevance_score, i_position_discount

        if n_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        # Determine the cutoff rank.
        cutoff = n_documents if self.cutoff < 0 else self.cutoff

        i_relevance_score = self.gain_cache[relevance_scores[i]]
        i_position_discount = self.discount_cache[document_ranks[i]]

        for j in range(offset, n_documents):
            out[j - offset] = 0.0

            if document_ranks[j] < cutoff:
                out[j - offset] += ((i_relevance_score -
                                    self.gain_cache[relevance_scores[j]]) /
                                   self.discount_cache[document_ranks[j]])

            if document_ranks[i] < cutoff:
                out[j - offset] += (self.gain_cache[relevance_scores[j]] -
                                   i_relevance_score) / i_position_discount

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j])

# =============================================================================
# Mean Precision Metric
# =============================================================================

cdef class MeanPrecision(Metric):
    '''
    Mean Precision (MPN) metric.
    '''
    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the MPN metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If smaller than
            or equal to 0, ValueError is raised.

        maximum_relevance : int
            Not used.

        maximum_documents : int
            Not used.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        if cutoff <= 0:
            raise ValueError('mean precision metric cutoff must '
                             'be a positive integer')

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (MeanPrecision, (self.cutoff, 0, 0, self.seed))

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the MPN metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            if self.cutoff < 0:
                cutoff = ranking.shape[0]
            else:
                cutoff = imin(self.cutoff, ranking.shape[0])

            result = 0.0

            for i in range(cutoff):
                result += (relevance_scores[ranking[i]] > 0)

            result = query_weight * (result / cutoff) / scale_value

        return result

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the MPN metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            if self.cutoff < 0:
                cutoff = ranked_relevance_scores.shape[0]
            else:
                cutoff = imin(self.cutoff, ranked_relevance_scores.shape[0])

            result = 0.0

            for i in range(cutoff):
                result += (ranked_relevance_scores[i] > 0)

            result = query_weight * (result / cutoff) / scale_value

        return result

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the MPN metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, k, n_queries, n_documents, rc, cutoff
            INT_t *rankings = NULL
            DOUBLE_t result, qresult, qweight, query_weights_sum
            INT_t *relevance_scores_ptr = NULL
            DOUBLE_t *document_weights_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight = 0.0, 1.0

                # Override the default query weight.
                if query_weights is not None:
                    qweight = query_weights[i]

                # Queries with zero weight are ignored.
                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                # The number of queries of the current query.
                n_documents = query_indptr[i + 1] - query_indptr[i]

                if self.cutoff < 0:
                    cutoff = n_documents
                else:
                    cutoff = imin(self.cutoff, n_documents)

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = (&relevance_scores[0] + query_indptr[i])

                if document_weights is None:
                    # This computation is faster if all documents should be
                    # used in evaluation of the metric.
                    for j in range(cutoff):
                        qresult += (relevance_scores_ptr[rankings[j]] > 0)
                    qresult /= cutoff
                else:
                    # This branch evaluates the metric while ignoring
                    # documents with zero weight.
                    document_weights_ptr = (&document_weights[0] +
                                            query_indptr[i])
                    k = 0
                    for j in range(n_documents):
                        if document_weights_ptr[rankings[j]] != 0.0:
                            qresult += (relevance_scores_ptr[rankings[j]] > 0)
                            k += 1
                            if k == cutoff:
                                break

                    # When all documents have 0 weight, it is as if there
                    # was no query at all.
                    if k == 0:
                        qweight = 0.0
                    else:
                        # Precision at `k`.
                        qresult /= k

                rankings -= query_indptr[i] - query_indptr[0]

                qresult *= qweight
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of Metric.delta(...) method.
        '''
        cdef:
            INT_t j, cutoff = self.cutoff
            DOUBLE_t i_relevance_score, j_relevance_score

        if n_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        i_relevance_score = (relevance_scores[i] > 0)

        for j in range(offset, n_documents):
            out[j - offset] = 0.0

            j_relevance_score = (relevance_scores[j] > 0)

            if document_ranks[j] < cutoff:
                out[j - offset] += (i_relevance_score - j_relevance_score)

            if document_ranks[i] < cutoff:
                out[j - offset] += (j_relevance_score - i_relevance_score)

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / cutoff / scale_value)
        else:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / cutoff)

# =============================================================================
# Winner Takes All Metric
# =============================================================================

cdef class WinnerTakesAll(Metric):
    '''
    Winner Takes All (WTA) metric.

    Note that this is a generalized version of WTA metric, which value is 1.0
    if at least one document with the highest relevance score is present in
    the top `cutoff` number of documents.
    '''
    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the WTA metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If smaller than
            or equal to 0, ValueError is raised.

        maximum_relevance : int
            Not used.

        maximum_documents : int
            Not used.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        if cutoff <= 0:
            raise ValueError('winner takes all metric cutoff must '
                             'be a positive integer')

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (WinnerTakesAll, (self.cutoff, 0, 0, self.seed))

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the WTA metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Ignored.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, hrel = -1
            DOUBLE_t result = 0.0

        with nogil:
            for i in range(imin(self.cutoff, ranking.shape[0])):
                if hrel < relevance_scores[ranking[i]]:
                    hrel = relevance_scores[ranking[i]]

            result = query_weight * hrel / scale_value

        return result

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the WTA metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Ignored.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, hrel = -1
            DOUBLE_t result = 0.0

        with nogil:
            for i in range(imin(self.cutoff, ranked_relevance_scores.shape[0])):
                if hrel < ranked_relevance_scores[i]:
                    hrel = ranked_relevance_scores[i]

            result = query_weight * hrel / scale_value

        return result

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the WTA metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        If `ranking_scores` is None, the queries are evaluated on the ideal
        rankings.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Ignored.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, k, n_queries, n_documents, rc, qhrel, cutoff
            INT_t *rankings = NULL
            DOUBLE_t result, qresult, qweight, query_weights_sum
            INT_t *relevance_scores_ptr = NULL
            DOUBLE_t *document_weights_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight, qhrel = 0.0, 1.0, 0

                # Override the default query weight.
                if query_weights is not None:
                    qweight = query_weights[i]

                # Queries with zero weight are ignored.
                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = &relevance_scores[query_indptr[i]]

                # The total number of queries in the current query...
                n_documents = query_indptr[i + 1] - query_indptr[i]

                # ... which can be adjusted by removing documents
                # with 0 weight.
                if document_weights is not None:
                    n_documents = filter_weightless_documents_from_ranking(
                                        rankings,
                                        &document_weights[query_indptr[i]],
                                        n_documents)

                # Determine the cutoff...
                if self.cutoff < 0:
                    cutoff = n_documents
                else:
                    cutoff = imin(self.cutoff, n_documents)

                # ... which is 0 only if all documents have 0 weight. Such
                # queries need to be ignored.
                if cutoff == 0:
                    if out is not None: out[i] = 0.0
                    continue

                for j in range(cutoff):
                    if qhrel < relevance_scores_ptr[rankings[j]]:
                        qhrel = relevance_scores_ptr[rankings[j]]

                rankings -= query_indptr[i] - query_indptr[0]

                qresult = qweight * qhrel
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=NULL) nogil:
        cdef:
            INT_t i, cutoff, r1 = 0, r2 = 0, n_r1 = 0
            WTADeltaInfo *info_ = NULL

        if n_documents == 0:
            return NULL

        if info == NULL:
            info_ = <WTADeltaInfo *> malloc(sizeof(WTADeltaInfo))
            if info_ == NULL:
                return NULL
        else:
            info_ = <WTADeltaInfo *> info

        cutoff = imin(self.cutoff, n_documents)

        if document_weights == NULL:
            for i in range(n_documents):
                if document_ranks[i] < cutoff:
                    if r1 <= relevance_scores[i]:
                        r2 = r1
                        r1 = relevance_scores[i]
                    elif r2 < relevance_scores[i]:
                        r2 = relevance_scores[i]
        else:
            for i in range(n_documents):
                if document_weights[i] != 0:
                    if document_ranks[i] < cutoff:
                        if r1 <= relevance_scores[i]:
                            r2 = r1
                            r1 = relevance_scores[i]
                        elif r2 < relevance_scores[i]:
                            r2 = relevance_scores[i]

        # Keep the highest and second highest relevance score
        # among to the documents above the cutoff.
        # score as well.
        info_.r1 = r1
        info_.r2 = r2

        return info_

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of Metric.delta(...) method.
        '''
        cdef:
            INT_t j, cutoff = self.cutoff
            DOUBLE_t i_relevance_score, j_relevance_score
            WTADeltaInfo *info_ = NULL

        if n_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        if info == NULL:
            info_ = <WTADeltaInfo *> self.prepare_delta_c(document_ranks,
                                                          relevance_scores,
                                                          document_weights,
                                                          n_documents,
                                                          scale_value)
        else:
            info_ = <WTADeltaInfo *> info

        if info_ == NULL:
            # FIXME: This should be reported to the caller!!!
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        for j in range(offset, n_documents):
            out[j - offset] = 0.0

            if document_ranks[i] < cutoff:
                if document_ranks[j] >= cutoff:
                    if relevance_scores[i] == info_.r1:
                        if relevance_scores[i] == info_.r2:
                            out[j - offset] = imax(0, (relevance_scores[j] -
                                                       info_.r1))
                        else:
                            out[j - offset] = imax(relevance_scores[j],
                                                   info_.r2) - info_.r1
                    else:
                        out[j - offset] = imax(0, (relevance_scores[j] -
                                                   info_.r1))
            elif document_ranks[j] < cutoff:
                if relevance_scores[j] == info_.r1:
                    if relevance_scores[j] == info_.r2:
                        out[j - offset] = imax(0, (relevance_scores[i] -
                                                   info_.r1))
                    else:
                        out[j - offset] = imax(relevance_scores[i],
                                               info_.r2) - info_.r1
                else:
                    out[j - offset] = imax(0, (relevance_scores[i] -
                                               info_.r1))

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j])

        if info == NULL:
            self.finalize_delta_c(info_)

    cdef void finalize_delta_c(self,
                               void *info) nogil:
        if info != NULL:
            free(info)

# =============================================================================
# Mean Average Precision Metric
# =============================================================================

cdef class MeanAveragePrecision(Metric):
    '''
    Mean Average Precision (MAP) metric.
    '''
    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the MAP metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance : int
            Not used.

        maximum_documents : int
            Not used.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        pass

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (MeanAveragePrecision, (self.cutoff, 0, 0, self.seed))

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the MAP metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, n_relevant, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranking.shape[0]

            if self.cutoff < 0:
                cutoff = ranking.shape[0]
            else:
                cutoff = imin(self.cutoff, ranking.shape[0])

            result = 0.0
            n_relevant = 0

            for i in range(cutoff):
                if relevance_scores[ranking[i]] > 0:
                    n_relevant += 1
                    result += (<DOUBLE_t> n_relevant) / (1.0 + i)

            for i in range(cutoff, n_documents):
                if relevance_scores[ranking[i]] > 0:
                    n_relevant += 1

            if n_relevant != 0:
                result *= query_weight
                result /= scale_value
                result /= imin(cutoff, n_relevant)

        return result

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the MAP metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, n_relevant, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranked_relevance_scores.shape[0]

            if self.cutoff < 0:
                cutoff = ranked_relevance_scores.shape[0]
            else:
                cutoff = imin(self.cutoff, ranked_relevance_scores.shape[0])

            result = 0.0
            n_relevant = 0

            for i in range(cutoff):
                if ranked_relevance_scores[i] > 0:
                    n_relevant += 1
                    result += (<DOUBLE_t> n_relevant) / (1.0 + i)

            for i in range(cutoff, n_documents):
                if ranked_relevance_scores[i] > 0:
                    n_relevant += 1

            if n_relevant != 0:
                result *= query_weight
                result /= scale_value
                result /= imin(cutoff, n_relevant)

        return result

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the MAP metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, k, n_queries, n_documents, n_relevant, cutoff, rc
            INT_t *rankings = NULL
            DOUBLE_t result, qresult, qweight, query_weights_sum
            INT_t *relevance_scores_ptr = NULL
            DOUBLE_t *document_weights_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight = 0.0, 1.0

                # Override the default query weight.
                if query_weights is not None:
                    qweight = query_weights[i]

                # Queries with zero weight are ignored.
                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                n_documents = query_indptr[i + 1] - query_indptr[i]
                n_relevant = 0

                # Count the number of documents with non-zero weight.
                if document_weights is None:
                    cutoff = n_documents
                else:
                    cutoff = 0
                    for j in range(n_documents):
                        cutoff += (document_weights[query_indptr[i] + j] > 0.)

                if self.cutoff > 0:
                    cutoff = imin(cutoff, self.cutoff)

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = (&relevance_scores[0] + query_indptr[i])

                if document_weights is None:
                    for j in range(cutoff):
                        if relevance_scores_ptr[rankings[j]] > 0:
                            n_relevant += 1
                            qresult += (<DOUBLE_t> n_relevant) / (1.0 + j)

                    for j in range(cutoff, n_documents):
                        if relevance_scores_ptr[rankings[j]] > 0:
                            n_relevant += 1

                    if n_relevant != 0:
                        qresult /= imin(cutoff, n_relevant)
                else:
                    document_weights_ptr = (&document_weights[0] +
                                            query_indptr[i])
                    k = 0
                    for j in range(n_documents):
                        if document_weights_ptr[rankings[j]] != 0.0:
                            if relevance_scores_ptr[rankings[j]] > 0:
                                n_relevant += 1
                                if k < cutoff:
                                    qresult += (<DOUBLE_t> n_relevant) / (1.0 + k)
                            k += 1

                    # When all documents have 0 weight, it is as if there
                    # was no query at all.
                    if k == 0:
                        qweight = 0.0
                    elif n_relevant != 0:
                        qresult /= imin(cutoff, n_relevant)

                rankings -= query_indptr[i] - query_indptr[0]

                qresult *= qweight
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=NULL) nogil:
        cdef:
            INT_t         i, j, k
            INT_t         cutoff = 0

            INT_t        *r = NULL
            DOUBLE_t     *b = NULL
            DOUBLE_t     *v = NULL

            DOUBLE_t      rcount = 0.0
            DOUBLE_t      tmp = 0.0

            MAPDeltaInfo *info_ = NULL

        if n_documents == 0:
            return NULL

        if document_weights != NULL:
            for i in range(n_documents):
                cutoff += (document_weights[i] != 0)
        else:
            cutoff = n_documents

        if self.cutoff > 0:
            cutoff = imin(cutoff, self.cutoff)

        if info == NULL:
            info_ = <MAPDeltaInfo *> malloc(sizeof(MAPDeltaInfo))

            if info_ == NULL:
                return NULL

            v = <DOUBLE_t *> malloc(2 * cutoff * sizeof(DOUBLE_t))

            if v == NULL:
                self.finalize_delta_c(info_)
                return NULL

            info_.v = v
            info_.c = cutoff

        else:
            info_ = <MAPDeltaInfo *> info

            # Checks there is enough space in the buffers.
            if info_.c < cutoff:
                v = <DOUBLE_t *> realloc(info_.v, 2 * cutoff * sizeof(DOUBLE_t))

                if v == NULL:
                    self.finalize_delta_c(info_)
                    return NULL

                info_.v = v
                info_.c = cutoff

        v = info_.v
        b = v + cutoff
        k = 0

        r = <INT_t *> malloc(n_documents * sizeof(INT_t))

        if r == NULL:
            self.finalize_delta_c(info_)
            return NULL

        for i in range(n_documents):
            r[document_ranks[i]] = i

        # Need to filter out documents with 0 weight from the ranking
        # and count the number of relevant documents.

        if document_weights != NULL:
            j = 0
            for i in range(n_documents):
                if document_weights[r[i]] != 0:
                    r[j] = relevance_scores[r[i]]
                    k += (r[j] > 0)
                    j += 1
        else:
            for i in range(n_documents):
                r[i] = relevance_scores[r[i]]
                k += (r[i] > 0)

        info_.k = imin(cutoff, k)
        info_.cutoff = cutoff

        for i in range(cutoff):
            if r[i] > 0:
                rcount += 1.0
                b[i] = 1.0 / (i + 1.0)
                v[i] = rcount / (i + 1.0)
            else:
                v[i] = (rcount + 1.0) / (i + 1.0)
                b[i] = 0

        for i in range(cutoff - 1, -1, -1):
            v[i] += tmp
            tmp += b[i]

        free(r)

        return info_

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of self.delta(...) method.
        '''
        cdef:
            INT_t j, rel_i, rel_j, rank_i, rank_j, cutoff
            DOUBLE_t *v = NULL
            DOUBLE_t k, d
            MAPDeltaInfo *info_ = NULL

        if n_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        if info == NULL:
            info_ = <MAPDeltaInfo *> self.prepare_delta_c(document_ranks,
                                                          relevance_scores,
                                                          document_weights,
                                                          n_documents,
                                                          scale_value)
        else:
            info_ = <MAPDeltaInfo *> info

        if info_ == NULL:
            # FIXME: This should be reported to the caller!!!
            return

        # If there is no relevant document, all deltas are set to 0.
        if info_.k == 0:
            memset(out, 0, (n_documents - offset) * sizeof(DOUBLE_t))
            if info == NULL:
                self.finalize_delta_c(info_)
            return

        v = info_.v
        k = info_.k
        cutoff = info_.cutoff

        rel_i = (relevance_scores[i] > 0)
        rank_i = document_ranks[i]

        d = 2. * rel_i - 1.

        for j in range(offset, n_documents):
            rel_j = (relevance_scores[j] > 0)
            rank_j = document_ranks[j]

            out[j - offset] = 0.

            if rel_i == rel_j:
                continue

            if rank_i < cutoff:
                if rank_j < cutoff:
                    out[j - offset] -=  d / (imax(rank_i, rank_j) + 1.)
                out[j - offset] -= v[rank_i]

            if rank_j < cutoff:
                out[j - offset] += v[rank_j]

            out[j - offset] = (out[j - offset]) / k

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j])

        if info == NULL:
            self.finalize_delta_c(info_)

    cdef void finalize_delta_c(self, void *info) nogil:
        if info != NULL:
            free((<MAPDeltaInfo *> info).v)
            free(info)

# =============================================================================
# Mean Reciprocal Rank Metric
# =============================================================================

cdef class MeanReciprocalRank(Metric):
    '''
    Mean Reciprocal Rank (MRR) metric.
    '''
    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the MRR metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance : int
            Not used.

        maximum_documents : int
            Not used.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        if cutoff <= 0:
            raise ValueError('mean reciprocal rank metric cutoff must '
                             'be a positive integer')

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (MeanReciprocalRank,
                (self.cutoff, self.maximum_relevance, self.maximum_documents,
                 self.seed))

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the MRR metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranking.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            for i in range(imin(cutoff, n_documents)):
                if relevance_scores[ranking[i]] > 0:
                    result = 1. / (i + 1.0)
                    break

            result = query_weight * result / scale_value

        return result

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the MRR metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranked_relevance_scores.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            for i in range(imin(cutoff, n_documents)):
                if ranked_relevance_scores[i] > 0:
                    result = 1. / (i + 1.0)
                    break

            result = query_weight * result / scale_value

        return result

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the MRR metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, k, n_queries, n_documents, cutoff, rc
            INT_t *rankings = NULL
            DOUBLE_t result, qresult, qweight, qp, query_weights_sum
            INT_t *relevance_scores_ptr = NULL
            DOUBLE_t *document_weights_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight = 0.0, 1.0

                if query_weights is not None:
                    qweight = query_weights[i]

                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                n_documents = query_indptr[i + 1] - query_indptr[i]

                if self.cutoff < 0:
                    cutoff = n_documents
                else:
                    cutoff = imin(self.cutoff, n_documents)

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = (&relevance_scores[0] + query_indptr[i])

                if document_weights is None:
                    for j in range(cutoff):
                        if relevance_scores_ptr[rankings[j]] > 0:
                            qresult = 1.0 / (j + 1.0)
                            break
                else:
                    document_weights_ptr = (&document_weights[0] +
                                            query_indptr[i])
                    k = 0
                    for j in range(n_documents):
                        if document_weights_ptr[rankings[j]] != 0.0:
                            k += 1
                            if relevance_scores_ptr[rankings[j]] > 0:
                                qresult = 1.0 / k
                                break
                            if k == cutoff:
                                break

                    # When all documents have 0 weight, it is as if there
                    # was no query at all.
                    if k == 0:
                        qweight = 0.0

                rankings -= query_indptr[i] - query_indptr[0]

                qresult *= qweight
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=NULL) nogil:
        cdef:
            INT_t i, rr1, rr2, cutoff
            MRRDeltaInfo *info_ = NULL

        if n_documents == 0:
            return NULL

        if info == NULL:
            info_ = <MRRDeltaInfo *> malloc(sizeof(MRRDeltaInfo))
            if info_ == NULL:
                return NULL
        else:
            info_ = <MRRDeltaInfo *> info

        if self.cutoff < 0:
            cutoff = n_documents
        else:
            cutoff = imin(self.cutoff, n_documents)

        rr1, rr2 = cutoff, cutoff

        if document_weights == NULL:
            for i in range(n_documents):
                if document_ranks[i] < cutoff and relevance_scores[i] > 0:
                    if rr1 > document_ranks[i]:
                        rr2 = rr1
                        rr1 = document_ranks[i]
                    elif rr2 > document_ranks[i]:
                        rr2 = document_ranks[i]
        else:
            for i in range(n_documents):
                if (document_weights[i] != 0 and
                    document_ranks[i] < cutoff and
                    relevance_scores[i] > 0):
                    if rr1 > document_ranks[i]:
                        rr2 = rr1
                        rr1 = document_ranks[i]
                    elif rr2 > document_ranks[i]:
                        rr2 = document_ranks[i]

        if rr2 != cutoff:
            info_.rr1 = rr1
        else:
            info_.rr1 = -1

        if rr2 != cutoff:
            info_.rr2 = rr2
        else:
            info_.rr2 = -1

        return info_

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of self.delta(...) method.
        '''
        cdef:
            INT_t j, rel_i, rel_j, rr1, rr2
            DOUBLE_t cv = 0.0
            MRRDeltaInfo *info_ = NULL

        if n_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        if info == NULL:
            info_ = <MRRDeltaInfo *> self.prepare_delta_c(document_ranks,
                                                          relevance_scores,
                                                          document_weights,
                                                          n_documents,
                                                          scale_value)
        else:
            info_ = <MRRDeltaInfo *> info

        if info_ == NULL:
            # FIXME: This should be reported to the caller!!!
            return

        rr1 = info_.rr1
        rr2 = info_.rr2

        if rr1 != -1:
            cv = 1.0 / (rr1 + 1.0)
        else:
            cv = 0
            rr1 = self.cutoff

        rel_i = (relevance_scores[i] > 0)

        for j in range(offset, n_documents):
            out[j - offset] = 0.0
            rel_j = (relevance_scores[j] > 0)

            if rel_i == rel_j:
                continue

            if document_ranks[i] < rr1:
                out[j - offset] = cv - 1.0 / (document_ranks[i] + 1.0)
            elif document_ranks[i] == rr1:
                out[j - offset] = cv
                if rr2 != -1:
                    out[j - offset] -= 1.0 / (imin(document_ranks[j], rr2) + 1.0)
            elif document_ranks[j] < rr1:
                out[j - offset] = cv - 1.0 / (document_ranks[j] + 1.0)
            elif document_ranks[j] == rr1:
                out[j - offset] = cv
                if rr2 != -1:
                    out[j - offset] -= 1.0 / (imin(document_ranks[i], rr2) + 1.0)

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j])

        if info == NULL:
            self.finalize_delta_c(info_)

    cdef void finalize_delta_c(self, void *info) nogil:
        if info != NULL:
            free(info)

# =============================================================================
# Expected Reciprocal Rank Metric
# =============================================================================

cdef class ExpectedReciprocalRank(Metric):
    '''
    Expected Reciprocal Rank (ERR) metric.
    '''

    cdef DOUBLE_t *gain_cache
    cdef INT_t     maximum_relevance

    cdef DOUBLE_t *discount_cache
    cdef INT_t     maximum_documents

    cdef bint      dcg_discount

    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed,
                  bint dcg_discount=False):
        '''
        Initialize the ERR metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        All these values should allow the metric object to pre-allocate
        and precompute `something`, which may help it to evaluate
        the metric for queries faster.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance : int
            The maximum relevance score a document can have.

        maximum_documents : int
            The maximum number of documents a query can have.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        cdef INT_t i
        cdef DOUBLE_t gain

        self.gain_cache = NULL
        self.discount_cache = NULL

        if maximum_relevance <= 0:
            maximum_relevance = 4

        if maximum_documents <= 0:
            maximum_documents = 1024

        self.maximum_relevance = maximum_relevance
        self.maximum_documents = maximum_documents

        self.gain_cache = <DOUBLE_t*> calloc(self.maximum_relevance + 1,
                                             sizeof(DOUBLE_t))

        if self.gain_cache == NULL:
            raise MemoryError()

        self.discount_cache = <DOUBLE_t*> calloc(self.maximum_documents,
                                                 sizeof(DOUBLE_t))

        if self.discount_cache == NULL:
            raise MemoryError()

        gain = 1.0
        for i in range(maximum_relevance + 1):
            self.gain_cache[i] = gain - 1.0
            gain *= 2

        gain /= 2

        for i in range(maximum_relevance + 1):
            self.gain_cache[i] /= gain

        self.dcg_discount = dcg_discount

        if dcg_discount:
            for i in range(maximum_documents):
                self.discount_cache[i] = log2(2.0 + i)
        else:
            for i in range(maximum_documents):
                self.discount_cache[i] = (1.0 + i)

    def __dealloc__(self):
        '''
        Clean up the cached gain and discount values.
        '''
        free(self.gain_cache)
        free(self.discount_cache)

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (ExpectedReciprocalRank,
                (self.cutoff, self.maximum_relevance, self.maximum_documents,
                 self.seed, self.dcg_discount))

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the ERR metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result, p

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranking.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            p = 1.0
            for i in range(imin(cutoff, n_documents)):
                result += p * self.gain_cache[relevance_scores[ranking[i]]] / self.discount_cache[i]
                p *= (1 - self.gain_cache[relevance_scores[ranking[i]]])
            result /= scale_value
            result *= query_weight

        return result

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the DCG metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result, p

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranked_relevance_scores.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            p = 1.0
            for i in range(imin(cutoff, n_documents)):
                result += p * self.gain_cache[ranked_relevance_scores[i]] / self.discount_cache[i]
                p *= (1 - self.gain_cache[ranked_relevance_scores[i]])
            result /= scale_value
            result *= query_weight

        return result

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the ERR metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, k, n_queries, n_documents, cutoff, rc
            INT_t *rankings = NULL
            DOUBLE_t result, qresult, qweight, qp, query_weights_sum
            INT_t *relevance_scores_ptr = NULL
            DOUBLE_t *document_weights_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight, qp = 0.0, 1.0, 1.0

                if query_weights is not None:
                    qweight = query_weights[i]

                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                n_documents = query_indptr[i + 1] - query_indptr[i]

                if self.cutoff < 0:
                    cutoff = n_documents
                else:
                    cutoff = imin(self.cutoff, n_documents)

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = (&relevance_scores[0] + query_indptr[i])

                if document_weights is None:
                    for j in range(cutoff):
                        qresult += qp * self.gain_cache[relevance_scores_ptr[rankings[j]]] / self.discount_cache[j]
                        qp *= (1 - self.gain_cache[relevance_scores_ptr[rankings[j]]])
                else:
                    document_weights_ptr = (&document_weights[0] +
                                            query_indptr[i])
                    k = 0
                    for j in range(n_documents):
                        if document_weights_ptr[rankings[j]] != 0.0:
                            qresult += qp * self.gain_cache[relevance_scores_ptr[rankings[j]]] / self.discount_cache[k]
                            qp *= (1 - self.gain_cache[relevance_scores_ptr[rankings[j]]])
                            k += 1
                            if k == cutoff:
                                break

                    # When all documents have 0 weight, it is as if there
                    # was no query at all.
                    if k == 0:
                        qweight = 0.0

                rankings -= query_indptr[i] - query_indptr[0]

                qresult *= qweight
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=NULL) nogil:
        cdef:
            INT_t i
            DOUBLE_t R
            DOUBLE_t *e = NULL
            DOUBLE_t *p = NULL
            ERRDeltaInfo *info_ = NULL

        if n_documents == 0:
            return NULL

        if info == NULL:
            info_ = <ERRDeltaInfo *> malloc(sizeof(ERRDeltaInfo))

            if info_ == NULL:
                return NULL

            info_.e = <DOUBLE_t *> malloc(2 * n_documents * sizeof(DOUBLE_t))
            info_.p = <DOUBLE_t *> malloc(2 * n_documents * sizeof(DOUBLE_t))
            info_.r = <INT_t *> malloc(2 * n_documents * sizeof(INT_t))
            info_.c = 2 * n_documents

            if (info_.e == NULL or info_.p == NULL or info_.r == NULL):
                self.finalize_delta_c(info_)
                return NULL
        else:
            info_ = <ERRDeltaInfo *> info

            # Checks there is enough space in the buffers.
            if info_.c < n_documents:
                e = <DOUBLE_t *> realloc(info_.e, 2 * n_documents * sizeof(DOUBLE_t))
                if e != NULL:
                    info_.e = e

                p = <DOUBLE_t *> realloc(info_.p, 2 * n_documents * sizeof(DOUBLE_t))
                if p != NULL:
                    info_.p = p

                r = <INT_t *> realloc(info_.r, 2 * n_documents * sizeof(INT_t))
                if r != NULL:
                    info_.r = r

                info_.c = 2 * n_documents

                if (e == NULL or p == NULL or r == NULL):
                    self.finalize_delta_c(info_)
                    return NULL

        e = info_.e
        p = info_.p
        r = info_.r

        # FIXME: Critical Assumption: document_ranks[i] < document_ranks[j]
        #        for all i and j for which document_weights[i] != 0 and
        #        document_weights[j] == 0, respectively.
        for i in range(n_documents):
            r[document_ranks[i]] = i

        R = self.gain_cache[relevance_scores[r[0]]]

        p[0] = 1.0
        e[0] = R / self.discount_cache[0]

        for i in range(1, n_documents):
            p[i] = p[i - 1] * (1 - R)
            R = self.gain_cache[relevance_scores[r[i]]]
            e[i] = e[i - 1] + p[i] * R / self.discount_cache[i]

        return info_

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of self.delta(...) method.
        '''
        cdef:
            INT_t j, j_, cutoff
            DOUBLE_t d, Ri, ri, Ti, Rj, rj, Tj
            DOUBLE_t *e = NULL
            DOUBLE_t *p = NULL
            ERRDeltaInfo *info_ = NULL

        if n_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        if info == NULL:
            info_ = <ERRDeltaInfo *> self.prepare_delta_c(document_ranks,
                                                          relevance_scores,
                                                          document_weights,
                                                          n_documents,
                                                          scale_value)
        else:
            info_ = <ERRDeltaInfo *> info

        if info_ == NULL:
            # FIXME: This should be reported to the caller!!!
            return

        e = info_.e
        p = info_.p

        if self.cutoff < 0:
            cutoff = nnz_documents
        else:
            cutoff = imin(self.cutoff, nnz_documents)

        Ri = self.gain_cache[relevance_scores[i]]
        ri = self.discount_cache[document_ranks[i]]
        Ti = (1 - Ri)

        for j in range(offset, n_documents):
            Rj = self.gain_cache[relevance_scores[j]]
            rj = self.discount_cache[document_ranks[j]]
            Tj = (1 - Rj)

            d = 0.0

            if document_ranks[i] < document_ranks[j]:

                if document_ranks[i] < cutoff:
                    d += (Ri - Rj) / ri * p[document_ranks[i]]

                    d += (1 - Tj / Ti) * (e[imin(document_ranks[j], cutoff) - 1] -
                                          e[document_ranks[i]])

                if document_ranks[j] < cutoff:
                    d += (Rj - Tj * Ri / Ti) / rj * p[document_ranks[j]]

            else:

                if document_ranks[j] < cutoff:
                    d += (Rj - Ri) / rj * p[document_ranks[j]]

                    d += (1 - Ti / Tj) * (e[imin(document_ranks[i], cutoff) - 1] -
                                          e[document_ranks[j]])

                if document_ranks[i] < cutoff:
                    d += (Ri - Ti * Rj / Tj) / ri * p[document_ranks[i]]

            out[j - offset] = d

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_documents - offset):
                out[j] = fabs(out[j])

        if info == NULL:
            self.finalize_delta_c(info_)

    cdef void finalize_delta_c(self, void *info) nogil:
        if info != NULL:
            free((<ERRDeltaInfo *> info).e)
            free((<ERRDeltaInfo *> info).p)
            free((<ERRDeltaInfo *> info).r)
            free(info)


# =============================================================================
# Clickthrough Rate Metric
# =============================================================================

cdef class ClickthroughRate(Metric):
    '''
    Clickthrough Rate Metric, internally using a CascadeUserModel
    as a source of the user simulated clicks.
    '''

    cdef public CascadeUserModel click_model
    cdef public bint             relative
    cdef public bint             sample
    cdef public INT_t            n_impressions

    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        '''
        Initialize the CTR metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        Parameters
        ----------
        cutoff : int
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance : int
            Not used.

        maximum_documents : int
            Not used.

        seed : int
            The seed for random number generator, which is used to
            break ties in rankings.
        '''
        pass

    def initialize_click_model(self, click_proba, stop_proba,
                               abandon_proba, relative=False,
                               sample=False, n_impressions=1000):
        '''
        Initializes the internal click model using the specified
        click probabilities, stop probabilities, and abandonment
        rate. `n_impressions` is the number of impressions of each
        query to a simulated user during computation of deltas, but
        only if `sample` is True.
        '''
        self.click_model = CascadeUserModel(click_proba, stop_proba,
                                            abandon_proba, self.seed)
        self.relative = relative
        self.sample = sample
        self.n_impressions = n_impressions

    def __reduce__(self):
        '''
        Reduce reimplementation, for pickling.
        '''
        return (ClickthroughRate, (self.cutoff, 0, 0, self.seed),
                self.__getstate__())

    def __setstate__(self, d):
        self.click_model = d['click_model']
        self.sample = d['sample']
        self.n_impressions = d['n_impressions']
        self.relative = d['relative']

    def __getstate__(self):
        d = {}
        d['click_model'] = self.click_model
        d['sample'] = self.sample
        d['n_impressions'] = self.n_impressions
        d['relative'] = self.relative
        return d

    cpdef evaluate_ranking(self,
                           INT_t[::1] ranking,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=1.0,
                           DOUBLE_t query_weight=1.0):
        '''
        Evaluate the CTR metric on the specified document ranking.

        Parameters
        ----------
        ranking : array of ints, shape = [n_documents]
            Specify the list of ranked documents.

        relevance_scores : array of ints, shape = [n_documents]
            Specify the relevance score for each document.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        cdef int i, j, n_documents
        cdef DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranking.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            if self.sample:
                for i in range(self.n_impressions):
                    if self.click_model.get_clicks_c(&ranking[0], cutoff,
                                                     &relevance_scores[0]) > 0:
                        result += 1.0
                result /= self.n_impressions
            else:
                result = self.click_model.get_clickthrough_rate_c(&ranking[0], cutoff, &relevance_scores[0], self.relative)

        return query_weight * result / scale_value

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=1.0,
                   DOUBLE_t query_weight=1.0):
        '''
        Evaluate the CTR metric on the specified ranked list
        of document relevance scores.

        Parameters
        ----------
        ranked_relevance_scores : array of ints, shape = [n_documents]
            Specify list of relevance scores.

        scale_value : float, optional (default=1.0)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight : float, optional (default=1.0)
            The weight of the query for which the metric is evaluated.
        '''
        return self.evaluate_ranking(np.arange(ranked_relevance_scores.shape[0],
                                               dtype=np.int32),
                                     ranked_relevance_scores,
                                     scale_value,
                                     query_weight)


    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out):
        '''
        Evaluate the CTR metric on the specified queries. The relevance scores
        and ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters
        ----------
        query_indptr : array of ints, shape = [n_queries + 1]
            The query index pointer.

        relevance_scores : array of ints, shape = [n_documents]
            The relevance scores of the documents.

        ranking_scores : array floats, shape = [n_documents]
            The ranking scores of the documents.

        scale_values : array of floats, shape = [n_queries], or None
            Optional parameter for implementation of metrics
            which evaluations need to be scaled by the ideal
            metric values.

        query_weights : array of floats, shape = [n_queries], or None
            The weight given to each query.

        document_weights : array of floats, shape = [n_documents], or None
            The weight of each document. It is used as an indicator of
            documents that should be ignored, which are those with 0 weight.

        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, cutoff, n_queries, n_documents, rc
            DOUBLE_t result, qresult, sample_result, qweight, query_weights_sum
            INT_t *rankings = NULL
            INT_t *relevance_scores_ptr = NULL

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0

            rankings = <INT_t*> calloc(n_documents, sizeof(INT_t))

            if rankings == NULL:
                with gil: raise MemoryError()

            rc = ranksort_queries_c(&query_indptr[0], n_queries,
                                    &ranking_scores[0], rankings,
                                    &self.seed)

            if rc == -1:
                free(rankings)
                with gil: raise MemoryError()

            result = 0.0

            for i in range(n_queries):
                qresult, qweight = 0.0, 1.0

                if query_weights is not None:
                    qweight = query_weights[i]

                if qweight == 0.0:
                    if out is not None: out[i] = 0.0
                    continue

                n_documents = query_indptr[i + 1] - query_indptr[i]

                # For convenient indexing of `i`-th query's document
                # ranking and relevance scores.
                rankings += query_indptr[i] - query_indptr[0]
                relevance_scores_ptr = &relevance_scores[query_indptr[i]]

                if document_weights is not None:
                    n_documents = filter_weightless_documents_from_ranking(
                                        rankings,
                                        &document_weights[query_indptr[i]],
                                        n_documents)

                if self.cutoff < 0:
                    cutoff = n_documents
                else:
                    cutoff = imin(self.cutoff, n_documents)

                if self.sample:
                    sample_result = 0.0
                    for j in range(self.n_impressions):
                        if self.click_model.get_clicks_c(rankings,
                                                         cutoff,
                                                         relevance_scores_ptr) > 0:
                            sample_result += 1.0
                    qresult = sample_result / self.n_impressions
                else:
                    qresult = self.click_model.get_clickthrough_rate_c(rankings,
                                                                       cutoff,
                                                                       relevance_scores_ptr,
                                                                       self.relative)

                rankings -= query_indptr[i] - query_indptr[0]

                qresult *= qweight
                query_weights_sum += qweight

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= query_weights_sum

            free(rankings)

        return result

    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=NULL) nogil:
        cdef:
            INT_t i, j
            INT_t *r = NULL
            CTRDeltaInfo *info_ = NULL

        if n_documents == 0:
            return NULL

        if info == NULL:
            info_ = <CTRDeltaInfo *> malloc(sizeof(CTRDeltaInfo))

            if info_ == NULL:
                return NULL

            info_.r = <INT_t *> malloc(2 * n_documents * sizeof(INT_t))

            if info_.r == NULL:
                self.finalize_delta_c(info_)
                return NULL

            info_.c = 2 * n_documents
        else:
            info_ = <CTRDeltaInfo *> info

            # Checks there is enough space in the buffers.
            if info_.c < n_documents:
                r = <INT_t *> realloc(info_.r, 2 * n_documents * sizeof(INT_t))

                if r != NULL:
                    info_.r = r
                else:
                    self.finalize_delta_c(info_)
                    return NULL

                info_.c = 2 * n_documents

        r = info_.r

        # FIXME: Critical Assumption: document_ranks[i] < document_ranks[j]
        #        for all i and j for which document_weights[i] != 0 and
        #        document_weights[j] == 0, respectively.
        for i in range(n_documents):
            r[document_ranks[i]] = i

        return info_

    cdef void delta_c(self,
                      INT_t i,
                      INT_t offset,
                      INT_t n_documents,
                      INT_t *document_ranks,
                      INT_t *relevance_scores,
                      DOUBLE_t *document_weights,
                      INT_t nnz_documents,
                      DOUBLE_t scale_value,
                      DOUBLE_t *out,
                      void *info=NULL) nogil:
        '''
        See description of self.delta(...) method.
        '''
        cdef:
            INT_t j, tmp, i_rank, j_rank, cutoff
            INT_t *ranking = NULL
            INT_t *filtered_document_ranks = NULL
            DOUBLE_t before = 0.0, after = 0.0
            CTRDeltaInfo *info_ = NULL

        if self.sample:
            self.delta_sample_c(i, offset, n_documents, document_ranks,
                                relevance_scores, document_weights,
                                nnz_documents, scale_value, out, info)
            return

        if n_documents == 0 or nnz_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        if info == NULL:
            info_ = <CTRDeltaInfo *> self.prepare_delta_c(document_ranks,
                                                          relevance_scores,
                                                          document_weights,
                                                          n_documents,
                                                          scale_value)
        else:
            info_ = <CTRDeltaInfo *> info

        if info_ == NULL:
            # FIXME: This should be reported to the caller!!!
            return

        ranking = info_.r
        cutoff = nnz_documents

        if self.cutoff >= 0 and cutoff > self.cutoff:
            cutoff = self.cutoff

        before = self.click_model.get_clickthrough_rate_c(
                        ranking, cutoff, relevance_scores, self.relative)

        i_rank = document_ranks[i]

        for j in range(offset, n_documents):
            out[j - offset] = 0.0

            j_rank = document_ranks[j]

            if i_rank < cutoff or j_rank < cutoff:
                # Swap document i and j in the ranking...
                tmp = ranking[i_rank]
                ranking[i_rank] = ranking[j_rank]
                ranking[j_rank] = tmp

                # ... and compute its clickthrough rate.
                after = self.click_model.get_clickthrough_rate_c(
                            ranking, cutoff, relevance_scores, self.relative)

                # Compute the (absolute) change in the metric...
                out[j - offset] = fabs(before - after)

                # ... and restore the original ranking for the next iteration.
                tmp = ranking[i_rank]
                ranking[i_rank] = ranking[j_rank]
                ranking[j_rank] = tmp

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] /= scale_value

        if info == NULL:
            self.finalize_delta_c(info_)

    cdef void delta_sample_c(self,
                             INT_t i,
                             INT_t offset,
                             INT_t n_documents,
                             INT_t *document_ranks,
                             INT_t *relevance_scores,
                             DOUBLE_t *document_weights,
                             INT_t nnz_documents,
                             DOUBLE_t scale_value,
                             DOUBLE_t *out,
                             void *info=NULL) nogil:
        '''
        A sampling variation of the `self.delta_c` method.
        '''
        cdef:
            INT_t j, k, tmp, i_rank, j_rank, cutoff
            INT_t *ranking = NULL
            DOUBLE_t before = 0.0, after = 0.0
            CTRDeltaInfo *info_ = NULL

        if n_documents == 0 or nnz_documents == 0:
            return

        # This should happen only when the documents
        # have the same relevance scores.
        if scale_value == 0.0:
            for j in range(n_documents - offset):
                out[j] = 0.0
            return

        if info == NULL:
            info_ = <CTRDeltaInfo *> self.prepare_delta_c(document_ranks,
                                                          relevance_scores,
                                                          document_weights,
                                                          n_documents,
                                                          scale_value)
        else:
            info_ = <CTRDeltaInfo *> info

        if info_ == NULL:
            # FIXME: This should be reported to the caller!!!
            return

        ranking = info_.r
        cutoff = nnz_documents

        if self.cutoff >= 0 and cutoff > self.cutoff:
            cutoff = self.cutoff

        # Get the number of clicks on the original list.
        before = 0.0
        for j in range(self.n_impressions):
            if self.click_model.get_clicks_c(ranking, cutoff,
                                             relevance_scores) > 0:
                before += 1.0

        # Estimated CTR of the original list.
        before /= self.n_impressions

        i_rank = document_ranks[i]

        for j in range(offset, n_documents):
            j_rank = document_ranks[j]

            if i_rank < cutoff or j_rank < cutoff:
                # Swap document i and j in the ranking...
                tmp = ranking[i_rank]
                ranking[i_rank] = ranking[j_rank]
                ranking[j_rank] = tmp

                # ... and compute (estimate) its clickthrough rate.
                after = 0.0
                for k in range(self.n_impressions):
                    if self.click_model.get_clicks_c(ranking, cutoff,
                                                     relevance_scores) > 0:
                        after += 1.0
                after /= self.n_impressions

                # Compute the (absolute) change in the metric...
                out[j - offset] = fabs(before - after)

                # ... and restore the original ranking for the next iteration.
                tmp = ranking[i_rank]
                ranking[i_rank] = ranking[j_rank]
                ranking[j_rank] = tmp

        if scale_value != 1.0:
            for j in range(n_documents - offset):
                out[j] /= scale_value

        if info == NULL:
            self.finalize_delta_c(info_)

    cdef void finalize_delta_c(self, void *info) nogil:
        if info != NULL:
            free((<CTRDeltaInfo *> info).r)
            free(info)

# =============================================================================
# Kendall Tau Distance
# =============================================================================

cdef class KendallTau:
    cdef INT_t    *mapping        # Buffer for remapping document IDs to 0, 1, 2, ...
    cdef DOUBLE_t *fenwick        # Fenwick tree for fast computation of weighted inversions.
    cdef DOUBLE_t *weights        # The position weights.
    cdef int       size           # The size of the internal arrays.
    cdef object    weights_func   # The Python function computing the weight of a given position.

    def __cinit__(self, weights, capacity=1024):
        '''
        Creates Kendall Tau distance metric.

        Parameters
        ----------
        weights : function
            A non-decreasing function of one integer
            parameter `i`, which returns the weight
            of a document at position `i` (0-based).
            It has to return a single float number.

            Consider this DCG discount weights,
            for example:

                weights(i): -1 / log2(i + 2)

            Remember that the parameter i is 0-based!

        capacity: int, optional (default is 1024)
            The initial capacity of the array for precomputed
            weight values.
        '''
        self.mapping = NULL
        self.fenwick = NULL
        self.weights = NULL
        self.size = 0
        self.weights_func = weights
        # Initialize the internal arrays.
        self.inflate_arrays()

    def __dealloc__(self):
        '''
        Free the allocated memory for internal arrays.
        '''
        free(self.mapping)
        free(self.fenwick)
        free(self.weights)

    def __reduce__(self):
        return (KendallTau, (self.weights_func,), self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int inflate_arrays(self, capacity=-1):
        '''
        Increase the capacity of the internal arrays to
        the given capacity.

        As the name of the function suggests, if `capacity`
        is smaller than the current capacity of the internal
        arrays, nothing happens.

        Parameters
        ----------
        capacity : int, optional (default is -1)
            The new capacity of the internal arrays. If -1
            is given the capacity of the internal arrays
            will be doubled.

        Returns
        -------
        code: int
            -1 on failure, 0 on success.
        '''
        cdef int i
        cdef void * ptr

        if capacity <= self.size and self.mapping != NULL:
            return 0

        if capacity <= -1:
            if self.size == 0:
                # Initial capacity.
                capacity = 1024
            else:
                # Double the current capacity.
                capacity = 2 * self.size

        # Because documents not appearing in both lists
        # are treated as if they were sitting at the
        # first position following the end of the lists.
        capacity += 1

        # Allocate mapping array.
        #########################
        ptr = realloc(self.mapping, capacity * sizeof(INT_t))

        if ptr == NULL:
            return -1

        self.mapping = <INT_t *> ptr

        # Initialize the new elements to -1.
        memset(<void *>(self.mapping + self.size), -1,
               (capacity - self.size) * sizeof(INT_t))

        # Allocate fenwick array.
        #########################
        ptr = realloc(self.fenwick, capacity * sizeof(DOUBLE_t))

        if ptr == NULL:
            return -1

        self.fenwick = <DOUBLE_t *> ptr

        # Initialize the new elements to 0.
        memset(<void *>(self.fenwick + self.size), 0,
               (capacity - self.size) * sizeof(DOUBLE_t))

        # Allocate weights array.
        #########################
        ptr = realloc(self.weights, capacity * sizeof(DOUBLE_t))

        if ptr == NULL:
            return -1

        self.weights = <DOUBLE_t *> ptr

        # Initialize the values of new weights using `self.weights_func`.
        for i in range(self.size, capacity):
            self.weights[i] = self.weights_func(i)

        self.size = capacity
        return 0

    def evaluate(self, X, check_input=True):
        '''
        Computes the Kendall Tau distance between the given
        list X and its ascendingly sorted version.
        '''
        cdef int size
        cdef DOUBLE_t tau

        if check_input:
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype='int32', order='C')

            if X.ndim != 1:
                raise ValueError('X is not one dimensional.')

            if X.dtype != np.int32 or not X.flags.c_contiguous:
                X = np.ascontiguousarray(X, dtype='int32')

        cdef np.ndarray[INT_t, ndim=1] Y = np.sort(X)

        # +1 in case of 0-based permutations.
        size = max(max(X), max(Y)) + 1

        # This may cause trouble for huge document IDs!
        if self.inflate_arrays(size) != 0:
            raise MemoryError('Cannot allocate %d bytes for internal arrays.'
                              % (sizeof(DOUBLE_t) * size))

        cdef INT_t *x =  <INT_t *> np.PyArray_DATA(X)
        cdef INT_t *y =  <INT_t *> np.PyArray_DATA(Y)

        size = min(X.shape[0], Y.shape[0])

        with nogil:
            tau = self.kendall_tau(x, y, size)

        return tau


    def distance(self, X, Y, check_input=True):
        '''
        Computes the Kendall Tau distance between the given
        lists X and Y.

        X and Y does not necessarily need to contain the same
        set of numbers. In case the numbers differ it is assumed
        that the lists are prefixes of longer lists, which
        were cutoff. In that matter, the lists does not even
        have to be of the same length, if that is the case,
        the minimum length of the two lists is considered.

        If `check_input` is True, X and Y can be lists/iterables
        of integer numbers, these arrays will be converted to
        numpy arrays with `numpy.int32` dtype.

        If `check_input` is False you need to make sure that
        X and Y are numpy arrays with `numpy.int32` dtype,
        unless you want to suffer severe consequences.
        '''
        cdef int size
        cdef DOUBLE_t tau

        if check_input:
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype='int32', order='C')

            if X.ndim != 1:
                raise ValueError('X is not one dimensional.')

            if X.dtype != np.int32 or not X.flags.c_contiguous:
                X = np.ascontiguousarray(X, dtype='int32')

            if not isinstance(Y, np.ndarray):
                Y = np.array(Y, dtype='int32', order='C')

            if Y.ndim != 1:
                raise ValueError('Y is not one dimensional.')

            if Y.dtype != np.int32 or not Y.flags.c_contiguous:
                Y = np.ascontiguousarray(Y, dtype='int32')

        # +1 in case of 0-based permutations.
        size = max(max(X), max(Y)) + 1

        # This may cause trouble for huge document IDs!
        if self.inflate_arrays(size) != 0:
            raise MemoryError('Cannot allocate %d bytes for internal arrays.'
                              % (sizeof(DOUBLE_t) * size))

        cdef INT_t *x =  <INT_t *> np.PyArray_DATA(X)
        cdef INT_t *y =  <INT_t *> np.PyArray_DATA(Y)

        size = min(X.shape[0], Y.shape[0])

        with nogil:
            tau = self.kendall_tau(x, y, size)

        return tau

    cdef DOUBLE_t kendall_tau(self, INT_t *X, INT_t *Y, int size) nogil:
        return self.kendall_tau_fenwick(X, Y, size)

    cdef inline DOUBLE_t kendall_tau_array(self, INT_t *X, INT_t *Y, int size) nogil:
        '''
        Computes Kendall Tau distance between X and Y using a simple array.
        This variant should be prefarable in case of short lists.
        '''
        cdef int i, j
        cdef double tau = 0.0

        for i in range(size):
            self.mapping[X[i]] = i

        # Process documents of Y.
        for j in range(size):
            i = self.mapping[Y[j]]
            # The document in Y that is not in X is treated
            # as if it was the first document following the
            # end of list X.
            tau += self._update_array(i if i >= 0 else size, j, size)
            if i >= 0:
                # Offset documents that appear in both lists.
                # This becomes useful for finding documents
                # that appeared only in X (see below).
                self.mapping[Y[j]] += size

        # Process documents of X that does not appear in Y.
        for j in range(size):
            i = self.mapping[X[j]]
            # j >= size ==> X[i] is in Y, we need to
            # clear it from the array such that it
            # will not interfere with calculation
            # of inversions for X[i]'s that are not
            # in Y.
            if i >= size:
                self._restore_array(i - size, size)
                # Offset the documents back again
                # for restoring the arrays.
                self.mapping[X[j]] -= size
            else:
                tau += self._get_array(i, size)

        # Restore the internal arrays.
        for j in range(size):
            i = self.mapping[Y[j]]
            # Restore the array for documents appearing
            # only in Y. These documents are put to the
            # same position, hence the restoration can
            # be called only once.
            if i < 0:
                self._restore_array(size, size)
                break

        # Finish the restoration of the arrays
        # by clearing the mapping.
        for i in range(size):
            self.mapping[X[i]] = -1

        return tau

    cdef inline DOUBLE_t _update_array(self, int i, int sigma, int size) nogil:
        '''
        Add a document at position `i` and `sigma` in respective lists
        X and Y into an array and compute the weighted number of
        inversions the document is with all previously added documents.

        Parameters
        ----------
        i : int
            The position of the document in X, or -1
            if it is not there.

        sigma : int
            The position of the document in Y.

        size: int
            The length of the document lists.

        Return
        ------
        tau: float
            The weighted number of inversions of the document
            with all the previously processed documents.
        '''
        cdef DOUBLE_t weight, tau = 0.0

        if i == sigma:
            weight = 1.0 # No displacement.
        else:
            # The weight of "bubbling" document from position
            # i to position sigma.
            weight = self.weights[i] - self.weights[sigma]
            # The average weight (denominator makes the weight
            # always positive).
            weight /= i - sigma

        sigma = size - i

        # Update the array.
        self.fenwick[sigma] += weight

        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        sigma -= 1
        while sigma >= 0:
            tau += self.fenwick[sigma] * weight
            sigma -= 1

        return tau

    cdef inline DOUBLE_t _get_array(self, int i, int size) nogil:
        '''
        Return the weighted number of invertions for i-th document
        of X, which do not appear in list Y.
        '''
        cdef DOUBLE_t weight, tau = 0.0

        # The weight of "bubbling" document
        # from position i to the first position
        # beyond the end of the list.
        weight = self.weights[i] - self.weights[size]

        # The average weight (denominator makes
        # the weight always positive).
        weight /= i - size

        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        while size >= 0:
            tau += self.fenwick[size] * weight
            size -= 1

        return tau

    cdef inline void _restore_array(self, int i, int size) nogil:
        '''
        Remove the weights at position `size - i` from the array.
        '''
        self.fenwick[size - i] = 0.0

    cdef inline DOUBLE_t kendall_tau_fenwick(self, INT_t *X, INT_t *Y, int size) nogil:
        '''
        Computes Kendall Tau distance between X and Y using a simple array.
        This variant should be prefarable in case of short lists.
        '''
        cdef int i, j
        cdef double tau = 0.0

        for i in range(size):
            self.mapping[X[i]] = i

        # Process documents of Y.
        for j in range(size):
            i = self.mapping[Y[j]]
            # The document in Y that is not in X is treated
            # as if it was the first document following the
            # end of list X.
            tau += self._update_fenwick(i if i >= 0 else size, j, size)
            if i >= 0:
                # Offset documents that appear in both lists.
                # This becomes useful for finding documents
                # that appeared only in X (see below).
                self.mapping[Y[j]] += size

        # Process documents of X that does not appear in Y.
        for j in range(size):
            i = self.mapping[X[j]]
            # j >= size ==> X[i] is in Y, we need to
            # clear it from the array such that it
            # will not interfere with calculation
            # of inversions for X[i]'s that are not
            # in Y.
            if i >= size:
                self._restore_fenwick(i - size, size)
                # Offset the documents back again
                # for restoring the arrays.
                self.mapping[X[j]] -= size
            else:
                tau += self._get_fenwick(i, size)

        # Restore the internal arrays.
        for j in range(size):
            i = self.mapping[Y[j]]
            # Restore the array for documents appearing
            # only in Y. These documents are put to the
            # same position, hence the restoration can
            # be called only once.
            if i < 0:
                self._restore_fenwick(size, size)
                break

        # Finish the restoration of the arrays
        # by clearing the mapping.
        for i in range(size):
            self.mapping[X[i]] = -1

        return tau


    cdef inline DOUBLE_t _update_fenwick(self, int i, int sigma, int size) nogil:
        '''
        Insert the weight of a document with displacement |i - sigma|
        into the Fenwick tree and compute the weighted number of invertions
        the document is in with all previously inserted documents.
        '''
        cdef DOUBLE_t weight, tau = 0.0

        if i == sigma:
            weight = 1.0 # No displacement.
        else:
            # The weight of "bubbling" document from position
            # i to position sigma.
            weight = self.weights[i] - self.weights[sigma]
            # The average weight (denominator makes the weight
            # always positive).
            weight /= i - sigma

        sigma = size - i

        if sigma != 0:
            tau += self.fenwick[0] * weight

        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        while sigma > 0:
            tau += self.fenwick[sigma] * weight
            sigma -= sigma & -sigma

        # Invert the indexing.
        sigma = size - i

        # Update the Fenwick tree.
        if sigma == 0:
            # Document below cutoff.
            self.fenwick[0] += weight
        else:
            # Update the Fenwick tree.
            while sigma <= size:
                self.fenwick[sigma] += weight
                sigma += sigma & -sigma

        return tau


    cdef inline DOUBLE_t _get_fenwick(self, int i, int size) nogil:
        '''
        Return the weighted number of invertions for i-th document
        of X, which do not appear in list Y.
        '''
        cdef DOUBLE_t weight, tau = 0.0

        # The weight of "bubbling" document
        # from position i to the first position
        # beyond the end of the list.
        weight = self.weights[i] - self.weights[size]

        # The average weight (denominator makes
        # the weight always positive).
        weight /= i - size

        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        while size > 0:
            tau += self.fenwick[size] * weight
            size -= size & -size

        tau += self.fenwick[0] * weight

        return tau


    cdef inline void _restore_fenwick(self, int i, int size) nogil:
        '''
        Remove the weight at position `size - i` from the Fenwick tree.
        '''
        cdef int j, k
        cdef DOUBLE_t weight

        # Invert the indexing.
        k = size - i

        # Document below cutoff.
        if k == 0:
            self.fenwick[k] = 0.0
        else:
            # Need to find the weight of the document first.
            weight = self.fenwick[k]

            j = k - (k & -k)
            k -= 1

            while k > j:
                weight -= self.fenwick[k]
                k -= k & -k

            # Remove the weight from the Fenwick tree.
            i = size - i
            while i <= size:
                self.fenwick[i] -= weight
                i += i & -i
