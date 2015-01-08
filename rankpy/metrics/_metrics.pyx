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

from libc.stdlib cimport calloc, free
from libc.math cimport log2

from ._utils cimport ranksort_relevance_scores_queries_c


# =============================================================================
# Types, constants, inline and helper functions
# =============================================================================


cdef inline INT_t imin(INT_t a, INT_t b) nogil:
    return b if b < a else a


cdef inline DOUBLE_t fabs(DOUBLE_t a) nogil:
    return -a if a < 0 else a


# =============================================================================
# Metric
# =============================================================================


cdef class Metric:
    '''
    The interface for an information retrieval evaluation metric.
    '''

    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance, INT_t maximum_documents):
        '''
        Initialize the metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        All these values should allow the metric object to pre-allocate
        and precompute `something`, which may help it to evaluate
        the metric for queries faster.

        cutoff: integer
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance: integer
            The maximum relevance score a document can have.

        maximum_documents: integer
            The maximum number of documents a query can have.
        '''
        self.cutoff = cutoff

        if cutoff == 0:
            raise ValueError('cutoff has to be positive integer or (-1) but 0 was given')


    cpdef evaluate_ranking(self, INT_t[::1] ranking, INT_t[::1] relevance_scores, DOUBLE_t scale_value):
        '''
        Evaluate the metric on the specified document ranking.

        Parameters:
        -----------
        ranking: array of integers, shape = (n_documents,)
            Specify the list of ranked documents.

        relevance_scores: array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scale_value: double
            'Optional' parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values. Specific metric
            implementations, such as NDCG, may use this parameter to speed up
            their computation.
        '''
        pass


    cpdef evaluate(self, INT_t[::1] ranked_relevance_scores, DOUBLE_t scale_value):
        '''
        Evaluate the metric on the specified ranked list of document relevance scores.

        Parameters:
        -----------
        ranked_relevance_scores: array of integers, shape = (n_documents,)
            Specify list of relevance scores.

        scale_value: double, shape = (n_documents,)
            'Optional' parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values. Specific metric
            implementations, such as NDCG, may use this parameter to speed up
            their computation.
        '''
        pass


    cpdef evaluate_queries(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] scores, DOUBLE_t[::1] scale_values, DOUBLE_t[::1] out):
        '''
        Evaluate the metric on the specified queries. The relevance scores and
        ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and  
        `scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.
        
        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scores: array, shape=(n_documents,)
            Specify the ranking score for each document.

        scale_values: array, shape=(n_queries,), optional
            'Optional' parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values. Specific metric
            implementations, such as NDCG, may use this parameter to speed
            up their computation.

        out: array, shape=(n_documents,), optional
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        pass


    cpdef delta(self, INT_t i, INT_t offset, INT_t[::1] document_ranks, INT_t[::1] relevance_scores, DOUBLE_t scale_value, DOUBLE_t[::1] out):
        '''
        Compute the change in the metric caused by swapping document `i` with every
        document `offset`, `offset + 1`, ...

        The relevance score and document rank of document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters:
        -----------
        i: integer
            The index of the one document that is being swapped with all
            the others.

        offset: integer
            The start index of the sequence of documents that are
            being swapped.

        document_ranks: array of integers
            Specify the rank for each document.

        relevance_scores: array of integers
            Specify the relevance score for each document.

        scale_value: double, shape = (n_documents,)
            'Optional' parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values. Specific metric
            implementations, such as NDCG, may use this parameter to speed up
            their computation.

        out: array of doubles
            The output array. The array size is expected to be at least as big
            as the the number of document pairs being swapped, which should be
            `len(document_ranks) - offset`.
        '''
        pass


    cdef void delta_c(self, INT_t i, INT_t offset, INT_t n_documents, INT_t *document_ranks, INT_t *relevance_scores, DOUBLE_t scale_value, DOUBLE_t *out) nogil:
        '''
        See description of self.delta(...) method.
        '''
        pass


    cpdef evaluate_queries_ideal(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] scale_values):
        '''
        Compute the ideal metric value for every one of the specified queries.
        The relevance scores of documents, which belong to query `i`, should be
        stored in `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` in
        descending order.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scale_values: output array of doubles, shape=(n_queries,)
            Output array for the ideal metric value of each query.
        '''
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

    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance, INT_t maximum_documents):
        '''
        Initialize the metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        These values are used to pre-allocate the gain and discount
        for document relevance scores and ranks, respectively.

        cutoff: integer
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance: integer
            The maximum relevance score a document can have.

        maximum_documents: integer
            The maximum number of documents a query can have.
        '''
        cdef INT_t i
        cdef DOUBLE_t gain

        self.gain_cache = NULL
        self.discount_cache = NULL

        if maximum_relevance <= 0:
            maximum_relevance = 8

        if maximum_documents <= 0:
            maximum_documents = 4096

        self.maximum_relevance = maximum_relevance
        self.maximum_documents = maximum_documents

        self.gain_cache = <DOUBLE_t*> calloc(self.maximum_relevance + 1, sizeof(DOUBLE_t))
        self.discount_cache = <DOUBLE_t*> calloc(self.maximum_documents, sizeof(DOUBLE_t))

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
        return (DiscountedCumulativeGain, (self.cutoff, self.maximum_relevance, self.maximum_documents))


    cpdef evaluate_ranking(self, INT_t[::1] ranking, INT_t[::1] relevance_scores, DOUBLE_t scale_value):
        '''
        Evaluate the metric on the specified document ranking.

        Parameters:
        -----------
        ranking: array of integers, shape = (n_documents,)
            Specify the list of ranked documents.

        relevance_scores: array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scale_value: double
            Should be 1.0.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result

        # Should we worry about precision?
        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranking.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            for i in range(imin(cutoff, n_documents)):
                result += self.gain_cache[relevance_scores[ranking[i]]] / self.discount_cache[i]
            result /= scale_value

        return result


    cpdef evaluate(self, INT_t[::1] ranked_relevance_scores, DOUBLE_t scale_value):
        '''
        Evaluate the metric on the specified ranked list of document relevance scores.

        Parameters:
        -----------
        ranked_relevance_scores: array of integers, shape = (n_documents,)
            Specify list of relevance scores.

        scale_value: double
            Should be 1.0.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result

        # Should we worry about precision?
        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranked_relevance_scores.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            for i in range(imin(cutoff, n_documents)):
                result += self.gain_cache[ranked_relevance_scores[i]] / self.discount_cache[i]
            result /= scale_value

        return result


    cpdef evaluate_queries(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] ranking_scores, DOUBLE_t[::1] scale_values, DOUBLE_t[::1] out):
        '''
        Evaluate the DCG metric on the specified queries. The relevance scores and
        ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_queries + 1,)
            Specify the relevance score for each document.

        ranking_scores: array, shape=(n_queries,)
            Specify the ranking score for each document.

        scale_values: array, shape=(n_queries,), optional
            Should be None (defaults to all 1s).

        out: array, shape=(n_documents,), optional
            If not None, it will be filled with the metric value
            for each individual query.
        '''    
        cdef:
            INT_t i, j, n_queries, n_documents
            INT_t *ranked_relevance_scores
            DOUBLE_t result, qresult

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            
            ranked_relevance_scores = <INT_t*> calloc(n_documents, sizeof(INT_t))

            ranksort_relevance_scores_queries_c(&query_indptr[0], n_queries, &ranking_scores[0], &relevance_scores[0], ranked_relevance_scores)

            result = 0.0

            for i in range(n_queries):
                n_documents = query_indptr[i + 1] - query_indptr[i]
                cutoff = n_documents if self.cutoff < 0 else imin(self.cutoff, n_documents)

                qresult = 0.0
                for j in range(cutoff):
                    qresult += self.gain_cache[ranked_relevance_scores[query_indptr[i] + j]] / self.discount_cache[j]

                if scale_values is not None:
                    # Should we worry about precision?
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            result /= n_queries

            free(ranked_relevance_scores)
            
        return result


    cpdef delta(self, INT_t i, INT_t offset, INT_t[::1] document_ranks, INT_t[::1] relevance_scores, DOUBLE_t scale_value, DOUBLE_t[::1] out):
        '''
        Compute the change in the metric caused by swapping document `i` with every
        document `offset`, `offset + 1`, ...

        The relevance score and document rank of document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters:
        -----------
        i: integer
            The index of the one document that is being swapped with all
            the others.

        offset: integer
            The start index of the sequence of documents that are
            being swapped.

        document_ranks: array of integers
            Specify the rank for each document.

        relevance_scores: array of integers
            Specify the relevance score for each document.

        scale_value: double, shape = (n_documents,)
            Should be 1.0.

        out: array of doubles
            The output array. The array size is expected to be at least as big
            as the the number of document pairs being swapped, which should be
            `len(document_ranks) - offset`.
        '''
        with nogil:
            self.delta_c(i, offset, document_ranks.shape[0], &document_ranks[0], &relevance_scores[0], scale_value, &out[0])


    cpdef evaluate_queries_ideal(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] ideal_values):
        '''
        Compute the ideal DCG metric value for every one of the specified queries.

        The relevance scores of documents, which belong to query `i`, should be
        stored in `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` in
        descending order.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document. It is expected
            that these values are sorted in descending order.

        ideal_values: output array of doubles, shape=(n_queries,)
            Output array for the ideal metric value of each query.
        '''
        cdef INT_t i, j, n_documents, cutoff

        with nogil:
            for i in range(query_indptr.shape[0] - 1):
                ideal_values[i] = 0.0
                n_documents = query_indptr[i + 1] - query_indptr[i]
                cutoff = n_documents if self.cutoff < 0 else imin(self.cutoff, n_documents)
                for j in range(cutoff):
                    ideal_values[i] += self.gain_cache[relevance_scores[query_indptr[i] + j]] / self.discount_cache[j]


    cdef void delta_c(self, INT_t i, INT_t offset, INT_t n_documents, INT_t *document_ranks, INT_t *relevance_scores, DOUBLE_t scale_value, DOUBLE_t *out) nogil:
        '''
        See description of self.delta(...) method.
        '''
        cdef:
            INT_t j, n_swapped_document_pairs, cutoff
            DOUBLE_t i_relevance_score, i_position_discount
            bint i_above_cutoff, j_above_cutoff

        n_swapped_document_pairs = n_documents - offset
        cutoff  = n_documents if self.cutoff < 0 else self.cutoff

        i_relevance_score = self.gain_cache[relevance_scores[i]]
        i_position_discount = self.discount_cache[document_ranks[i]]
        i_above_cutoff = (document_ranks[i] < cutoff)

        # Does the document 'i' influences the evaluation (at all)?
        if i_above_cutoff:
            for j in range(n_swapped_document_pairs):
                out[j] = -i_relevance_score / i_position_discount
        else:
            for j in range(n_swapped_document_pairs):
                out[j] = 0.0

        for j in range(offset, n_documents):
            j_above_cutoff = (document_ranks[j] < cutoff)

            if j_above_cutoff:
                out[j - offset] += (i_relevance_score - self.gain_cache[relevance_scores[j]]) / self.discount_cache[document_ranks[j]]

            if i_above_cutoff:
                out[j - offset] += self.gain_cache[relevance_scores[j]] / i_position_discount

        if scale_value != 1.0:
            # Should we worry about precision?
            if scale_value == 0.0:
                for j in range(n_swapped_document_pairs):
                    out[j] = 0.0
            else:
                for j in range(n_swapped_document_pairs):
                    out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_swapped_document_pairs):
                out[j] = fabs(out[j])
