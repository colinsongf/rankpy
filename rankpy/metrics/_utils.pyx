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
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.


from cython cimport view

from libc.stdlib cimport calloc, free, rand, srand, qsort
from libc.time cimport time, time_t


# =============================================================================
# Global variables and structure declarations
# =============================================================================


# Global indicator that srand has been called.
cdef bint srand_called = 0

# Auxiliary document structure, used for sort-ing and
# argsort-ing.
cdef struct DOCUMENT_t:
    INT_t position      # The document position in the input list ('document ID').
    INT_t nonce         # Randomly generated number used to break ties in ranking scores.
    DOUBLE_t score      # The document ranking score.


# =============================================================================
# C function definitions
# =============================================================================


cdef int __compare(const void *a, const void *b) nogil:
    '''
    Compare function used in stdlib's qsort. The parameters are 2 instances
    of DOCUMENT_t, which are being sorted in descending order according to
    their ranking scores, i.e. the higher the ranking score the lower the
    resulting rank of the document.
    '''
    cdef DOUBLE_t diff = ((<DOCUMENT_t*>b)).score - ((<DOCUMENT_t*>a)).score
    if diff < 0:
        if diff > -1e-12: # "Equal to zero from left"
            return ((<DOCUMENT_t*>b)).nonce - ((<DOCUMENT_t*>a)).nonce
        else:
            return -1
    else:
        if diff < 1e-12: # "Equal to zero from right"
            return ((<DOCUMENT_t*>b)).nonce - ((<DOCUMENT_t*>a)).nonce
        else:
            return 1


cdef void __argranksort(DOUBLE_t *ranking_scores, DOCUMENT_t *documents, INT_t document_position_offset, INT_t n_documents) nogil:
    '''
    Auxiliary function for ranksort and argranksort functions.
    '''
    cdef INT_t i

    if not srand_called:
        srand(time(<time_t *>0))
        global srand_called
        srand_called = True

    for i in range(n_documents):
        documents[i].position = i + document_position_offset
        documents[i].nonce = rand()
        documents[i].score = ranking_scores[i]

    qsort(documents, n_documents, sizeof(DOCUMENT_t), __compare)


cdef void __argranksort_queries(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, DOCUMENT_t *documents) nogil:
    '''
    Auxiliary function for ranksort_queries and argranksort_queries functions.
    '''
    cdef INT_t i

    for i in range(n_queries):
        __argranksort(ranking_scores + query_indptr[i], documents + query_indptr[i] - query_indptr[0],
                      query_indptr[i], query_indptr[i + 1] - query_indptr[i])


cdef void argranksort_c(DOUBLE_t *ranking_scores, INT_t *ranks, INT_t n_documents) nogil:
    '''
    Return the rank position of the documents associated with the specified ranking_scores,
    i.e. `ranks[i]` is the position of the `ranking_scores[i]` within the sorted array
    (in descending order) of `ranking_scores`.
    '''
    cdef:
        INT_t i
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort(ranking_scores, documents, 0, n_documents)

    for i in range(n_documents):
        ranks[documents[i].position] = i

    free(documents)
    

cdef void argranksort_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *ranks) nogil:
    '''
    Return the rank position of the documents within the document list of
    specified queries, which is determined using the specified ranking scores.
    '''    
    cdef:
        INT_t i, j, r, n_documents = query_indptr[n_queries] - query_indptr[0]
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort_queries(query_indptr, n_queries, ranking_scores, documents)

    for i in range(n_queries):
        r = 0
        for j in range(query_indptr[i] - query_indptr[0], query_indptr[i + 1] - query_indptr[0]):
            ranks[documents[j].position - query_indptr[0]] = r
            r += 1

    free(documents)


cdef void ranksort_c(DOUBLE_t *ranking_scores, INT_t *ranking, INT_t n_documents) nogil:
    '''
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking[i]` identifies the ranking score which would be placed at i-th
    position within the sorted array (in descending order) of `ranking_scores`.
    '''
    cdef:
        INT_t i
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort(ranking_scores, documents, 0, n_documents)

    for i in range(n_documents):
        ranking[i] = documents[i].position

    free(documents)


cdef void ranksort_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *ranking) nogil:
    '''
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking_scores[ranking[i]]` will be the ranking score which would be placed
    at i-th position within the sorted array (in descending order) of `ranking_scores`.
    '''
    cdef:
        INT_t i, j, n_documents = query_indptr[n_queries] - query_indptr[0]
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort_queries(query_indptr, n_queries, ranking_scores, documents)

    for i in range(n_queries):
        for j in range(query_indptr[i] - query_indptr[0], query_indptr[i + 1] - query_indptr[0]):
            ranking[j] = documents[j].position - query_indptr[i]

    free(documents)


cdef void ranksort_relevance_scores_c(DOUBLE_t *ranking_scores, INT_t *relevance_scores, INT_t n_documents, INT_t *out) nogil:
    '''
    Rank the specified relevance scores according to the specified ranking scores.
    '''
    cdef:
        INT_t i
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort(ranking_scores, documents, 0, n_documents)

    for i in range(n_documents):
        out[i] = relevance_scores[documents[i].position]

    free(documents)


cdef void ranksort_relevance_scores_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *relevance_scores, INT_t *out) nogil:
    '''
    Rank the specified relevance scores according to the specified ranking
    scores with respect to the given queries.
    '''
    cdef:
        INT_t i, j, n_documents = query_indptr[n_queries] - query_indptr[0]
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort_queries(query_indptr, n_queries, ranking_scores, documents)

    for i in range(n_queries):
        for j in range(query_indptr[i] - query_indptr[0], query_indptr[i + 1] - query_indptr[0]):
            out[j] = relevance_scores[documents[j].position]

    free(documents)


# =============================================================================
# Python bindings for the C functions defined above.
# =============================================================================


cpdef argranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks):
    '''
    Return the rank position of the documents associated with the specified ranking_scores,
    i.e. `ranks[i]` is the position of the `ranking_scores[i]` within the sorted array
    (in descending order) of `ranking_scores`.
    '''
    with nogil:
        argranksort_c(&ranking_scores[0], &ranks[0], ranking_scores.shape[0])


cpdef argranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks):
    '''
    Return the rank position of the documents within the document list of
    specified queries, which is determined using the specified ranking scores.
    '''    
    with nogil:
        argranksort_queries_c(&query_indptr[0], query_indptr.shape[0] - 1, &ranking_scores[0], &ranks[0])


cpdef ranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking):
    '''
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking[i]` identifies the ranking score which would be placed at i-th
    position within the sorted array (in descending order) of `ranking_scores`.
    '''
    with nogil:
        ranksort_c(&ranking_scores[0], &ranking[0], ranking_scores.shape[0])


cpdef ranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking):
    '''
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking_scores[ranking[i]]` will be the ranking score which would be placed
    at i-th position within the sorted array (in descending order) of `ranking_scores`.
    '''
    with nogil:
        ranksort_queries_c(&query_indptr[0], query_indptr.shape[0] - 1, &ranking_scores[0], &ranking[0])


cpdef ranksort_relevance_scores(DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out):
    '''
    Rank the specified relevance scores according to the specified ranking scores.
    '''
    with nogil:
        ranksort_relevance_scores_c(&ranking_scores[0], &relevance_scores[0], ranking_scores.shape[0], &out[0])


cpdef ranksort_relevance_scores_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out):
    '''
    Rank the specified relevance scores according to the specified ranking
    scores with respect to the given queries.
    '''
    with nogil:
        ranksort_relevance_scores_queries_c(&query_indptr[0], query_indptr.shape[0] - 1, &ranking_scores[0], &relevance_scores[0], &out[0])
