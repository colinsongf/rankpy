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
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.

cimport numpy as np
import numpy as np

from cython cimport view

from numpy import float64 as DOUBLE

from libc.stdlib cimport malloc, free, rand, srand, qsort
from libc.time cimport time, time_t


cdef bint srand_called = 0


cdef struct DOCUMENT_t:
    INT_t position
    INT_t nonce
    DOUBLE_t score


cdef int __compare(const void *a, const void *b) nogil:
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


cdef void __argranksort(DOUBLE_t[:] scores, DOCUMENT_t * documents) nogil:
    cdef:
        INT_t i
        INT_t n_documents = scores.shape[0]

    if not srand_called:
        srand(time(<time_t *>0))
        global srand_called
        srand_called = True

    for i in range(n_documents):
        documents[i].position = i
        documents[i].nonce = rand()
        documents[i].score = scores[i]

    qsort(<void *> documents, n_documents, sizeof(DOCUMENT_t), __compare)
    

cpdef argranksort(DOUBLE_t[:] scores, INT_t[:] ranks):
    cdef:
        INT_t i, n_documents = scores.shape[0]
        DOCUMENT_t *documents = <DOCUMENT_t *>malloc(n_documents * sizeof(DOCUMENT_t))

    __argranksort(scores, documents)

    for i in range(n_documents):
        ranks[documents[i].position] = i

    free(documents)


cdef void __argranksort_ext(DOUBLE_t[:] scores, DOCUMENT_t * documents, INT_t[:] queries_offsets) nogil:
    cdef:
        INT_t i, n_documents = scores.shape[0], n_queries = queries_offsets.shape[0] - 1

    if not srand_called:
        srand(time(<time_t *>0))
        global srand_called
        srand_called = True

    for i in range(n_documents):
        documents[i].position = i
        documents[i].nonce = rand()
        documents[i].score = scores[i]

    for i in range(n_queries):
        qsort(<void *>(documents + queries_offsets[i]), queries_offsets[i + 1] - queries_offsets[i], sizeof(DOCUMENT_t), __compare)


cpdef argranksort_ext(DOUBLE_t[:] scores, INT_t[:] ranks, INT_t[:] queries_offsets):
    cdef:
        INT_t i, j, r, n_documents = scores.shape[0], n_queries = queries_offsets.shape[0] - 1
        DOCUMENT_t *documents = <DOCUMENT_t *>malloc(n_documents * sizeof(DOCUMENT_t))

    __argranksort_ext(scores, documents, queries_offsets)

    for i in range(n_queries):
        r = 0
        for j in range(queries_offsets[i], queries_offsets[i + 1]):
            ranks[documents[j].position] = r
            r += 1

    free(documents)


cpdef ranksort_relevance_labels(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] out):
    cdef:
        INT_t i, n_documents = scores.shape[0]
        DOCUMENT_t *documents = <DOCUMENT_t *>malloc(n_documents * sizeof(DOCUMENT_t))

    __argranksort(scores, documents)

    for i in range(n_documents):
        out[i] = labels[documents[i].position]

    free(documents)


cpdef ranksort_relevance_labels_ext(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] queries_offsets, INT_t[:] out):
    cdef:
        INT_t i, n_documents = scores.shape[0]
        DOCUMENT_t *documents = <DOCUMENT_t *>malloc(n_documents * sizeof(DOCUMENT_t))

    __argranksort_ext(scores, documents, queries_offsets)

    for i in range(n_documents):
        out[i] = labels[documents[i].position]

    free(documents)
