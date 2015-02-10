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

cimport numpy as np
np.import_array()

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t   INT_t


# =============================================================================
# C function definitions
# =============================================================================


cdef void argranksort_c(DOUBLE_t *ranking_scores, INT_t *ranks, INT_t n_documents) nogil
cdef void argranksort_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *ranks) nogil

cdef void ranksort_c(DOUBLE_t *ranking_scores, INT_t *ranking, INT_t n_documents) nogil
cdef void ranksort_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *ranking) nogil

cdef void ranksort_relevance_scores_c(DOUBLE_t *ranking_scores, INT_t *relevance_scores, INT_t n_documents, INT_t *out) nogil
cdef void ranksort_relevance_scores_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *relevance_scores, INT_t *out) nogil


# =============================================================================
# Python bindings for the C functions defined above.
# =============================================================================

cpdef set_seed(unsigned int seed)

cpdef argranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks)
cpdef argranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks)

cpdef ranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking)
cpdef ranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking)

cpdef ranksort_relevance_scores(DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out)
cpdef ranksort_relevance_scores_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out)
