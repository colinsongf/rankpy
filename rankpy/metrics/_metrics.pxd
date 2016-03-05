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

cimport numpy as np
np.import_array()

ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_int32   INT_t
ctypedef np.npy_uint8   BOOL_t

# =============================================================================
# Metric
# =============================================================================


cdef class Metric:
    #
    # The interface for an information retrieval evaluation metric.
    #

    # Fields
    cdef public INT_t cutoff       # The metric cutoff threshold.
    cdef public unsigned int seed  # The RNG seed for random shuffling of
                                   # documents with the same ranking score.

    # =========================================================================
    # Python methods
    # =========================================================================
    cpdef evaluate_ranking(self,
                           INT_t[::1] document_ranks,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t scale_value=?,
                           DOUBLE_t query_weight=?)

    cpdef evaluate(self,
                   INT_t[::1] ranked_relevance_scores,
                   DOUBLE_t scale_value=?,
                   DOUBLE_t query_weight=?)

    cpdef evaluate_queries(self,
                           INT_t[::1] query_indptr,
                           INT_t[::1] relevance_scores,
                           DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values,
                           DOUBLE_t[::1] query_weights,
                           DOUBLE_t[::1] document_weights,
                           DOUBLE_t[::1] out)

    cpdef delta(self,
                INT_t i,
                INT_t offset,
                INT_t[::1] document_ranks,
                INT_t[::1] relevance_scores,
                DOUBLE_t[::1] document_weights,
                INT_t nnz_documents,
                DOUBLE_t scale_value,
                DOUBLE_t[::1] out)

    # =========================================================================
    # C methods
    # =========================================================================
    cdef void* prepare_delta_c(self,
                               INT_t *document_ranks,
                               INT_t *relevance_scores,
                               DOUBLE_t *document_weights,
                               INT_t n_documents,
                               DOUBLE_t scale_value,
                               void *info=?) nogil

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
                      void *info=?) nogil

    cdef void finalize_delta_c(self,
                               void *info) nogil