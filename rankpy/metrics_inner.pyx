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

import numpy as np
cimport numpy as np
np.import_array()

from cython cimport view

from utils_inner cimport DOUBLE_t, INT_t
from utils_inner cimport ranksort_relevance_labels_ext

from numpy import float64 as DOUBLE


cdef inline int int_min(INT_t a, INT_t b):
    return b if b < a else a


def compute_delta_dcg(DOUBLE_t[:] gain, DOUBLE_t[:] discount, INT_t cutoff, INT_t i, INT_t offset, INT_t[:] document_ranks, INT_t[:] relevance_scores, DOUBLE_t scale, DOUBLE_t[:] out):
    cdef:
        INT_t    j
        DOUBLE_t i_relevance_score, i_position_discount
        bint     i_above_cutoff, j_above_cutoff

    i_relevance_score   = gain[relevance_scores[i]]
    i_position_discount = discount[document_ranks[i]]
    i_above_cutoff      = (document_ranks[i] < cutoff)

    # Does the document 'i' influences the evaluation (at all)?.
    if i_above_cutoff:
        for j in range(out.shape[0]):
            out[j] = -i_relevance_score / i_position_discount
    else:
        for j in range(out.shape[0]):
            out[j] = 0.0

    for j in range(offset, document_ranks.shape[0]):
        j_above_cutoff = (document_ranks[j] < cutoff)

        if j_above_cutoff:
            out[j - offset] += (i_relevance_score - gain[relevance_scores[j]]) / discount[document_ranks[j]]

        if i_above_cutoff:
            out[j - offset] += gain[relevance_scores[j]] / i_position_discount

    if scale != 1.0:
        for j in range(out.shape[0]):
            out[j] /= scale


def compute_dcg_metric(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] queries_offsets, DOUBLE_t[:] gains, DOUBLE_t[:] discounts, INT_t cutoff):
    cdef:
        INT_t i, j, n_queries = queries_offsets.shape[0] - 1
        INT_t[:] ranked_labels = np.empty_like(labels)
        DOUBLE_t performance = 0.0

    ranksort_relevance_labels_ext(scores, labels, queries_offsets, ranked_labels)

    for i in range(n_queries):
        for j in range(queries_offsets[i], queries_offsets[i + 1] if cutoff < 0 else int_min(queries_offsets[i] + cutoff, queries_offsets[i + 1])):
            performance += gains[ranked_labels[j]] / discounts[j - queries_offsets[i]]

    return performance / n_queries


cpdef compute_ideal_dcg_metric_per_query(INT_t[:] sorted_labels, INT_t[:] queries_offsets, DOUBLE_t[:] gains, DOUBLE_t[:] discounts, INT_t cutoff):
    cdef:
        INT_t i, j, n_queries = queries_offsets.shape[0] - 1
        np.ndarray[DOUBLE_t, ndim=1] ideal_dcg = np.zeros((n_queries,), dtype=DOUBLE)

    for i in range(n_queries):
        for j in range(queries_offsets[i], queries_offsets[i + 1] if cutoff < 0 else int_min(queries_offsets[i] + cutoff, queries_offsets[i + 1])):
            ideal_dcg[i] += gains[sorted_labels[j]] / discounts[j - queries_offsets[i]]

    return ideal_dcg


def compute_ndcg_metric(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] queries_offsets, DOUBLE_t[:] gains, DOUBLE_t[:] discounts, INT_t cutoff, DOUBLE_t[:] scale=None):
    cdef:
        INT_t i, j, n_queries = queries_offsets.shape[0] - 1
        INT_t[:] ranked_labels = np.empty_like(labels)
        DOUBLE_t query_ndcg, performance = 0.0

    if scale is None:
        scale = compute_ideal_dcg_metric_per_query(labels, queries_offsets, gains, discounts, cutoff)

    ranksort_relevance_labels_ext(scores, labels, queries_offsets, ranked_labels)

    for i in range(n_queries):
        query_ndcg = 0.0
        for j in range(queries_offsets[i], queries_offsets[i + 1] if cutoff < 0 else int_min(queries_offsets[i] + cutoff, queries_offsets[i + 1])):
            query_ndcg += gains[ranked_labels[j]] / discounts[j - queries_offsets[i]]
        if scale[i] > 0:
            query_ndcg /= scale[i]
        performance += query_ndcg

    return performance / n_queries
