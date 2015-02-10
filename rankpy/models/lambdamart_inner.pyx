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

from libc.stdlib cimport calloc, free
from libc.string cimport memset
from libc.math cimport exp

from ..metrics._utils cimport INT_t
from ..metrics._utils cimport DOUBLE_t
from ..metrics._utils cimport argranksort_queries_c

from ..metrics._metrics cimport Metric


def parallel_compute_lambdas_and_weights(INT_t qstart, INT_t qend, INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores,
                                         INT_t[::1] relevance_scores, INT_t[:, ::1] relevance_strides, Metric metric,
                                         DOUBLE_t[::1] scale_values, DOUBLE_t[:, ::1] influences, INT_t[::1] leaves_idx,
                                         DOUBLE_t[::1] output_lambdas, DOUBLE_t[::1] output_weights):
    '''
    Helper function computing pseudo-responses (`lambdas`) and 'optimal'
    gradient steps (`weights`) for the documents belonging to the specified
    queries. This method is suitable for doing some heavy computation
    in a multithreading backend (releases GIL).

    Parameters:
    -----------
    qstart: integer
        Start index of the query for which documents the lambas
        and weights will be computed.

    qend: integer
        End index of the query (eclusive) for which documents
        the lambas and weights will be computed.

    query_indptr: array, shape = (n_queries + 1)
        The query index pointer array.

    ranking_scores: array, shape = (n_documents,)
        The ranking scores of the documents.

    relevance_scores: array, shape = (n_documents,)
        The relevance scores of the documents.

    relevance_strides: array, shape = (n_query, n_relevance_scores)
        The array of index pointers for each query. `relevance_strides[i, s]`
        contains the index of the document (with respect to query `i`)
        which has lower relevance than `s`.

    metric: Metric cython backend
        The evaluation metric, for which the lambdas and weights
        are to be computed.

    scale_values: array, shape=(n_queries,), optional (default is None)
        The precomputed metric scale value for every query.

    influences: array, shape=(n_max_relevance, n_max_relevance) or None
        Used to keep track of (proportional) contribution from lambdas
        (force interpretation) of low relevant documents.

    leaves_idx: array, shape=(n_documents, n_leaf_nodes) or None:
        The indices of terminal nodes which the documents fall into.
        This parameter can be used to recompute lambdas and weights
        after regression tree is built.

    output_lambdas: array, shape=(n_documents,)
        Computed lambdas for every document.

    output_weights: array, shape=(n_documents,)
        Computed weights for every document.
    '''
    cdef:
        INT_t i, j, k, start,rstart, end, n_documents
        INT_t j_relevance_score, max_relevance
        INT_t *document_ranks = NULL
        DOUBLE_t *document_deltas = NULL
        DOUBLE_t *influence_by_relevance = NULL
        DOUBLE_t j_push_down, lambda_, weight, rho, scale

    with nogil:
        # Total number of documents to process.
        n_documents = query_indptr[qend] - query_indptr[qstart]

        # More than enough memory to hold what we want.
        document_ranks = <INT_t *> calloc(n_documents, sizeof(INT_t))
        document_deltas = <DOUBLE_t *> calloc(n_documents, sizeof(DOUBLE_t))

        if influences is not None:
            max_relevance = influences.shape[0]
            influence_by_relevance = <DOUBLE_t *> calloc(max_relevance, sizeof(DOUBLE_t))
        else:
            max_relevance = -1

        # Find the rank of each document with respect to the ranking scores over all queries.
        argranksort_queries_c(&query_indptr[0] + qstart, qend - qstart, &ranking_scores[0], document_ranks)

        # Clear output array for lambdas since we will be incrementing.
        memset(&output_lambdas[0] + query_indptr[qstart], 0, n_documents * sizeof(DOUBLE_t))

        # Clear output array for weights since we will be incrementing.
        memset(&output_weights[0] + query_indptr[qstart], 0, n_documents * sizeof(DOUBLE_t))

        # Loop through the queries and compute lambdas and weights for every document.
        for i in range(qstart, qend):
            start, end = query_indptr[i], query_indptr[i + 1]

            scale = 1.0 if scale_values is None else scale_values[i]

            # The number of documents of the current query.
            n_documents = end - start

            # Find the rank of each document with respect to the ranking scores.
            # argranksort_c(&ranking_scores[0] + start, document_ranks, n_documents)

            # Loop through the documents of the current query.
            for j in range(start, end):
                j_relevance_score = relevance_scores[j]

                # The smallest index of a document with a lower relevance score than document 'j'.
                rstart = relevance_strides[i, j_relevance_score]

                # Is there any document less relevant than document 'j'?
                if rstart >= end:
                    break

                # Compute the (absolute) changes in the metric caused by swapping document 'j' with all
                # documents 'k' (k >= rstart), which have lower relevance with respect to the query 'i'.
                metric.delta_c(j - start, rstart - start, n_documents, document_ranks + start - query_indptr[qstart], &relevance_scores[0] + start, scale, document_deltas)

                # Clear the influences for the current document.
                if max_relevance > 0:
                    memset(influence_by_relevance, 0, j_relevance_score * sizeof(DOUBLE_t))

                # Current forces pushing document 'j' down.
                j_push_down = output_lambdas[j]

                for k in range(rstart, end):
                    if leaves_idx is not None and leaves_idx[j] == leaves_idx[k]:
                        continue

                    rho = (<DOUBLE_t> 1.0) / ((<DOUBLE_t> 1.0) + (<DOUBLE_t> exp(ranking_scores[j] - ranking_scores[k])))

                    lambda_ = rho * document_deltas[k - rstart]
                    weight = (1 - rho) * lambda_

                    output_lambdas[j] += lambda_
                    output_lambdas[k] -= lambda_

                    output_weights[j] += weight
                    output_weights[k] += weight

                    if max_relevance > 0:
                        influence_by_relevance[relevance_scores[k]] += lambda_

                if max_relevance > 0:
                    for k in range(j_relevance_score):
                        if influence_by_relevance[k] <= output_lambdas[j]:
                            influences[k, j_relevance_score] += influence_by_relevance[k] / output_lambdas[j]
                        influences[j_relevance_score, k] += influence_by_relevance[k] / (output_lambdas[j] - 2 * j_push_down)

        free(document_ranks)
        free(document_deltas)
        free(influence_by_relevance)
