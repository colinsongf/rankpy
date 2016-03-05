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

import numpy as np

from cython cimport view

from libc.stdlib cimport malloc, calloc, free

from libc.string cimport memset, memcpy

from libc.math cimport exp, log

from ..metrics._utils cimport INT_t
from ..metrics._utils cimport DOUBLE_t
from ..metrics._utils cimport get_seed
from ..metrics._utils cimport argranksort_queries_c
from ..metrics._utils cimport relevance_argsort_v1_c

from ..metrics._metrics cimport Metric

cdef DOUBLE_t EPSILON  = np.finfo('d').eps
cdef DOUBLE_t NaN = np.nan


def parallel_compute_lambdas_and_weights(INT_t qstart,
                                         INT_t qend,
                                         INT_t[::1] query_indptr,
                                         DOUBLE_t[::1] ranking_scores,
                                         INT_t[::1] relevance_scores,
                                         INT_t maximum_relevance,
                                         INT_t[:, ::1] relevance_strides,
                                         Metric metric,
                                         DOUBLE_t[::1] scale_values,
                                         DOUBLE_t[:, ::1] influences,
                                         INT_t[::1] leaves_idx,
                                         DOUBLE_t[::1] query_weights,
                                         DOUBLE_t[::1] document_weights,
                                         DOUBLE_t[::1] output_lambdas,
                                         DOUBLE_t[::1] output_weights,
                                         object random_state=None):
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

    maximum_relevance: int
        The maximum relevance score.

    relevance_strides: array, shape = (n_query, maximum_relevance + 1)
        The array of index pointers for each query. `relevance_strides[i, s]`
        contains the index of the document (with respect to query `i`)
        which has lower relevance than `s`.

    metric: Metric cython backend
        The evaluation metric, for which the lambdas and weights
        are to be computed.

    scale_values: array, shape=(n_queries,), optional (default is None)
        The precomputed metric scale value for every query.

    influences: array, shape=(maximum_relevance + 1, maximum_relevance + 1)
                or None
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

    random_state: RandomState instance
        Random number generator used for shuffling of documents
        with the same ranking score.

    Returns
    -------
    loss: float
        The LambdaMART loss of the rankings induced by the specified
        ranking scores.
    '''
    cdef unsigned int seed = get_seed(random_state)
    cdef DOUBLE_t *ranking_scores_copy
    cdef DOUBLE_t loss

    ranking_scores_copy = <DOUBLE_t *> malloc((query_indptr[qend] -
                                               query_indptr[qstart]) *
                                              sizeof(DOUBLE_t))

    if ranking_scores_copy == NULL:
        raise MemoryError()

    # Copy the ranking scores because documents with 0 weight will
    # have their scores set to NaN.
    memcpy(ranking_scores_copy, &ranking_scores[0] + query_indptr[qstart],
           (query_indptr[qend] - query_indptr[qstart]) * sizeof(DOUBLE_t))

    # Offset the pointer to make the following call treat the copy of
    # the ranking scores as the orignal.
    ranking_scores_copy -= query_indptr[qstart]

    loss = parallel_compute_lambdas_and_weights_c(
                qstart, qend, &query_indptr[0], ranking_scores_copy,
                &relevance_scores[0], maximum_relevance,
                NULL if relevance_strides is None else &relevance_strides[0, 0],
                metric, &scale_values[0],
                NULL if influences is None else &influences[0, 0],
                NULL if leaves_idx is None else &leaves_idx[0],
                &query_weights[0], &document_weights[0],
                &output_lambdas[0], &output_weights[0], seed)

    # Put the pointer back to the allocated memory...
    ranking_scores_copy += query_indptr[qstart]

    # ... and free it.
    free(ranking_scores_copy)

    return loss

cdef DOUBLE_t parallel_compute_lambdas_and_weights_c(
                INT_t qstart, INT_t qend, INT_t *query_indptr,
                DOUBLE_t *ranking_scores, INT_t *relevance_scores,
                INT_t maximum_relevance, INT_t *relevance_strides,
                Metric metric, DOUBLE_t *scale_values, DOUBLE_t *influences,
                INT_t *leaves_idx, DOUBLE_t *query_weights,
                DOUBLE_t *document_weights, DOUBLE_t *output_lambdas,
                DOUBLE_t *output_weights,
                unsigned int seed):
    '''
    The guts of `parallel_compute_lambdas_and_weights`.
    '''
    cdef:
        INT_t i, j, k, start, rstart, end, n_documents, j_relevance_score, rc
        INT_t *sort_indices = NULL
        INT_t *document_ranks = NULL
        INT_t *query_nnz_documents = NULL
        DOUBLE_t *document_deltas = NULL
        DOUBLE_t *influence_by_relevance = NULL
        DOUBLE_t j_push_down, lambda_, weight_, rho
        DOUBLE_t qweight, qscale, j_document_weight, k_document_weight
        DOUBLE_t qjk_weight, loss = 0.0
        bint resort = False
        void *deltas_info = NULL

    with nogil:
        # The maximal number of documents to process.
        n_documents = query_indptr[qend] - query_indptr[qstart]

        # More than enough memory to hold what we want.
        document_ranks = <INT_t *> calloc(n_documents, sizeof(INT_t))
        document_deltas = <DOUBLE_t *> calloc(n_documents, sizeof(DOUBLE_t))
        query_nnz_documents = <INT_t *> calloc((qend - qstart), sizeof(INT_t))

        if ((document_ranks == NULL) or (document_deltas == NULL) or
            (query_nnz_documents == NULL)):
            free(document_ranks)
            free(document_deltas)
            free(query_nnz_documents)
            with gil:
                raise MemoryError()

        # maximum_relevance + 1 is the row stride of
        # `influence_by_relevance` and `relevance_strides`.
        maximum_relevance += 1

        if influences != NULL:
            influence_by_relevance = <DOUBLE_t *> calloc(maximum_relevance,
                                                         sizeof(DOUBLE_t))

            if influence_by_relevance == NULL:
                free(document_ranks)
                free(document_deltas)
                free(query_nnz_documents)
                with gil:
                    raise MemoryError()

        # Relevance strides were not given, hence we need to build
        # them from the relevances.
        if relevance_strides == NULL:
            # Indicate that we need to resort the arrays back when we are done.
            resort = True

            # Create relevance_strides.
            relevance_strides = <INT_t *> calloc((qend - qstart) *
                                                 maximum_relevance,
                                                 sizeof(INT_t))

            if relevance_strides == NULL:
                free(document_ranks)
                free(document_deltas)
                free(query_nnz_documents)
                free(influence_by_relevance)
                with gil:
                    raise MemoryError()

            # Allocate memory for sorting and resorting indices.
            sort_indices = <INT_t *> malloc(2 * n_documents * sizeof(INT_t))

            if sort_indices == NULL:
                free(document_ranks)
                free(document_deltas)
                free(query_nnz_documents)
                free(influence_by_relevance)
                free(relevance_strides)
                with gil:
                    raise MemoryError()

            for i in range(qstart, qend):
                start, end = query_indptr[i], query_indptr[i + 1]

                # Makes the indexing easier.
                sort_indices += start - query_indptr[qstart]
                relevance_scores += start
                relevance_strides += (i - qstart) * maximum_relevance
                ranking_scores += start
                document_weights += start

                if leaves_idx != NULL:
                    leaves_idx += start

                # Get sorting indices (permutation) of the relevance
                # scores for query 'i'.
                rc = relevance_argsort_v1_c(relevance_scores, sort_indices,
                                            end - start, maximum_relevance)

                if rc == -1:
                    free(document_ranks)
                    free(document_deltas)
                    free(query_nnz_documents)
                    free(influence_by_relevance)
                    free(relevance_strides)
                    free(sort_indices)
                    with gil:
                        raise MemoryError()

                # Get inverse sort indices.
                for j in range(end - start):
                    sort_indices[sort_indices[j] + n_documents] = j

                # Sort related arrays according to the query relevance scores.
                sort_in_place(sort_indices, end - start,
                              relevance_scores, ranking_scores,
                              leaves_idx, document_weights)

                # Build relevance_strides for query 'i'.
                for j in range(end - start):
                    relevance_strides[relevance_scores[j]] += 1

                # Offset the relevance_strides properly to make it look
                # like it came from the input parameters.
                k = start
                for j in range(maximum_relevance - 1, -1, -1):
                    if relevance_strides[j] == 0:
                        relevance_strides[j] = -1
                    else:
                        relevance_strides[j] += k
                        k = relevance_strides[j]

                # Revert back the offseting.
                sort_indices -= start - query_indptr[qstart]
                relevance_scores -= start
                relevance_strides -= (i - qstart) * maximum_relevance
                ranking_scores -= start

                if leaves_idx != NULL:
                    leaves_idx -= start

                if document_weights != NULL:
                    document_weights -= start

            # Need to offset the `relevance_strides`
            # to make the indexing work later.
            relevance_strides -= qstart * maximum_relevance

        # Find documents with 0 weight and replace their ranking score
        # with NaN to make them fall to the lowest ranks in the rankings
        # and compute the number of documents with non-zero weight.
        for i in range(qstart, qend):
            for j in range(query_indptr[i], query_indptr[i + 1]):
                if document_weights[j] == 0.0:
                    ranking_scores[j] = NaN
                else:
                    query_nnz_documents[i - qstart] += 1

        # Find the rank of each document with respect to
        # the ranking scores over all queries.
        rc = argranksort_queries_c(query_indptr + qstart,
                                   qend - qstart,
                                   ranking_scores,
                                   document_ranks,
                                   &seed)
        if rc == -1:
            free(document_ranks)
            free(document_deltas)
            free(query_nnz_documents)
            free(influence_by_relevance)
            if resort:
                free(relevance_strides)
                free(sort_indices)
            with gil:
                raise MemoryError()

        # Clear output array for lambdas since we will be incrementing.
        memset(output_lambdas + query_indptr[qstart], 0,
               n_documents * sizeof(DOUBLE_t))

        # Clear output array for weights since we will be incrementing.
        memset(output_weights + query_indptr[qstart], 0,
               n_documents * sizeof(DOUBLE_t))

        # Loop through the queries and compute lambdas
        # and weights for every document.
        for i in range(qstart, qend):
            qweight = query_weights[i]

            # Queries with 0 weight are ignored.
            if qweight == 0.0:
                continue

            start, end = query_indptr[i], query_indptr[i + 1]

            qscale = scale_values[i]

            # The number of documents of the current query and the number of
            # documents with non-zero weight.
            n_documents = end - start
            nnz_documents = query_nnz_documents[i - qstart]

            # Prepare/reuse delta info structure for computation
            # of metric deltas.
            deltas_info = metric.prepare_delta_c(
                                document_ranks + start - query_indptr[qstart],
                                relevance_scores + start,
                                document_weights + start,
                                n_documents, qscale, info=deltas_info)

            # Loop through the documents of the current query.
            for j in range(start, end):
                j_relevance_score = relevance_scores[j]
                j_document_weight = document_weights[j]

                # Documents with 0 weight are ignored.
                if j_document_weight == 0.0:
                    continue

                # The smallest index of a document with a lower
                # relevance score than document 'j'.
                rstart = relevance_strides[i * maximum_relevance +
                                           j_relevance_score]

                # Is there any document relevant less than document 'j'?
                if rstart >= end:
                    break

                # Compute absolute changes in the metric caused by swapping
                # document 'j' with all documents 'k' (k >= rstart) which
                # are all guaranteed to be less relevant than 'j'.
                metric.delta_c(j - start,
                               rstart - start,
                               n_documents,
                               document_ranks + start - query_indptr[qstart],
                               relevance_scores + start,
                               document_weights + start,
                               nnz_documents,
                               qscale,
                               document_deltas,
                               info=deltas_info)

                # Clear the influences for the current document.
                if influence_by_relevance != NULL:
                    memset(influence_by_relevance, 0,
                           j_relevance_score * sizeof(DOUBLE_t))

                # Current forces pushing document 'j' down.
                j_push_down = output_lambdas[j]

                for k in range(rstart, end):
                    k_document_weight = document_weights[k]

                    # Documents with 0 weight are skipped.
                    if k_document_weight == 0.0:
                        continue

                    qjk_weight = (qweight * j_document_weight *
                                  k_document_weight)

                    rho = ((<DOUBLE_t> 1.0) /
                           ((<DOUBLE_t> 1.0) +
                            (<DOUBLE_t> exp(ranking_scores[j] -
                                            ranking_scores[k]))))

                    # Compute the loss for this pair of documents.
                    loss -= (qjk_weight * document_deltas[k - rstart] *
                             log(EPSILON if (1 - rho) < EPSILON else (1 - rho)))

                    # If the pair of documents fall into the same terminal
                    # node of a regression tree their contribution to
                    # the gradient of the loss with respect to the node
                    # prediction value is 0.
                    if leaves_idx != NULL and leaves_idx[j] == leaves_idx[k]:
                        continue

                    lambda_ = (rho * qjk_weight * document_deltas[k - rstart])
                    weight_ = (1 - rho) * lambda_

                    output_lambdas[j] += lambda_
                    output_lambdas[k] -= lambda_

                    output_weights[j] += weight_
                    output_weights[k] += weight_

                    if influence_by_relevance != NULL:
                        influence_by_relevance[relevance_scores[k]] += lambda_

                if influence_by_relevance != NULL:
                    for k in range(j_relevance_score):
                        if influence_by_relevance[k] <= output_lambdas[j]:
                            influences[k * maximum_relevance + j_relevance_score] += influence_by_relevance[k] / output_lambdas[j]
                        influences[j_relevance_score * maximum_relevance + k] += influence_by_relevance[k] / (output_lambdas[j] - 2 * j_push_down)

        # Collecting.
        metric.finalize_delta_c(deltas_info)

        # `relevance_strides` array has been constructed here.
        # We need to resort all the arrays back and free the memory.
        if resort:
            # Total number of documents sorted.
            n_documents = query_indptr[qend] - query_indptr[qstart]

            # The inverse sort indices are in the second half of the array.
            sort_indices += n_documents

            for i in range(qstart, qend):
                start, end = query_indptr[i], query_indptr[i + 1]

                # Make the indexing easier... maybe.
                sort_indices += start - query_indptr[qstart]
                relevance_scores += start
                ranking_scores += start

                if leaves_idx != NULL:
                    leaves_idx += start

                if document_weights != NULL:
                    document_weights += start

                output_lambdas += start
                output_weights += start

                # Revert back the earlier sort of related arrays.
                sort_in_place(sort_indices, end - start, relevance_scores,
                              ranking_scores, leaves_idx, document_weights,
                              output_lambdas, output_weights)

                # Revert back the offseting.
                sort_indices -= start - query_indptr[qstart]
                relevance_scores -= start
                ranking_scores -= start

                if leaves_idx != NULL:
                    leaves_idx -= start

                if document_weights != NULL:
                    document_weights -= start

                output_lambdas -= start
                output_weights -= start

            # Offset `sort_indices` and `relevance_strides` back.
            relevance_strides += qstart * maximum_relevance
            sort_indices -= n_documents

            free(relevance_strides)
            free(sort_indices)

        free(document_ranks)
        free(document_deltas)
        free(query_nnz_documents)
        free(influence_by_relevance)

    return loss


cdef void sort_in_place(INT_t *indices,
                        INT_t n_documents,
                        INT_t *relevance_scores,
                        DOUBLE_t *ranking_scores,
                        INT_t *leaves_idx,
                        DOUBLE_t *document_weights,
                        DOUBLE_t *lambdas=NULL,
                        DOUBLE_t *weights=NULL) nogil:
    '''
    Sort the given arrays according to `indices` in-place. Once done,
    indices will contain identity permutation.
    '''
    cdef INT_t start, end, tmp_relevance_score, tmp_leave_idx
    cdef DOUBLE_t tmp_ranking_score, tmp_document_weight
    cdef DOUBLE_t tmp_lambda, tmp_weight

    for i in range(n_documents):
        # Skipping fixed points (these elements are in the right place).
        if indices[i] != i:
            start = i

            # Temporarily store the items at the beginning
            # of the permutation cycle.
            tmp_relevance_score = relevance_scores[start]
            tmp_ranking_score = ranking_scores[start]

            if leaves_idx != NULL:
                tmp_leave_idx = leaves_idx[start]

            if document_weights != NULL:
                tmp_document_weight = document_weights[start]

            if lambdas != NULL:
                tmp_lambda = lambdas[start]
                tmp_weight = weights[start]

            # merry go round... ihaaa!
            while indices[start] != i:
                end = indices[start]

                relevance_scores[start] = relevance_scores[end]
                ranking_scores[start] = ranking_scores[end]

                if leaves_idx != NULL:
                    leaves_idx[start] = leaves_idx[end]

                if document_weights != NULL:
                    document_weights[start] = document_weights[end]

                if lambdas != NULL:
                    lambdas[start] = lambdas[end]
                    weights[start] = weights[end]

                indices[start] = start
                start = end

            # Move the items from the beginning of
            # the permutation cycle to the end.
            relevance_scores[end] = tmp_relevance_score
            ranking_scores[end] = tmp_ranking_score

            if leaves_idx != NULL:
                leaves_idx[end] = tmp_leave_idx

            if document_weights != NULL:
                document_weights[end] = tmp_document_weight

            if lambdas != NULL:
                lambdas[start] = tmp_lambda
                weights[start] = tmp_weight

            indices[end] = end
