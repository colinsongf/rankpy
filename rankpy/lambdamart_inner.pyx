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

from .metrics._utils cimport INT_t
from .metrics._utils cimport DOUBLE_t
from .metrics._utils cimport argranksort_c

from .metrics._metrics cimport Metric


def _parallel_compute_lambdas_and_weights(INT_t qstart, INT_t qend, INT_t[::1] query_indptr,
                                          DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores,
                                          INT_t[:, ::1] relevance_strides, Metric metric, DOUBLE_t[::1] metric_scale,
                                          DOUBLE_t[::1] output_lambdas, DOUBLE_t[::1] output_weights):
    cdef:
        INT_t i, j, k, start,rstart, end, n_documents
        INT_t *document_ranks
        DOUBLE_t *document_deltas
        DOUBLE_t lambda_, weight, rho, scale

    with nogil:
        # Total number of documents to process.
        n_documents = query_indptr[qend] - query_indptr[qstart]

        # More than enough memory to hold what we want.
        document_ranks = <INT_t *> calloc(n_documents, sizeof(INT_t))
        document_deltas = <DOUBLE_t *> calloc(n_documents, sizeof(DOUBLE_t))

        # Clear output array for lambdas since we will be incrementing.
        memset((&output_lambdas[0]) + query_indptr[qstart], 0, n_documents * sizeof(DOUBLE_t))

        # Clear output array for weights since we will be incrementing.
        memset((&output_weights[0]) + query_indptr[qstart], 0, n_documents * sizeof(DOUBLE_t))

        # Loop through the queries and compute lambdas and weights for every document.
        for i in range(qstart, qend):
            start, end = query_indptr[i], query_indptr[i + 1]

            scale = 1.0 if metric_scale is None else metric_scale[i]

            # The number of documents of the current query.
            n_documents = end - start

            # Find the rank of each document with respect to the ranking socres.
            argranksort_c(&ranking_scores[0], document_ranks, n_documents)

            # Loop through the documents of the current query.
            for j in range(start, end):
                # The smallest index of a document with a lower relevance score than document 'j'.
                rstart = relevance_strides[i, relevance_scores[j]]

                # Is there any document less relevant than document 'j'?
                if rstart >= end:
                    break

                # Compute the (absolute) changes in the metric caused by swapping document 'j' with all
                # documents 'k' (k >= rstart), which have lower relevance with respect to the query 'i'.
                metric.delta_c(j - start, rstart - start, n_documents, document_ranks, (&relevance_scores[0]) + start, scale, document_deltas)

                for k in range(rstart, end):
                    rho = (<DOUBLE_t> 1.0) / ((<DOUBLE_t> 1.0) + (<DOUBLE_t> exp(ranking_scores[j] - ranking_scores[k])))

                    lambda_ = rho * document_deltas[k - rstart]
                    weight = (1 - rho) * lambda_

                    output_lambdas[j] += lambda_
                    output_lambdas[k] -= lambda_

                    output_weights[j] += weight
                    output_weights[k] += weight

        free(document_ranks)
        free(document_deltas)