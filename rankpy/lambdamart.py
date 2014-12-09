#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import logging
import sklearn

import numpy as np

from multiprocessing import Pool

from functools import partial

from .metrics import DiscountedCumulativeGain

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF

from scipy.special import expit

from shutil import rmtree
from tempfile import mkdtemp

from .externals.joblib import Parallel, delayed, cpu_count

from utils import pickle, unpickle
from .utils_inner import argranksort, ranksort


logger = logging.getLogger(__name__)


def _prepare_parallel_attributes(queries, scores, metric, n_jobs, metric_scale=None):
    ''' 
    Prepare arguments for calling of the method `_parallel_compute_lambda_and_weight`. These
    arguments are just views into the queries data, which will be processed by independent
    workers.

    The method returns two lists. The first contains query index list, which marks the chunk
    of queries that will be processed by a single call to `_parallel_compute_lambda_and_weight`
    with attributes in the second list.

    Parameters:
    -----------
    queries: rankpy.queries.Queries
        The queries from which this LambdaMART model is being trained.

    scores: array, shape = (n_documents,)
        The ranking score for each document in the queries set.

    metric: object implementing metrics.AbstractMetric
        The evaluation metric which will be used as a utility
        function optimized by this model.

    n_jobs: int
        The number of workers.

    metric_scale: float, optional (default is None)
        The precomputed ideal metric value for the specified queries.
    '''
    # Prepared parameters for the workers
    attributes = []

    # Each process is working on an independent continuous chunk of queries.
    if queries.query_count() >= n_jobs:
        training_query_sequence = np.linspace(0, queries.query_count(), n_jobs + 1).astype(np.intc)
    else:
        training_query_sequence = np.arange(queries.query_count() + 1, dtype=np.intc)

    # Prepare the appropriate views into the queries data. Every parameter is
    # just a view into needed portion of the query arrays memory.
    for i in range(len(training_query_sequence) - 1):
        qstart = training_query_sequence[i]
        qend = training_query_sequence[i + 1]
        dstart = queries.query_indptr[qstart]
        dend = queries.query_indptr[qend]
        query_indptr = queries.query_indptr[qstart:qend + 1]
        q_scores = scores[dstart:dend]
        relevance_scores = queries.relevance_scores[dstart:dend]
        relevance_strides = queries.query_relevance_strides[qstart:qend, :]
        q_metric_scale = None if metric_scale is None else metric_scale[qstart:qend]
        attributes.append((query_indptr, q_scores, relevance_scores, relevance_strides, metric, q_metric_scale))

    return training_query_sequence, attributes


def _paralell_compute_lambda_and_weight((query_indptr, scores, relevance_scores, relevance_strides, metric, metric_scale)):
    ''' 
    Compute the pseudo-responses (`lambdas`) and gradient descent steps (`weights`)
    for each document associated with the queries indexed via `query_indptr`.
    The pseudo-responses are calculated using the specified evaluation metric.

    Parameters:
    -----------
    query_indptr: array, shape = (n_queries + 1,), dtype = np.int32
        The start and end index into the shared array of documents.

    scores: array, shape = (n_documents,), dtype = np.float64
        The current scores the model have assigned to the documents.

    relevance_scores: array, shape = (n_documents,), dtyoe = np.int32
        The relevance scores of the documents associated with the query.

    relevance_strides: array, shape = (n_documenet, n_relevance_levels), dtype = np.int32
        For each relevance score it stores the (smallest) index of
        a document with smaller relevance.

    metric: object implementing metrics.AbstractMetric
        The evaluation metric (e.g. NDCG), for/from which will
        be the pseudoresponses computed.

    metric_scale: array, shape = (n_queries,) dtype = np.float64, optional (default is None)
        The precomputed ideal metric values for each of the given queries. These values
        can be used to speed up the computation because they are constant and do not need
        to be recomputed in every iteration, which is what happens if the value is None and
        the metric needs a normalizing constant.
    '''
    # Need to hold this for correct offseting.
    offset = query_indptr[0]

    # Make the query index pointer 0-based.
    query_indptr -= offset

    # Create output arrays
    output_lambdas = np.zeros(query_indptr[-1], dtype=np.float64)
    output_weights = np.zeros(query_indptr[-1], dtype=np.float64)

    # Create temporary buffer arrays.
    document_ranks_buffer = np.zeros(query_indptr[-1], dtype=np.int32)
    deltas_buffer = np.empty(query_indptr[-1], dtype=np.float64)

    # Loop through the queries and compute lambdas
    # and weights for every document.
    for i in range(query_indptr.size - 1):
        start = query_indptr[i]
        end = query_indptr[i + 1]

        scale = None if metric_scale is None else metric_scale[i]

        # The rank of each document of the current query.
        document_ranks = document_ranks_buffer[start:end]
        argranksort(scores[start:end], document_ranks)

        # Loop through the documents of the current query.
        for j in range(start, end):
            # rstart: the lowest index of a document with lower relevance score than document 'j'.
            rstart = relevance_strides[i, relevance_scores[j]] - offset

            if rstart >= end:
                break

            # Compute the (absolute) changes in the metric caused by swapping document 'j' with all
            # documents 'k' ('k' >= rstart), which have lower relevance to the current query.
            deltas = metric.compute_delta(j - start, rstart - start, document_ranks,
                                          relevance_scores[start:end], scale, deltas_buffer[rstart:end])

            scores[rstart:end] -= scores[j]

            # rho_i_j = 1.0 / (1.0 + exp(score[i] - score[j]))
            # Note: overloading output array to save some time and memory.
            weights = expit(scores[rstart:end])

            # Restore the scores after computing rho_i_j's.
            scores[rstart:end] += scores[j]

            # lambda_i_j for all less relevant (than i) documents j
            lambdas = weights * deltas 
            weights *= (1.0 - weights)
            weights *= deltas

            # lambda_i += sum_{j:relevance[i]>relevance[j]} lambda_i_j 
            output_lambdas[j] += lambdas.sum()
            # lambda_j -= lambda_i_j
            output_lambdas[rstart:end] -= lambdas

            # Weights are invariant when swapping i and j.
            output_weights[j] += weights.sum()
            output_weights[rstart:end] += weights

    return output_lambdas, output_weights


def _compute_lambda_and_weight(qid, queries, scores, metric, out_lambdas, out_weights, metric_scale=None):
    ''' 
    Compute the pseudo-responses (`lambdas`) and gradient steps sizes (`weights`) for each
    document associated with the specified query (`qid`). The pseudo-responses are
    calculated using the given evaluation metric.

    Parameters:
    -----------
    queries: rankpy.queries.Queries
        The queries from which this LambdaMART model is being trained.

    scores: array, shape = (n_documents,)
        The ranking score for each document in the queries set.

    metric: object implementing metrics.AbstractMetric
        The evaluation metric which will be used as a utility
        function optimized by this model.

    metric_scale: float, optional (default is None)
        The precomputed ideal metric value for the specified query.
    '''
    start = queries.query_indptr[qid]
    end   = queries.query_indptr[qid + 1]

    # The rank of each document for the query `qid`.
    document_ranks = np.empty(end - start, dtype=np.intc)
    argranksort(scores[start:end], document_ranks)

    # The relevance scores of the documents associated with the query `qid`.
    relevance_scores = queries.relevance_scores[start:end]

    # Prepare the output values for accumulation.
    out_lambdas[start:end].fill(0)
    out_weights[start:end].fill(0)

    for i in range(start, end):
        # rstart: the lowest index of a document with lower relevance score than document `i`.
        rstart = queries.query_relevance_strides[qid, relevance_scores[i - start]]

        if rstart >= end:
            break

        # Compute the (absolute) changes in the metric caused by swapping document `i` with all
        # documents `j` (`j` >= rstart) that have lower relevance to the query.
        deltas = metric.compute_delta(i - start, rstart - start, document_ranks, relevance_scores, metric_scale)

        scores[rstart:end] -= scores[i]

        # rho_i_j = 1.0 / (1.0 + exp(score[i] - score[j]))
        # Note: using name overloading to spare memory.
        weights = expit(scores[rstart:end])

        # Restore the scores after computing rho_i_j's.
        scores[rstart:end] += scores[i]

        # lambda_i_j for all less relevant (than `i`) documents `j`
        lambdas = weights * deltas 
        weights *= (1.0 - weights)
        weights *= deltas

        # lambda_i += sum_{j:relevance[i]>relevance[j]} lambda_i_j 
        out_lambdas[i] += lambdas.sum()
        # lambda_j -= lambda_i_j
        out_lambdas[rstart:end] -= lambdas

        # Weights are invariant in respect to swapping i and j.
        out_weights[i] += weights.sum()
        out_weights[rstart:end] += weights


def _compute_lambdas_and_weights(queries, scores, metric, out_lambdas, out_weights,
                                 pool=None, parallel_attributes=None, metric_scale=None):
    ''' 
    Compute the pseudo-responses (`lambdas`) and gradient steps sizes (`weights`)
    for each document in the specified set of queries. The pseudo-responses are calculated
    using the given evaluation metric.

    Note that this method decides to run either paralellized computation of the values
    using `_paralell_compute_lambda_and_weight` or single-threaded version `_compute_lambda_and_weight`
    based on value of parameter `pool` (and `parallel_attributes`).

    Parameters:
    -----------
    queries: rankpy.queries.Queries
        The queries from which this LambdaMART model is being trained.

    scores: array, shape = (n_documents,)
        The ranking score for each document in the queries set.

    metric: object implementing metrics.AbstractMetric
        The evaluation metric which will be used as a utility
        function optimized by this model.

    out_lambdas: array, shape=(n_documents,)
        Computed lambdas.

    out_weights: array, shape=(n_documents,)
        Computed weights.

    parallel_attributes: list of tuples, optional (default is None)
        2-item list constaining a list of 2-tuples marking
        the chunk of queries processed by a single worker
        with associated arguments for the method
        `_parallel_compute_lambdas_and_weights`.

    pool: Pool of workers, optional (default is None)
        The pool with workers processing given parallel_attributes.

    metric_scale: array, shape=(n_queries,), optional (default is None)
        The precomputed ideal metric value for each of the specified
        queries.
    '''
    # Decide wheter to use paralellized code or not.
    if pool is None:
        for qid in range(queries.query_count()):
            _compute_lambda_and_weight(qid, queries, scores, metric, out_lambdas, out_weights,
                                       None if metric_scale is None else metric_scale[qid])
    else:
        # Execute the computation of lambdas and weights in paralell. This ruins the whole thing (parameters
        # will be copied into child processes) but its just a couple of MBs anyway...
        for pid, (lambdas, weights) in enumerate(pool.imap(_paralell_compute_lambda_and_weight, parallel_attributes[1], chunksize=1)):
            start = queries.query_indptr[parallel_attributes[0][pid]]
            end   = queries.query_indptr[parallel_attributes[0][pid + 1]]
            np.copyto(out_lambdas[start:end], lambdas)
            np.copyto(out_weights[start:end], weights)


def _estimate_newton_gradient_steps(estimator, queries, lambdas, weights):
    ''' 
    Compute one iteration of Newton-Raphson method to estimate (optimal) gradient
    steps for each terminal node of the given estimator (regression tree).

    Parameters:
    -----------
    estimator: DecisionTreeRegressor or RandomForestRegressor
        The regression tree for which the gradient steps are computed.

    queries:
        The queries determine which terminal nodes of the tree the associated
        pseudo-responces (lambdas) and weights fall down to.

    lambdas:
        The current 1st order derivatives of the loss function.

    weights:
        The current 2nd order derivatives of the loss function.
    '''
    is_random_forest = isinstance(estimator, RandomForestRegressor)
    for estimator in estimator.estimators_ if is_random_forest else [estimator]:
        # Get the number of nodes (internal + terminal) in the current regression tree.
        node_count = estimator.tree_.node_count
        indices = estimator.tree_.apply(queries.feature_vectors)

        np.copyto(estimator.tree_.value, np.bincount(indices, lambdas, node_count).reshape(-1, 1, 1))

        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(estimator.tree_.value, np.bincount(indices, weights, node_count).reshape(-1, 1, 1), out=estimator.tree_.value)

        # Remove inf's and nas's from the tree.
        estimator.tree_.value[~np.isfinite(estimator.tree_.value)] = 0.0


def _parallel_build_trees(tree_offset, trees, queries, metric, use_newton_method, metric_scale):
    ''' 
    Train the trees on the specified queries using the LambdaMART's lambdas computed
    using the specified metric.

    Parameters:
    -----------
    tree_offset: int
        The index offset from which the indexing of the specified
        trees begins. Used only for logging progress.

    trees: list of DecisionTreeRegressors
        The list of regresion tree to train.

    queries: Queries object
        The set of queries from which the tree models will be trained.

    metric: metrics.AbstractMetric object
        Specify evaluation metric which will be used as a utility
        function (i.e. metric of `goodness`) optimized by the trees.

    use_newton_method: bool
        Estimate the gradient step in each terminal node of regression
        trees using Newton-Raphson method.

    metric_scale: array of floats, shape = (n_queries, )
        The precomputed ideal metric value for the specified queries.
    '''
    for tree_index, tree in enumerate(trees):
        logger.info('Started fitting LambdaDecisionTree no. %d.' % (tree_offset + tree_index))

        # Start with random ranking scores.
        training_scores = 0.0001 * np.random.rand(queries.document_count()).astype(np.float64)

        # The pseudo-responses (lambdas) for each document.
        training_lambdas = np.empty(queries.document_count(), dtype=np.float64)

        # The optimal gradient descent step sizes for each document.
        training_weights = np.empty(queries.document_count(), dtype=np.float64)

        # Compute the pseudo-responses (lambdas) and gradient step sizes (weights).
        _compute_lambdas_and_weights(queries, training_scores, metric, training_lambdas, training_weights,
                                     metric_scale=metric_scale)

        # Train the regression tree.
        tree.fit(queries.feature_vectors, training_lambdas)

        # Estimate the ('optimal') gradient step sizes using one iteration
        # of Newton-Raphson method.
        if use_newton_method:
           _estimate_newton_gradient_steps(tree, queries, training_lambdas, training_weights)

        logger.info('Finished fitting LambdaDecisionTree no. %d.' % (tree_offset + tree_index))

    return trees


def _parallel_helper(obj, methodname, *args, **kwargs):
    ''' 
    Helper function to avoid pickling problems when using multiprocessing.
    '''
    return getattr(obj, methodname)(*args, **kwargs)


class LambdaMART(object):
    ''' 
    LambdaMART learning to rank model.

    Arguments:
    -----------
    n_estimators: int, optional (default is 100)
        The number of regression tree estimators that will
        compose this ensemble model.

    shrinkage: float, optional (default is 0.1)
        The learning rate (a.k.a. shrinkage factor) that will
        be used to regularize the predictors (prevent them
        from making the full (optimal) Newton step.

    use_newton_method: bool, optional (default is True)
        Estimate the gradient step in each terminal node of regression
        trees using Newton-Raphson method.

    max_depth: int, optional (default is 5)
        The maximum depth of the regression trees. This parameter is ignored
        if `max_leaf_nodes` is specified (see description of `max_leaf_nodes`).

    max_leaf_nodes: int, optional (default is None)
        The maximum number of leaf nodes. If not None, the `max_depth` parameter
        will be ignored. The tree building strategy also changes from depth
        search first to best search first, which can lead to substantial decrease
        of training time.

    estopping: int, optional (default is 32)
        The number of subsequent iterations after which the training is stopped
        early if no improvement is observed on the validation queries.

    max_features: int or None, optional (default is None)
        The maximum number of features that is considered for splitting when
        regression trees are built. If None, all feature will be used.

    base_model: Base learning to rank model, optional (default is None)
        The base model, which is used to get initial ranking scores.

    n_jobs: int, optional (default is 1)
        The number of working sub-processes that will be spawned to compute
        the desired values faster. If -1, the number of CPUs will be used.

    seed: int, optional (default is None)
        The seed for random number generator that internally is used. This
        value should not be None only for debugging.
    '''
    def __init__(self, n_estimators=1000, shrinkage=0.1, use_newton_method=True, use_random_forest=-1,
                 max_depth=5, max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, estopping=100,
                 max_features=None, base_model=None, n_jobs=1, seed=None):
        self.estimators = []
        self.n_estimators = n_estimators
        self.shrinkage = shrinkage
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.estopping = estopping
        self.base_model = base_model
        self.use_newton_method=use_newton_method
        self.use_random_forest=use_random_forest
        self.n_jobs = max(1, n_jobs if n_jobs >= 0 else n_jobs + cpu_count() + 1)
        self.training_performance  = None
        self.validation_performance = None
        self.best_performance = None
        self.trained = False
        self.seed = seed

        # `max_leaf_nodes` were introduced in version 15 of scikit-learn.
        if self.max_leaf_nodes is not None and int(sklearn.__version__.split('.')[1]) < 15:
            raise ValueError('cannot use parameter `max_leaf_nodes` with scikit-learn of '
                             'version smaller than 15, use max_depth instead.' )


    def fit(self, metric, queries, validation=None, trace=None):
        ''' 
        Train the LambdaMART model on the specified queries. Optinally, use the
        specified queries for finding an optimal number of trees using validation.

        Parameters:
        -----------
        metric: metrics.AbstractMetric object
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.

        queries: Queries object
            The set of queries from which this LambdaMART model will be trained.

        validation: Queries object
            The set of queries used in validation for early stopping.

        trace: list of strings, optional (default is None)
            Supported values are: `lambdas`, `gradients`. Since the number of documents
            and estimators can be large it is not adviced to use the values together.
            When `lambdas` is given, then the true and estimated lambdas will be stored,
            and similarly, when `gradients` are given, then the true and estimated
            gradient will be stored. These two quantities differ only if the Newton
            method is used for estimating the gradient steps.
        '''
        # Initialize the random number generator.
        np.random.seed(self.seed)

        # Start with random ranking scores. Originally, the randomness was needed to avoid
        # biased evaluation of the rankings because the query documents are kept in sorted
        # order by their relevance scores (see `rankpy.queries.Queries`). But since
        # `utils_inner.argranksort` uses its own randomization (documents with the equal
        # scores are shuffled randomly) this purpose lost its meaning. Now it only serves
        # as a random initialization point as in any other gradient-based type of methdods. 
        if self.base_model is None:
            training_scores = 0.0001 * np.random.rand(queries.document_count()).astype(np.float64)
        else:
            training_scores = np.asanyarray(base_model.predict(queries, n_jobs=self.n_jobs), dtype=np.float64)

        # If the metric used for training is normalized, it is obviously advantageous
        # to precompute the scaling factor for each query in advance.
        training_metric_scale = metric.compute_scale(queries)
        self.training_performance = np.zeros(self.n_estimators, dtype=np.float64)

        # The pseudo-responses (lambdas) for each document.
        training_lambdas = np.empty(queries.document_count(), dtype=np.float64)

        # The optimal gradient descent step sizes for each document.
        training_weights = np.empty(queries.document_count(), dtype=np.float64)

        # The lambdas and predictions may be kept for late analysis.
        if trace is not None:
            # Create temporary directory for traced data.
            TEMP_DIRECTORY_NAME = mkdtemp(prefix='lambdamart.trace.data.tmp', dir='.') 

            logger.info('Created temporary directory (%s) for traced data.' % TEMP_DIRECTORY_NAME)

            self.trace_lambdas = trace.count('lambdas') > 0
            self.trace_gradients = trace.count('gradients') > 0

            # The pseudo-responses (lambdas) for each document: the true and estimated values.
            if self.trace_lambdas:
                # Use memory mapping to store large matrices.
                self.stage_training_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
                self.stage_training_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.lambdas.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))

                self.stage_validation_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))
                self.stage_validation_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))

            # The (loss) gradient steps for each query-document pair: the true and estimated by the regression trees.
            if self.trace_gradients:
                if self.use_newton_method is False:
                    raise ValueError('gradients are computed only if use_newton_method is True')
                # Use memory mapping to store large matrices.
                self.stage_training_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
                self.stage_training_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))

                self.stage_validation_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))
                self.stage_validation_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))

            # Used when the model is saved to get rid of it.
            self.tmp_directory = TEMP_DIRECTORY_NAME
        else:
            self.trace_lambdas = False
            self.trace_gradients = False

        # Initialize same the components for validation queries as for training.
        if validation is not None:
            if self.base_model is None:
                self.validation_scores = np.zeros(validation.document_count(), dtype=np.float64)
            else:
                self.validation_scores = np.asanyarray(self.base_model.predict(validation, n_jobs=self.n_jobs), dtype=np.float64)

            validation_metric_scale = metric.compute_scale(validation)
            self.validation_performance = np.zeros(self.n_estimators, dtype=np.float64)

            if self.trace_lambdas:
                # The pseudo-responses (lambdas) for each document in validation queries.
                self.validation_lambdas = np.empty(validation.document_count(), dtype=np.float64)
                # The optimal gradient descent step sizes for each document in validation queries.
                self.validation_weights = np.empty(validation.document_count(), dtype=np.float64)

        # Hope we will be always maximizing :).
        best_performance = -np.inf
        best_performance_k = -1

        # How many iterations the performance has not improved on validation set.
        performance_not_improved = 0

        # Prepare training pool (if wanted).
        training_pool = None if self.n_jobs == 1 else Pool(processes=self.n_jobs)

        if training_pool is not None:
            # Prepare parameters for background workers
            training_parallel_attributes = _prepare_parallel_attributes(queries, training_scores, metric,
                                                                        training_pool._processes, training_metric_scale)
            if validation is not None:
                validation_parallel_attributes = _prepare_parallel_attributes(validation, self.validation_scores, metric,
                                                                              training_pool._processes, validation_metric_scale)
        else:
            training_parallel_attributes = None
            validation_parallel_attributes = None

        logger.info('Training of LambdaMART model has started.')

        # Iteratively build a sequence of regression trees.
        for k in range(self.n_estimators):
            # Computes the pseudo-responses (lambdas) and gradient step sizes (weights) for the current regression tree.
            _compute_lambdas_and_weights(queries, training_scores, metric, training_lambdas,
                                         training_weights, training_pool, training_parallel_attributes,
                                         training_metric_scale)

            # Build the predictor for the gradients of the loss using either decision tree or random forest.
            if self.use_random_forest > 0:
                estimator = RandomForestRegressor(n_estimators=self.use_random_forest,
                                                  max_depth=self.max_depth,
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  max_features=self.max_features,
                                                  min_samples_split=self.min_samples_split,
                                                  min_samples_leaf=self.min_samples_leaf,
                                                  n_jobs=self.n_jobs)
            else:
                estimator = DecisionTreeRegressor(max_depth=self.max_depth,
                                                  max_leaf_nodes=self.max_leaf_nodes,
                                                  max_features=self.max_features,
                                                  min_samples_split=self.min_samples_split,
                                                  min_samples_leaf=self.min_samples_leaf)
            # Train the regression tree.
            estimator.fit(queries.feature_vectors, training_lambdas)

            # Store the estimated lambdas for later analysis (if wanted).
            if self.trace_lambdas:
                np.copyto(self.stage_training_lambdas_truth[k], training_lambdas)
                np.copyto(self.stage_training_lambdas_predicted[k], estimator.predict(queries.feature_vectors))

                if validation is not None:
                    _compute_lambdas_and_weights(validation, self.validation_scores, metric, self.validation_lambdas,
                                                 self.validation_weights, training_pool, validation_parallel_attributes,
                                                 validation_metric_scale)
                    np.copyto(self.stage_validation_lambdas_truth[k], self.validation_lambdas)
                    np.copyto(self.stage_validation_lambdas_predicted[k], estimator.predict(validation.feature_vectors))

            # Estimate the ('optimal') gradient step sizes using one iteration
            # of Newton-Raphson method.
            if self.use_newton_method:
                _estimate_newton_gradient_steps(estimator, queries, training_lambdas, training_weights)

            # Store the true and estimated gradients for later analysis.
            if self.trace_gradients:
                with np.errstate(divide='ignore', invalid='ignore'):
                    np.copyto(self.stage_training_gradients_truth[k], training_lambdas)
                    np.divide(self.stage_training_gradients_truth[k], training_weights, out=self.stage_training_gradients_truth[k])
                    self.stage_training_gradients_truth[k, ~np.isfinite(self.stage_training_gradients_truth[k])] = 0.0
                np.copyto(self.stage_training_gradients_predicted[k], estimator.predict(queries.feature_vectors))
                # Do the same thing as above for validation queries.
                if validation is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        np.copyto(self.stage_validation_gradients_truth[k], self.validation_lambdas)
                        np.divide(self.stage_validation_gradients_truth[k], self.validation_weights, out=self.stage_validation_gradients_truth[k])
                        self.stage_validation_gradients_truth[k, ~np.isfinite(self.stage_validation_gradients_truth[k])] = 0.0
                    np.copyto(self.stage_validation_gradients_predicted[k], estimator.predict(validation.feature_vectors))

            # Update the document scores using the new gradient predictor.
            if self.trace_gradients:
                training_scores += self.shrinkage * self.stage_training_gradients_predicted[k, :]
            else:
                training_scores += self.shrinkage * estimator.predict(queries.feature_vectors)

            # Add the new tree to the company.
            self.estimators.append(estimator)

            self.training_performance[k] = metric.evaluate_queries(queries, training_scores, scale=training_metric_scale)

            if validation is None:
                logger.info('#%08d: %s (training): %11.8f' % (k + 1, metric, self.training_performance[k]))

            # If validation queries have been given, estimate the model performance on them
            # and decide whether the training should not be stopped early due to no significant
            # performance improvements.
            if validation is not None:
                self.validation_scores += self.shrinkage * self.estimators[-1].predict(validation.feature_vectors)
                self.validation_performance[k] = metric.evaluate_queries(validation, self.validation_scores, scale=validation_metric_scale)

                logger.info('#%08d: %s (training):   %11.8f  |  (validation):   %11.8f' % (k + 1, metric, self.training_performance[k], self.validation_performance[k]))

                if self.validation_performance[k] > best_performance:
                    best_performance = self.validation_performance[k]
                    best_performance_k = k
                    performance_not_improved = 0
                else:
                    performance_not_improved += 1

            elif self.training_performance[k] > best_performance:
                    best_performance = self.training_performance[k]
                    best_performance_k = k

            if performance_not_improved >= self.estopping:
                logger.info('Stopping early since no improvement on validation queries'\
                            ' has been observed for %d iterations (since iteration %d)' % (self.estopping, best_performance_k + 1))
                break

        if validation is not None:
            logger.info('Final model performance (%s) on validation queries: %11.8f' % (metric, best_performance))

        # Leave the estimators that led to the best performance,
        # either on training or validation set.
        del self.estimators[best_performance_k + 1:]

        # Correct the number of trees.
        self.n_estimators = len(self.estimators)

        # Set these for further inspection.
        self.training_performance = np.resize(self.training_performance, k + 1)
        self.best_performance = best_performance

        if validation is not None:
            self.validation_performance = np.resize(self.validation_performance, k + 1)

        # Mark the model as trained.
        self.trained = True

        logger.info('Training of LambdaMART model has finished.')


    @staticmethod
    def __predict(trees, shrinkage, feature_vectors, output):
        for tree in trees:
            output += tree.predict(feature_vectors)
        output *= shrinkage


    def predict(self, queries, n_jobs=1):
        ''' 
        Predict the ranking score for each individual document of the given queries.

        n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        if self.trained is False:
            raise ValueError('the model has not been trained yet')

        predictions = np.zeros(queries.document_count(), dtype=np.float64)

        n_jobs = max(1, min(n_jobs if n_jobs >= 0 else n_jobs + cpu_count() + 1, queries.document_count()))

        indices = np.linspace(0, queries.document_count(), n_jobs + 1).astype(np.intc)

        Parallel(n_jobs=n_jobs, backend="threading")(delayed(_parallel_helper, check_pickle=False)
                (LambdaMART, '_LambdaMART__predict', self.estimators, self.shrinkage,
                 queries.feature_vectors[indices[i]:indices[i + 1]],
                 predictions[indices[i]:indices[i + 1]]) for i in range(indices.size - 1))

        return predictions


    def predict_ranking(self, query):
        ''' 
        Predict the document ranking of the documents for the given query.

        query: rankpy.queries.Query object
            The query whose documents should be ranked.
        '''
        if self.trained is False:
            raise ValueError('the model has not been trained yet')

        predictions = np.zeros(query.document_count(), dtype=np.float64)
        ranking = np.zeros(query.document_count(), dtype=np.intc)

        # Predict the ranking scores for the documents.
        LambdaMART.__predict(self.estimators, self.shrinkage, query.feature_vectors, predictions)

        ranksort(predictions, ranking)

        return ranking


    def feature_importances(self):
        ''' 
        Return the feature importances.
        '''
        if self.trained is False:   
            raise ValueError('the model has not been trained yet')

        importances = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(getattr, check_pickle=False)
                              (tree, 'feature_importances_') for tree in self.estimators)

        return sum(importances) / self.n_estimators


    @classmethod
    def load(cls, filepath, mmap='r'):
        ''' 
        Load the previously saved LambdaMART model from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a LambdaMART object will be loaded.

        mmap: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}, optional (default is 'r')
            If not None, then memory-map the traced data (if any), using
            the given mode (see `numpy.memmap` for a details).
        '''
        logger.info("Loading %s object from %s" % (cls.__name__, filepath))

        obj = unpickle(filepath)

        if obj.trace_lambdas:
            logger.info('Loading traced (true) lambda values from %s.training.lambdas.truth.npy' % filepath)
            setattr(obj, 'stage_training_lambdas_truth', np.load(filepath + '.training.lambdas.truth.npy', mmap_mode=mmap))

            logger.info('Loading traced (predicted) lambda values from %s.training.lambdas.predicted.npy' % filepath)
            setattr(obj, 'stage_training_lambdas_predicted', np.load(filepath + '.training.lambdas.predicted.npy', mmap_mode=mmap))

            if hasattr(obj, 'validation_performance'):
                logger.info('Loading traced (true) lambda values from %s.validation.lambdas.truth.npy' % filepath)
                setattr(obj, 'stage_validation_lambdas_truth', np.load(filepath + '.validation.lambdas.truth.npy', mmap_mode=mmap))

                logger.info('Loading traced (predicted) lambda values from %s.validation.lambdas.predicted.npy' % filepath)
                setattr(obj, 'stage_validation_lambdas_predicted', np.load(filepath + '.validation.lambdas.predicted.npy', mmap_mode=mmap))

        if obj.trace_gradients:
            logger.info('Loading traced (true) gradient values from %s.training.gradients.truth.npy' % filepath)
            setattr(obj, 'stage_training_gradients_truth', np.load(filepath + '.training.gradients.truth.npy', mmap_mode=mmap))

            logger.info('Loading traced (predicted) gradient values from %s.training.gradients.predicted.npy' % filepath)
            setattr(obj, 'stage_training_gradients_predicted', np.load(filepath + '.training.gradients.predicted.npy', mmap_mode=mmap))

            if hasattr(obj, 'validation_performance'):
                logger.info('Loading traced (true) gradient values from %s.validation.gradients.truth.npy' % filepath)
                setattr(obj, 'stage_validation_gradients_truth', np.load(filepath + '.validation.gradients.truth.npy', mmap_mode=mmap))

                logger.info('Loading traced (predicted) gradient values from %s.validation.gradients.predicted.npy' % filepath)
                setattr(obj, 'stage_validation_gradients_predicted', np.load(filepath + '.validation.gradients.predicted.npy', mmap_mode=mmap))

        return obj


    def save(self, filepath):
        ''' 
        Save te LambdaMART model into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        logger.info("Saving %s object into %s" % (self.__class__.__name__, filepath))

        # Deal with saving the memory-mapped arrays: only the used part of the arrays are saved
        # with the model, separately, i.e., the arrays are standalone *.npy files. The temporary
        # directory is removed after this.
        if self.trace_lambdas:
            logger.info('Saving traced (true) lambda values into %s.training.lambdas.truth.npy' % filepath)
            np.save(filepath + '.training.lambdas.truth.npy', self.stage_training_lambdas_truth[:self.training_performance.shape[0]])
            del self.stage_training_lambdas_truth

            logger.info('Saving traced (predicted) lambda values into %s.training.lambdas.predicted.npy' % filepath)
            np.save(filepath + '.training.lambdas.predicted.npy', self.stage_training_lambdas_predicted[:self.training_performance.shape[0]])
            del self.stage_training_lambdas_predicted

            if hasattr(self, 'validation_performance'):
                logger.info('Saving traced (true) lambda values into %s.validation.lambdas.truth.npy' % filepath)
                np.save(filepath + '.validation.lambdas.truth.npy', self.stage_validation_lambdas_truth[:self.validation_performance.shape[0]])
                del self.stage_validation_lambdas_truth

                logger.info('Saving traced (predicted) lambda values into %s.validation.lambdas.predicted.npy' % filepath)
                np.save(filepath + '.validation.lambdas.predicted.npy', self.stage_validation_lambdas_predicted[:self.validation_performance.shape[0]])
                del self.stage_validation_lambdas_predicted

        if self.trace_gradients:
            logger.info('Saving traced (true) gradient values into %s.training.gradients.truth.npy' % filepath)
            np.save(filepath + '.training.gradients.truth.npy', self.stage_training_gradients_truth[:self.training_performance.shape[0]])
            del self.stage_training_gradients_truth

            logger.info('Saving traced (predicted) gradient values into %s.training.gradients.predicted.npy' % filepath)
            np.save(filepath + '.training.gradients.predicted.npy', self.stage_training_gradients_predicted[:self.training_performance.shape[0]])
            del self.stage_training_gradients_predicted

            if hasattr(self, 'validation_performance'):
                logger.info('Saving traced (true) gradient values into %s.validation.gradients.truth.npy' % filepath)
                np.save(filepath + '.validation.gradients.truth.npy', self.stage_validation_gradients_truth[:self.validation_performance.shape[0]])
                del self.stage_validation_gradients_truth

                logger.info('Saving traced (predicted) gradient values into %s.validation.gradients.predicted.npy' % filepath)
                np.save(filepath + '.validation.gradients.predicted.npy', self.stage_validation_gradients_predicted[:self.validation_performance.shape[0]])
                del self.stage_validation_gradients_predicted

        # Get rid of the temporary directory.
        if hasattr(self, 'tmp_directory'):
            logger.info('Deleting temporary directory (%s) for traced data.' % self.tmp_directory)
            rmtree(self.tmp_directory)
            del self.tmp_directory

        pickle(self, filepath)

        if self.trace_lambdas:
            self.stage_training_lambdas_truth = np.load(filepath + '.training.lambdas.truth.npy', mmap_mode='r')
            self.stage_training_lambdas_predicted = np.load(filepath + '.training.lambdas.predicted.npy', mmap_mode='r')

            if hasattr(self, 'validation_performance'):
                self.stage_validation_lambdas_truth = np.load(filepath + '.validation.lambdas.truth.npy', mmap_mode='r')
                self.stage_validation_lambdas_predicted = np.load(filepath + '.validation.lambdas.predicted.npy', mmap_mode='r')

        if self.trace_gradients:
            self.stage_training_gradients_truth = np.load(filepath + '.training.gradients.truth.npy', mmap_mode='r')
            self.stage_training_gradients_predicted = np.load(filepath + '.training.gradients.predicted.npy', mmap_mode='r')

            if hasattr(self, 'validation_performance'):
                self.stage_validation_gradients_truth = np.load(filepath + '.validation.gradients.truth.npy', mmap_mode='r')
                self.stage_validation_gradients_predicted = np.load(filepath + '.validation.gradients.predicted.npy', mmap_mode='r')


    def save_as_text(self, filepath):
        ''' 
        Save the model into the file in an XML format.
        '''
        if self.use_random_forest > 0:
            raise ValueError('cannot save model to text when using it uses random forest')

        with open(filepath, 'w') as ofile:
            padding = '\t\t'

            ofile.write('<LambdaMART>\n')
            ofile.write('\t<parameters>\n')
            ofile.write('\t\t<trees> %d </trees>\n' % self.n_estimators)
            if self.max_leaf_nodes is not None:
                ofile.write('\t\t<leaves> %d </leaves>\n' % self.max_leaf_nodes)
            else:
                ofile.write('\t\t<depth> %d </depth>\n' % self.max_depth)
            ofile.write('\t\t<features> %d </features>\n' % -1 if self.max_features is None else self.max_features)
            ofile.write('\t\t<shrinkage> %.2f </shrinkage>\n' % self.shrinkage)
            ofile.write('\t\t<estopping> %d </estopping>\n' % self.estopping)
            ofile.write('\t<ensemble>\n')
            for id, tree in enumerate(self.estimators, start=1):
                # Getting under the tree bark...
                tree = tree.tree_

                ofile.write(padding + '<tree id="%d">\n' % id)

                # Stack of 3-tuples: (depth, parent, node_id).
                stack = [(1, TREE_UNDEFINED, 0)]

                while stack:
                    depth, parent, node_id = stack.pop()

                    # End of split mark.
                    if node_id < 0:
                        ofile.write(padding + (depth * '\t'))
                        ofile.write('</split>\n')
                        continue

                    ofile.write(padding + (depth * '\t'))
                    ofile.write('<split')

                    if parent == TREE_UNDEFINED:
                        ofile.write('>\n')
                    else:
                        pos = 'left' if tree.children_left[parent] == node_id else 'right'
                        ofile.write(' pos="%s">\n' % pos)

                    # If the node is a leaf.
                    if tree.children_left[node_id] == TREE_LEAF:
                        ofile.write(padding + ((depth + 1) * '\t'))
                        ofile.write('<output> %.17f </output>\n' % tree.value[node_id])
                        ofile.write(padding + (depth * '\t'))
                        ofile.write('</split>\n')
                    else:
                        ofile.write(padding + ((depth + 1) * '\t'))
                        # FIXME: Feature indexing should be marked somewhere if it
                        # realy is 0-based or not. Here we are assuming it is NOT!
                        ofile.write('<feature> %d </feature>\n' % (tree.feature[node_id] + 1))
                        ofile.write(padding + ((depth + 1) * '\t'))
                        ofile.write('<threshold> %.9f </threshold>\n' % tree.threshold[node_id])

                        # Push the end of split mark first... then push the right and left child.
                        stack.append((depth, parent, -1))
                        stack.append((depth + 1, node_id, tree.children_right[node_id]))
                        stack.append((depth + 1, node_id, tree.children_left[node_id]))

                ofile.write(padding + '</tree>\n')
            ofile.write('\t</ensemble>\n')
            ofile.write('</LambdaMART>\n')

    def __str__(self):
        ''' 
        Return textual representation of the LambdaMART model.
        '''
        return 'LambdaMART(trees=%d, max_depth=%s, max_leaf_nodes=%s, shrinkage=%.2f, max_features=%s, use_newton_method=%s, trained=%s)' % \
               (self.n_estimators, self.max_depth if self.max_leaf_nodes is None else '?', '?' if self.max_leaf_nodes is None else self.max_leaf_nodes,
                self.shrinkage, 'all' if self.max_features is None else self.max_features, self.use_newton_method, self.trained)


class LambdaRandomForest(object):
    ''' 
    LambdaRandomForest learning to rank model.

    Arguments:
    -----------
    n_estimators: int, optional (default is 100)
        The number of regression ranomized tree estimators that will
        compose this ensemble model.

    use_newton_method: bool, optional (default is True)
        Estimate the gradient step in each terminal node of regression
        trees using Newton-Raphson method.

    max_depth: int, optional (default is 5)
        The maximum depth of the regression trees. This parameter is ignored
        if `max_leaf_nodes` is specified (see description of `max_leaf_nodes`).

    max_leaf_nodes: int, optional (default is None)
        The maximum number of leaf nodes. If not None, the `max_depth` parameter
        will be ignored. The tree building strategy also changes from depth
        search first to best search first, which can lead to substantial decrease
        of training time.

    max_features: int or None, optional (default is None)
        The maximum number of features that is considered for splitting when
        regression trees are built. If None, all feature will be used.

    n_jobs: int, optional (default is 1)
        The number of working sub-processes that will be spawned to compute
        the desired values faster. If -1, the number of CPUs will be used.

    seed: int, optional (default is None)
        The seed for random number generator that internally is used. This
        value should not be None only for debugging.
    '''
    def __init__(self, n_estimators=1000, use_newton_method=True, max_depth=None, max_leaf_nodes=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None, n_jobs=1, shuffle=True,
                 seed=None):
        self.estimators = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.use_newton_method=use_newton_method
        self.shuffle=shuffle
        self.n_jobs = max(1, n_jobs if n_jobs >= 0 else n_jobs + cpu_count() + 1)
        self.trained = False
        self.seed = seed

        # `max_leaf_nodes` were introduced in version 15 of scikit-learn.
        if self.max_leaf_nodes is not None and int(sklearn.__version__.split('.')[1]) < 15:
            raise ValueError('cannot use parameter `max_leaf_nodes` with scikit-learn of version smaller than 15')


    def fit(self, metric, queries):
        ''' 
        Train the LambdaMART model on the specified queries. Optinally, use the
        specified queries for finding an optimal number of trees using validation.

        Parameters:
        -----------
        metric: metrics.AbstractMetric object
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.

        queries: Queries object
            The set of queries from which this LambdaMART model will be trained.
        '''
        # Initialize the random number generator.
        np.random.seed(self.seed)

        # If the metric used for training is normalized, it is obviously advantageous
        # to precompute the scaling factor for each query in advance.
        training_metric_scale = metric.compute_scale(queries)

        logger.info('Training of LambdaRandomForest model has started.')

        if self.shuffle:
            estimators = []

            for k in range(self.n_estimators):
                estimators.append(DecisionTreeRegressor(max_depth=self.max_depth,
                                                        max_leaf_nodes=self.max_leaf_nodes,
                                                        min_samples_split=self.min_samples_split,
                                                        min_samples_leaf=self.min_samples_leaf,
                                                        max_features=self.max_features))

            n_jobs = max(1, min(self.n_jobs if self.n_jobs >= 0 else self.n_jobs + cpu_count() + 1, self.n_estimators))

            indptr = np.linspace(0, self.n_estimators, n_jobs + 1).astype(np.intc)

            # Note: Using multithreading would be nice, but since lambdas and weights are not
            #       yet computed in Cython (without GIL) the multprocessing here is much more
            #       efficient.
            #
            # Warning: The queries need to be memory-mapped...
            estimators = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(_parallel_build_trees, check_pickle=False)
                                 (indptr[i], estimators[indptr[i]:indptr[i + 1]], queries, metric,
                                  self.use_newton_method, training_metric_scale) for i in range(indptr.size - 1))

            # Collect the trained estimators.
            self.estimators = sum(estimators, [])
        else:
            # Start with random ranking scores.
            # training_scores = np.zeros(queries.document_count(), dtype=np.float64)
            training_scores = 0.0001 * np.random.rand(queries.document_count()).astype(np.float64)

            # The pseudo-responses (lambdas) for each document.
            training_lambdas = np.empty(queries.document_count(), dtype=np.float64)

            # The optimal gradient descent step sizes for each document.
            training_weights = np.empty(queries.document_count(), dtype=np.float64)

    	    # Prepare training pool (if wanted).
            training_pool = None if self.n_jobs == 1 else Pool(processes=self.n_jobs)

            if training_pool is not None:
                # Prepare parameters for background workers
                training_parallel_attributes = _prepare_parallel_attributes(queries, training_scores, metric,
                                                                            training_pool._processes, training_metric_scale)
            else:
                training_parallel_attributes = None

            # Compute the pseudo-responses (lambdas) and gradient step sizes (weights) just once.
            _compute_lambdas_and_weights(queries, training_scores, metric, training_lambdas, training_weights,
                                         training_pool, training_parallel_attributes, training_metric_scale)

            # Build the predictor for the gradients of the loss.
            estimator = RandomForestRegressor(n_estimators=self.n_estimators,
                                              max_depth=self.max_depth,
                                              max_leaf_nodes=self.max_leaf_nodes,
                                              max_features=self.max_features,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              n_jobs=self.n_jobs)

            # Train the regression tree.
            estimator.fit(queries.feature_vectors, training_lambdas)

            # Estimate the ('optimal') gradient step sizes using one iteration
            # of Newton-Raphson method.
            if self.use_newton_method:
                _estimate_newton_gradient_steps(estimator, queries, training_lambdas, training_weights)

            # Add the trees from the random forest into our own list of estimators.
            self.estimators.extend(estimator.estimators_)
        
        # Mark the model as trained.
        self.trained = True

        logger.info('Training of LambdaRandomForest model has finished.')


    @staticmethod
    def __predict(estimators, feature_vectors, output):
        for estimator in estimators:
            output += estimator.predict(feature_vectors)


    def predict(self, queries, n_jobs=1):
        ''' 
        Predict the ranking score for each individual document of the given queries.

        n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        if self.trained is False:
            raise ValueError('the model has not been trained yet')

        predictions = np.zeros(queries.document_count(), dtype=np.float64)

        n_jobs = max(1, min(n_jobs if n_jobs >= 0 else n_jobs + cpu_count() + 1, queries.document_count()))

        indices = np.linspace(0, queries.document_count(), n_jobs + 1).astype(np.intc)

        Parallel(n_jobs=n_jobs, backend="threading")(delayed(_parallel_helper, check_pickle=False)
                (LambdaRandomForest, '_LambdaRandomForest__predict', self.estimators,
                 queries.feature_vectors[indices[i]:indices[i + 1]],
                 predictions[indices[i]:indices[i + 1]]) for i in range(indices.size - 1))

        predictions /= len(self.estimators)

        return predictions


    def predict_ranking(self, query):
        ''' 
        Predict the document ranking of the documents for the given query.

        query: rankpy.queries.Query object
            The query whose documents should be ranked.
        '''
        if self.trained is False:
            raise ValueError('the model has not been trained yet')

        predictions = np.zeros(query.document_count(), dtype=np.float64)
        ranking = np.zeros(query.document_count(), dtype=np.intc)

        # Predict the ranking scores for the documents.
        self.__predict(self.estimators, self.shrinkage, query.feature_vectors, predictions)

        ranksort(predictions, ranking)

        return ranking


    def feature_importances(self):
        ''' 
        Return the feature importances.
        '''
        if self.trained is False:   
            raise ValueError('the model has not been trained yet')

        importances = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(getattr, check_pickle=False)
                              (tree, 'feature_importances_') for tree in self.estimators)

        return sum(importances) / self.n_estimators


    @classmethod
    def load(cls, filepath):
        ''' 
        Load the previously saved LambdaRandomForest model from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a LambdaRandomForest object will be loaded.
        '''
        logger.info("Loading %s object from %s" % (cls.__name__, filepath))
        return unpickle(filepath)


    def save(self, filepath):
        ''' 
        Save te LambdaRandomForest model into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        logger.info("Saving %s object into %s" % (self.__class__.__name__, filepath))
        pickle(self, filepath)


    def __str__(self):
        ''' 
        Return textual representation of the LambdaRandomForest model.
        '''
        return 'LambdaRandomForest(trees=%d, max_depth=%s, max_leaf_nodes=%s, max_features=%s, use_newton_method=%s, trained=%s)' % \
               (self.n_estimators, self.max_depth if self.max_leaf_nodes is None else '?', '?' if self.max_leaf_nodes is None else self.max_leaf_nodes,
                'all' if self.max_features is None else self.max_features, self.use_newton_method, self.trained)


if __name__ == '__main__':
    pass
