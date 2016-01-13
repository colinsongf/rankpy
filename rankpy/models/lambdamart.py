# -*- coding: utf-8 -*-
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


import os
import logging
import sklearn
import warnings

import numpy as np

from ..externals.joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor

from sklearn.utils import check_random_state

try:
    # Try to import tinkered sklearn trees...
    from sklemot import DecisionTreeRegressor
    from sklearn import ExtraTreeRegressor
    from sklearn._tree import TREE_UNDEFINED, TREE_LEAF
    SKLEMOT_TREE_IMPORTED = True

except ImportError:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import ExtraTreeRegressor
    from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF
    SKLEMOT_TREE_IMPORTED = False

from shutil import rmtree
from tempfile import mkdtemp

from ..utils import pickle
from ..utils import unpickle
from ..utils import parallel_helper
from ..utils import _get_partition_indices
from ..utils import _get_n_jobs

from ..metrics._utils import ranksort_queries

from collections import deque

from .lambdamart_inner import parallel_compute_lambdas_and_weights


logger = logging.getLogger(__name__)


def compute_lambdas_and_weights(queries, ranking_scores, metric,
                                output_lambdas, output_weights,
                                scale_values=None, relevance_scores=None,
                                query_weights=None, document_weights=None,
                                influences=None, indices=None,
                                random_state=None, n_jobs=1):
    '''
    Compute the first derivatives (`lambdas`) and the second derivatives
    (`weights`) of an implicit cost function from the given metric and
    rankings of query documents derived from ranking scores.

    Parameters:
    -----------
    queries: Queries
        The set of queries with documents.

    ranking_scores: array, shape = (n_documents,)
        A ranking score for each document in the set of queries.

    metric: Metric
        The evaluation metric, ffrom which the lambdas and the weights
        are to be computed.

    output_lambdas: array, shape=(n_documents,)
        Computed lambdas for every document.

    output_weights: array, shape=(n_documents,)
        Computed weights for every document.

    scale_value: array, shape=(n_queries,) or None
        The precomputed metric scale value for every query.

    influences: array, shape=(n_max_relevance, n_max_relevance) or None
        Used to keep track of (proportional) contribution in lambdas
        of high relevant documents from low relevant documents.

    indices: array, shape=(n_documents, n_leaf_nodes) or None:
        The indices of terminal nodes which the documents fall into.
        This parameter can be used to recompute lambdas and weights
        after regression tree is built.

    n_jobs: integer, optional (default is 1)
        The number of workers, which are used to compute the lambdas
        and weights in parallel.

    Returns
    -------
    loss: float
        The LambdaMART loss of the rankings induced by the specified
        ranking scores.
    '''

    query_indptr = _get_partition_indices(0, len(queries), n_jobs)

    if relevance_scores is None:
        relevance_scores = queries.relevance_scores
        query_relevance_strides = queries.query_relevance_strides
    else:
        # These will be computed in `parallel_compute_lambdas_and_weights`
        # from the given `relevance_scores`.
        query_relevance_strides = None

    return sum(Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(parallel_compute_lambdas_and_weights, check_pickle=False)(
            query_indptr[i], query_indptr[i + 1], queries.query_indptr,
            ranking_scores, relevance_scores, queries.max_score,
            query_relevance_strides, metric.backend(), scale_values,
            influences, indices, query_weights, document_weights,
            output_lambdas, output_weights, random_state
        )
        for i in range(query_indptr.shape[0] - 1))
    )


def _estimate_newton_gradient_steps(estimator, queries, ranking_scores, metric,
                                    lambdas, weights, scale_values=None,
                                    relevance_scores=None, query_weights=None,
                                    document_weights=None, random_state=None,
                                    n_jobs=1):
    '''
    Compute n_iterations of Newton's method to estimate optimal gradient
    steps for each terminal node of the given regression tree. Note that
    random forest is not suported and calling the method with it will do
    only a single iteration.

    If n_iterations is None, the algorithm runs until convergence.

    Parameters:
    -----------
    estimator: DecisionTreeRegressor or RandomForestRegressor instance
        The regression tree/forest for which the gradient steps are computed.

    queries: Queries instance
        The query documents determine which terminal nodes of the tree the
        cresponding lambdas and weights fall down into.

    ranking_scores: array, shape = (n_documents,)
        A ranking score for each document in the set of queries.

    metric: Metric
        The evaluation metric, ffrom which the lambdas and the weights
        are to be computed.

    lambdas: array, shape = (n_documents,)
        The current 1st order derivatives of the implicit loss function.

    weights: array, shape = (n_documents,)
        The current 2nd order derivatives of the implicit loss function.

    scale_value: array, shape=(n_queries,) or None
        The precomputed metric scale value for every query.

    n_jobs: integer, optional (default is 1)
        The number of workers, which are used to compute the lambdas
        and weights in parallel.
    '''

    if isinstance(estimator, RandomForestRegressor):
        estimators = estimator.estimators_
    else:
        estimators = [estimator]

    for estimator in estimators:
        # Get the number of nodes (internal + terminal)
        # in the current regression tree.
        node_count = estimator.tree_.node_count

        indices = estimator.tree_.apply(
                    queries.feature_vectors).astype('int32')

        # To get correct weights for the gradients we need to recompute
        # them with the information about what terminal nodes the documents
        # fall into.
        #
        # NOTE: After getting consistently significantly better results
        #       without this "correction step", this this no longer used.
        # 
        # compute_lambdas_and_weights(queries, ranking_scores, metric,
        #                             lambdas, weights, scale_values,
        #                             relevance_scores, query_weights,
        #                             document_weights, None, indices,
        #                             random_state, n_jobs)

        gradients = np.bincount(indices, lambdas, node_count)

        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(gradients, np.bincount(indices, weights, node_count),
                      out=gradients)

        # Remove inf's and nas's from the tree.
        gradients[~np.isfinite(gradients)] = 0.0

        np.copyto(estimator.tree_.value, gradients.reshape(-1, 1, 1))


class LambdaMART(object):
    '''
    LambdaMART learning to rank model.

    Arguments:
    ----------
    n_estimators: int, optional (default is 1000)
        The number of regression tree estimators that will
        compose this ensemble model.

    shrinkage: float, optional (default is 0.1)
        The learning rate (a.k.a. shrinkage factor) that will
        be used to regularize the predictors (prevent them
        from making the full (optimal) Newton step.

    use_newton_method: bool, optional (default is True)
        Estimate the gradient step in each terminal node of regression
        trees using Newton-Raphson method.

    use_random_forest: int, optional (default is 0):
        If positive, specify the number of trees within the random forest
        which will be used for regression instead of a single tree.

    max_depth: int, optional (default is 5)
        The maximum depth of the regression trees. This parameter is ignored
        if `max_leaf_nodes` is specified (see description of `max_leaf_nodes`).

    max_leaf_nodes: int, optional (default is None)
        The maximum number of leaf nodes. If not None, the `max_depth`
        parameter will be ignored. The tree building strategy also changes
        from depth search first to best search first, which can lead to
        substantial decrease of training time.

    min_samples_split : int, optional (default is 2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default is 1)
        The minimum number of samples required to be at a leaf node.

    estopping: int, optional (default is 100)
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

    min_n_estimators: int, optional, default: 1
        The minimum number of estimators to train. This number of estimators
        will be trained regardless of the `self.n_estimators` and
        `self.estopping`, and even if the best performance on training
        (validation) queries was better for fewer models.

    presort: bool, optional, default: False
        Whether to presort the data to speed up the finding of best
        splits in fitting.

    Attributes:
    -----------
    training_performance: array of doubles
        The performance of the model measured after training each
        tree/forest regression estimator on training queries.

    validation_performance: array of doubles
        The performance of the model measured after training each
        tree/forest regression estimator on validation queries.
    '''
    def __init__(self, n_estimators=10000, shrinkage=0.1,
                 use_newton_method=True, use_random_forest=0,
                 max_depth=None, max_leaf_nodes=7, min_samples_split=2,
                 min_samples_leaf=1, estopping=100, max_features=None,
                 base_model=None, min_n_estimators=1, random_thresholds=False,
                 presort=False, n_jobs=1, random_state=None):
        self.estimators = []
        self.n_estimators = n_estimators
        self.min_n_estimators = min_n_estimators
        self.shrinkage = shrinkage
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.estopping = estopping
        self.base_model = base_model
        self.use_newton_method = use_newton_method
        self.use_random_forest = use_random_forest
        self.random_thresholds = random_thresholds
        self.n_jobs = _get_n_jobs(n_jobs)
        self.training_performance = None
        self.validation_performance = None
        self.best_performance = None
        self.presort = False
        self.random_state = check_random_state(random_state)

        # Force the use of newer version of scikit-learn.
        if int(sklearn.__version__.split('.')[1]) < 17:
            raise ValueError('LambdaMART is built on scikit-learn '
                             'implementation of regression trees '
                             'version 17. Please, update your'
                             'scikit-learn package before using RankPy.')

    def fit_tree(self, metric, queries, max_estimators=None):
        '''
        Train just a single tree and add it to the LambdaMART model.

        Parameters:
        -----------
        metric: metrics.AbstractMetric object
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized for this model.

        queries: Queries object
            The set of queries from which one LambdaMART tree will be trained.

        max_estimators: integer
            The maximum number of trees in the model. If a new tree is fitted
            the oldest is removed.
        '''
        # If the model contains at least one tree, compute the ranking scores.
        if len(self.estimators) > 0:
            ranking_scores = self.predict(queries, n_jobs=self.n_jobs)
        else:
            ranking_scores = np.zeros(queries.document_count(),
                                      dtype='float64')

        if not isinstance(self.estimators, deque):
            self.estimators = deque(self.estimators, maxlen=max_estimators)

        # If the metric used for training is normalized,
        # it is advantageous to precompute the scaling
        # factor for each query in advance.
        queries_scale_values = metric.compute_scale(queries)

        # The pseudo-responses (lambdas) for each document.
        queries_lambdas = np.empty(queries.document_count(), dtype='float64')

        # The optimal gradient descent step sizes for each document.
        queries_weights = np.empty(queries.document_count(), dtype='float64')

        # Not used.
        self.trace_lambdas = False
        self.trace_gradients = False
        self.trace_influences = False

        logger.info('Training of LambdaMART tree started.')

        # Computes the pseudo-responses (lambdas) and gradient step
        # factors (weights) for the current regression tree.
        compute_lambdas_and_weights(queries, ranking_scores, metric,
                                    queries_lambdas, queries_weights,
                                    queries_scale_values, None, None,
                                    None, None, None, self.random_state,
                                    n_jobs=self.n_jobs)

        # Build the predictor for the gradients of the loss
        # using either decision tree or random forest.
        if self.use_random_forest > 0:
            estimator = RandomForestRegressor(
                            n_estimators=self.use_random_forest,
                            max_depth=self.max_depth,
                            max_leaf_nodes=self.max_leaf_nodes,
                            max_features=self.max_features,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            n_jobs=self.n_jobs)
        else:
            if self.random_thresholds:
                estimator = ExtraTreeRegressor(
                                max_depth=self.max_depth,
                                max_leaf_nodes=self.max_leaf_nodes,
                                max_features=self.max_features,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf)
            else:
                estimator = DecisionTreeRegressor(
                                max_depth=self.max_depth,
                                max_leaf_nodes=self.max_leaf_nodes,
                                max_features=self.max_features,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf)

        # Train the regression tree.
        estimator.fit(queries.feature_vectors, queries_lambdas)

        # Estimate the optimal gradient steps using Newton's method.
        if self.use_newton_method:
            _estimate_newton_gradient_steps(
                estimator, queries, ranking_scores, metric, queries_lambdas,
                queries_weights, queries_scale_values, None, None, None,
                n_iterations=self.n_iterations, shrinkage=self.shrinkage,
                verbose=self.verbose, random_state=random_state,
                n_jobs=self.n_jobs)

        # Add the new tree(s) to the company.
        self.estimators.append(estimator)

        # Correct the number of trees.
        self.n_estimators = len(self.estimators)

        logger.info('Training of LambdaMART tree finished - %s:  %11.8f'
                    % (metric,
                       metric.evaluate_queries(queries, ranking_scores,
                                               scale=queries_scale_values)))

    def fit(self, metric, queries, validation=None, query_weights=None,
            validation_query_weights=None, trace=None):
        '''
        Train the LambdaMART model on the specified queries. Optinally,
        use the specified queries for finding an optimal number of trees
        using validation.

        Parameters:
        -----------
        metric: Metric instance
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.

        queries: Queries instance
            The set of queries from which this LambdaMART
            model will be trained.

        validation: Queries instance
            The set of queries used in validation for early stopping.

        query_weights: array of doubles, shape = (n_queries,), optional
            (default is None) The weight given to each training query,
            which is used to measure its importance. Queries with 0.0
            weight will never be used in training of the model.

        validation_query_weights: array of doubles, shape = (n_queries,),
                                  optional (default is None)
            The weight given to each validation query, which is used to
            measure its importance. Queries with 0.0 weight will never
            be used in validation of the model.

        trace: list of strings, optional (default is None)
            Supported values are: `lambdas`, `gradients`, and `influences`.
            Since the number of documents and estimators can be large it is
            not adviced to use the values together. When `lambdas` is given,
            then the true and estimated lambdas will be stored, and similarly,
            when `gradients` are given, then the true and estimated gradient
            will be stored. These two quantities differ only if the Newton
            method is used for estimating the gradient steps. Use `influences`
            if you want to track (proportional) contribution of lambdas
            from lower relevant documents on high relevant ones.
        '''

        if self.base_model is None:
            training_scores = np.zeros(queries.document_count(),
                                       dtype='float64')
        else:
            training_scores = np.ascontiguousarray(
                                  self.base_model.predict(queries,
                                                          n_jobs=self.n_jobs),
                                  dtype='float64')

        # Give a weight to each document in a query with non-zero weight.
        if query_weights is None:
            document_weights = None
            nnz_document_weights_mask = None
        else:
            # Check the weight array shape and dtype.
            if (getattr(query_weights, 'dtype', None) != 'float64' or
                not query_weights.flags.contiguous):
                query_weights = np.ascontiguousarray(query_weights, dtype='float64')

            if query_weights.shape != (len(queries), ):
                raise ValueError('query weights array shape != (%d, )'
                                 % len(queries))

            if (query_weights < 0.0).any():
                raise ValueError('query weights must non-negative')

            document_weights = np.zeros(queries.document_count(),
                                        dtype='float64')

            # Set document weights for documents of queries
            # with non-zero weight to 1.0.
            for i, qw in enumerate(query_weights):
                if qw > 0.0:
                    document_weights[queries.qds[i]] = 1.0

            nnz_document_weights_mask = (document_weights > 0.0)

        # If the metric used for training is normalized, it is advantageous
        # to precompute the scaling factor for each query in advance.
        training_scale_values = metric.compute_scale(queries, query_weights)

        # Keep the training performance of LambdaMART
        # for every stage of training.
        self.training_performance = np.empty(self.n_estimators,
                                             dtype='float64')

        self.training_losses = np.zeros(self.n_estimators,
                                        dtype='float64')

        # The pseudo-responses (lambdas) for each document.
        training_lambdas = np.empty(queries.document_count(), dtype='float64')

        # The optimal gradient descent step sizes for each document.
        training_weights = np.empty(queries.document_count(), dtype='float64')

        # The lambdas and predictions may be kept for late analysis.
        if trace is not None:
            # Create temporary directory for traced data.
            TEMP_DIRECTORY_NAME = mkdtemp(prefix='lambdamart.trace.data.tmp',
                                          dir='.')

            logger.info('Created temporary directory (%s) for traced data.'
                        % TEMP_DIRECTORY_NAME)

            self.trace_lambdas = trace.count('lambdas') > 0
            self.trace_gradients = trace.count('gradients') > 0
            self.trace_influences = trace.count('influences') > 0

            # The pseudo-responses (lambdas) for each document:
            # the true and estimated values.
            if self.trace_lambdas:
                # Use memory mapping to store large matrices.
                self.stage_training_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
                self.stage_training_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.lambdas.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))

                if validation is not None:
                    self.stage_validation_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))
                    self.stage_validation_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.lambdas.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))

            if self.trace_gradients and not self.use_newton_method:
                warnings.warn('tracing gradients is possible only if '
                              'use_newton_method is True -- trace ignored')
                self.trace_gradients = False

            # The (loss) gradient steps for each query-document pair:
            # the true and estimated by the regression trees.
            if self.trace_gradients:
                # Use memory mapping to store large matrices.
                self.stage_training_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
                self.stage_training_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))

                if validation is not None:
                    self.stage_validation_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))
                    self.stage_validation_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation.document_count()))

            if self.trace_influences:
                self.stage_training_influences = np.zeros((self.n_estimators, queries.highest_relevance() + 1, queries.highest_relevance() + 1), dtype='float64')

                if validation is not None:
                    self.stage_validation_influences = np.zeros((self.n_estimators, validation.highest_relevance() + 1, validation.highest_relevance() + 1), dtype='float64')

                # Can work only in single threaded mode.
                if self.n_jobs > 1:
                    warnings.warn('cannot use multi-threaded training while '
                                  'tracing influences -- setting n_jobs to 1')
                    self.n_jobs = 1

            # Used when the model is saved to get rid of it.
            self.tmp_directory = TEMP_DIRECTORY_NAME
        else:
            self.trace_lambdas = False
            self.trace_gradients = False
            self.trace_influences = False

        # Initialize the same components for validation
        # queries as the training queries.
        if validation is not None:
            if validation_query_weights is None:
                validation_document_weights = None
                nnz_validation_document_weights_mask = None
            else:
                # Check the weight array shape and dtype.
                if (getattr(validation_query_weights, 'dtype', None) != 'float64' or
                    not validation_query_weights.flags.contiguous):
                    validation_query_weights, = np.ascontiguousarray(validation_query_weights,
                                                                     dtype='float64')

                if validation_query_weights.shape != (len(validation), ):
                    raise ValueError('validation query weights array '
                                     'shape != (%d, )' % len(validation))

                if (validation_query_weights < 0.0).any():
                    raise ValueError('validation query weights must '
                                     'be non-negative')

                validation_document_weights = np.zeros(validation.document_count(),
                                                       dtype='float64')

                # Set document weights for documents of validation
                # queries with non-zero weight to 1.0.
                for i, qw in enumerate(validation_query_weights):
                    if qw > 0.0:
                        validation_document_weights[validation.qds[i]] = 1.0

            nnz_validation_document_weights_mask = (validation_document_weights > 0.0)

            validation_scale_values = metric.compute_scale(validation,
                                                           validation_query_weights)

            # Keep the validation performance of LambdaMART
            # for every stage of training.
            self.validation_performance = np.empty(self.n_estimators,
                                                   dtype='float64')

            self.validation_losses = np.zeros(self.n_estimators,
                                              dtype='float64')

            if self.base_model is None:
                validation_scores = np.zeros(validation.document_count(),
                                             dtype='float64')
            else:
                validation_scores = np.ascontiguousarray(
                                        self.base_model.predict(validation),
                                        dtype='float64')

            if self.trace_lambdas or self.trace_influences:
                # The pseudo-responses (lambdas) for each document
                # in validation queries.
                self.validation_lambdas = np.empty(validation.document_count(),
                                                   dtype='float64')
                # The optimal gradient descent step sizes for each document
                # in validation queries.
                self.validation_weights = np.empty(validation.document_count(),
                                                   dtype='float64')

        # Presort feature values for faster training of the regression trees?
        if self.presort:
            feature_vectors_idx_sorted = \
                np.asfortranarray(np.argsort(queries.feature_vectors, axis=0),
                                  dtype='int32')
        else:
            feature_vectors_idx_sorted = None

        # The best iteration index and performance value
        # on validation (or training) queries.
        best_performance = -np.inf
        best_performance_k = -1

        # How many iterations the performance has not improved
        # on validation (or training) queries.
        performance_not_improved = 0

        logger.info('Training of LambdaMART model has started.')

        self.n_estimators = max(self.n_estimators, self.min_n_estimators)

        # Iteratively build a sequence of regression trees.
        for k in xrange(self.n_estimators):
            training_influences = self.stage_training_influences[k] if self.trace_influences else None

            # Computes the pseudo-responses (lambdas) and gradient step
            # factors (weights) for the current regression tree.
            self.training_losses[k] = compute_lambdas_and_weights(
                                            queries, training_scores, metric,
                                            training_lambdas, training_weights,
                                            training_scale_values, None,
                                            query_weights, document_weights,
                                            training_influences,
                                            random_state=self.random_state,
                                            n_jobs=self.n_jobs)

            # Build the predictor for the gradients of the loss using either
            # decision tree or random forest.
            if self.use_random_forest > 0:
                estimator = RandomForestRegressor(
                                n_estimators=self.use_random_forest,
                                max_depth=self.max_depth,
                                max_leaf_nodes=self.max_leaf_nodes,
                                max_features=self.max_features,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                n_jobs=self.n_jobs)

                # Train the regression forest.
                estimator.fit(queries.feature_vectors, training_lambdas,
                              sample_weight=document_weights)

            else:
                if self.random_thresholds:
                    estimator = ExtraTreeRegressor(
                                    max_depth=self.max_depth,
                                    max_leaf_nodes=self.max_leaf_nodes,
                                    max_features=self.max_features,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf)
                else:
                    estimator = DecisionTreeRegressor(
                                    max_depth=self.max_depth,
                                    max_leaf_nodes=self.max_leaf_nodes,
                                    max_features=self.max_features,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf)

                # Train the regression tree.
                estimator.fit(queries.feature_vectors, training_lambdas,
                              sample_weight=document_weights,
                              X_idx_sorted=feature_vectors_idx_sorted)

            # Store the estimated lambdas for later analysis (if wanted).
            if self.trace_lambdas:
                np.copyto(self.stage_training_lambdas_truth[k],
                          training_lambdas)

                np.copyto(self.stage_training_lambdas_predicted[k],
                          estimator.predict(queries.feature_vectors))

                # Set training lambdas of documents with 0 weight to
                # NaN indicating that they were never computed.
                if nnz_document_weights_mask is not None:
                    self.stage_training_lambdas_truth[k, nnz_document_weights_mask] = np.nan

            # Store the true and estimated gradients for later analysis.
            if self.trace_gradients:
                with np.errstate(divide='ignore', invalid='ignore'):
                    np.copyto(self.stage_training_gradients_truth[k],
                              training_lambdas)
                    np.divide(self.stage_training_gradients_truth[k],
                              training_weights,
                              out=self.stage_training_gradients_truth[k])
                    self.stage_training_gradients_truth[k, ~np.isfinite(self.stage_training_gradients_truth[k])] = 0.0

            if validation is not None:
                if self.trace_lambdas or self.trace_influences:
                    validation_influences = self.stage_validation_influences[k] if self.trace_influences else None

                    self.validation_losses[k] = \
                        compute_lambdas_and_weights(validation,
                                                    validation_scores, metric,
                                                    self.validation_lambdas,
                                                    self.validation_weights,
                                                    validation_scale_values,
                                                    None,
                                                    validation_query_weights,
                                                    validation_document_weights,
                                                    validation_influences,
                                                    random_state=self.random_state,
                                                    n_jobs=self.n_jobs)

                if self.trace_lambdas:
                    np.copyto(self.stage_validation_lambdas_truth[k],
                              self.validation_lambdas)

                    np.copyto(self.stage_validation_lambdas_predicted[k],
                              estimator.predict(validation.feature_vectors))

                    # Set validation lambdas of documents with 0 weight to
                    # NaN indicating that they were never computed.
                    if nnz_validation_document_weights_mask is not None:
                        self.stage_validation_lambdas_truth[k, nnz_validation_document_weights_mask] = np.nan

                if self.trace_gradients:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        np.copyto(self.stage_validation_gradients_truth[k],
                                  self.validation_lambdas)
                        np.divide(self.stage_validation_gradients_truth[k],
                                  self.validation_weights,
                                  out=self.stage_validation_gradients_truth[k])
                        self.stage_validation_gradients_truth[k, ~np.isfinite(self.stage_validation_gradients_truth[k])] = 0.0

            # Estimate the optimal gradient steps using Newton's method.
            if self.use_newton_method:
                _estimate_newton_gradient_steps(estimator, queries,
                                                training_scores, metric,
                                                training_lambdas,
                                                training_weights,
                                                training_scale_values,
                                                None, query_weights,
                                                document_weights,
                                                random_state=self.random_state,
                                                n_jobs=self.n_jobs)

                # Store the true and estimated gradients for later analysis.
                if self.trace_gradients:
                    np.copyto(self.stage_training_gradients_predicted[k],
                              estimator.predict(queries.feature_vectors))

                    if validation is not None:
                        np.copyto(self.stage_validation_gradients_predicted[k],
                                  estimator.predict(validation.feature_vectors))

            # Update the document scores using the new gradient predictor.
            if self.trace_gradients:
                training_scores += self.shrinkage * self.stage_training_gradients_predicted[k]
            else:
                training_scores += self.shrinkage * estimator.predict(queries.feature_vectors)

            # Add the new tree(s) to the company.
            self.estimators.append(estimator)

            self.training_performance[k] = metric.evaluate_queries(
                                               queries, training_scores,
                                               scale=training_scale_values,
                                               weights=query_weights)

            if validation is None:
                logger.info('#%08d: %s (training): %11.8f (%11.8f)'
                            % (k + 1, metric, self.training_performance[k],
                               self.training_losses[k]))

            # If validation queries have been given, estimate the model
            # performance on them and decide whether the training should not
            # be stopped early due to no significant performanceimprovements.
            if validation is not None:
                if self.trace_gradients:
                    validation_scores += self.shrinkage * self.stage_validation_gradients_predicted[k]
                else:
                    validation_scores += self.shrinkage * self.estimators[-1].predict(validation.feature_vectors)

                self.validation_performance[k] = metric.evaluate_queries(
                                                     validation,
                                                     validation_scores,
                                                     scale=validation_scale_values,
                                                     weights=validation_query_weights)

                logger.info('#%08d: %s (training):   %11.8f (%11.8f)  | '
                            ' (validation):   %11.8f (%11.8f)'
                            % (k + 1, metric,
                               self.training_performance[k],
                               self.training_losses[k],
                               self.validation_performance[k],
                               self.validation_losses[k]))

                if self.validation_performance[k] > best_performance:
                    best_performance = self.validation_performance[k]
                    best_performance_k = k
                    performance_not_improved = 0
                else:
                    performance_not_improved += 1

            elif self.training_performance[k] > best_performance:
                    best_performance = self.training_performance[k]
                    best_performance_k = k

            if performance_not_improved >= self.estopping and self.min_n_estimators <= k + 1:
                logger.info('Stopping early since no improvement on '
                            'validation queries has been observed for '
                            '%d iterations (since iteration %d)'
                            % (self.estopping, best_performance_k + 1))
                break

        logger.info('Final model performance (%s) on %s queries: %11.8f'
                    % (metric,
                       'training' if validation is None else 'validation',
                       best_performance))

        # Make sure the model has the wanted size.
        best_performance_k = max(best_performance_k, self.min_n_estimators - 1)

        # Leave the estimators that led to the best performance,
        # either on training or validation set.
        del self.estimators[best_performance_k + 1:]

        # Correct the number of trees.
        if self.n_estimators != len(self.estimators):
            self.n_estimators = len(self.estimators)
            logger.info('Setting the number of trees of the model to %d.'
                        % self.n_estimators)

        # Set these for further inspection.
        self.training_performance = np.resize(self.training_performance, k + 1)
        self.training_losses = np.resize(self.training_losses, k + 1)

        if validation is not None:
            self.validation_performance = \
                np.resize(self.validation_performance, k + 1)

        self.best_performance = best_performance

        if self.trace_influences:
            self.stage_training_influences = \
                np.resize(self.stage_training_influences,
                          (k + 1, queries.highest_relevance() + 1,
                           queries.highest_relevance() + 1))

            influences_normalizer = \
                np.bincount(queries.relevance_scores,
                            minlength=queries.highest_relevance() + 1)
            influences_normalizer = \
                np.triu(np.ones((queries.highest_relevance() + 1, 1)) *
                        influences_normalizer, 1)
            influences_normalizer += influences_normalizer.T

            # Normalize training influences appropriately.
            with np.errstate(divide='ignore', invalid='ignore'):
                self.stage_training_influences /= influences_normalizer
                self.stage_training_influences[np.isnan(self.stage_training_influences)] = 0.0

            if validation is not None:
                self.stage_validation_influences = \
                    np.resize(self.stage_validation_influences,
                              (k + 1, validation.highest_relevance() + 1,
                               validation.highest_relevance() + 1))

                influences_normalizer = \
                    np.bincount(validation.relevance_scores,
                                minlength=validation.highest_relevance() + 1)
                influences_normalizer = \
                    np.triu(np.ones((validation.highest_relevance() + 1, 1)) *
                            influences_normalizer, 1)
                influences_normalizer += influences_normalizer.T

                # Normalize training influences appropriately.
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.stage_validation_influences /= influences_normalizer
                    self.stage_validation_influences[np.isnan(self.stage_validation_influences)] = 0.0

        logger.info('Training of LambdaMART model has finished.')

    @staticmethod
    def __predict(trees, shrinkage, feature_vectors, output):
        for tree in trees:
            output += tree.predict(feature_vectors)
        output *= shrinkage

    def predict(self, queries, n_jobs=1):
        '''
        Predict the ranking score for each individual document
        in the given queries.

        n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        if len(self.estimators) == 0:
            raise ValueError('the model has not been trained yet')

        if self.base_model is not None:
            predictions = np.ascontiguousarray(
                              self.base_model.predict(queries,
                                                      n_jobs=n_jobs),
                              dtype='float64')
        else:
            predictions = np.zeros(queries.document_count(), dtype='float64')

        indices = _get_partition_indices(0, queries.document_count(),
                                         self.n_jobs)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(parallel_helper, check_pickle=False)(
                LambdaMART, '_LambdaMART__predict',
                self.estimators, self.shrinkage,
                queries.feature_vectors[indices[i]:indices[i + 1]],
                predictions[indices[i]:indices[i + 1]]
            )
            for i in range(indices.size - 1)
        )

        return predictions

    def predict_rankings(self, queries, compact=False, n_jobs=1):
        '''
        Predict rankings of the documents for the given queries.

        If `compact` is set to True then the output will be one
        long 1d array containing the rankings for all the queries
        instead of a list of 1d arrays.

        The compact array can be subsequently index using query
        index pointer array, see `queries.query_indptr`.

        query: Query
            The query whose documents should be ranked.

        compact: bool
            Specify to return rankings in compact format.

         n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        # Predict the ranking scores for the documents.
        predictions = self.predict(queries, n_jobs)

        rankings = np.zeros(queries.document_count(), dtype=np.intc)

        ranksort_queries(queries.query_indptr, predictions, rankings)

        if compact or len(queries) == 1:
            return rankings
        else:
            return np.array_split(rankings, queries.query_indptr[1:-1])

    def feature_importances(self):
        '''
        Return the feature importances.
        '''
        if len(self.estimators) == 0:
            raise ValueError('the model has not been trained yet')

        importances = Parallel(n_jobs=self.n_jobs, backend="threading")(
                          delayed(getattr, check_pickle=False)(
                              tree, 'feature_importances_'
                          )
                          for tree in self.estimators
                      )

        return sum(importances) / self.n_estimators

    @classmethod
    def load(cls, filepath, mmap='r', load_traced=False):
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

        if not load_traced:
            return obj

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
            raise ValueError('cannot save model to text when it is based on random forest')

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

    def __del__(self):
        '''
        Cleanup the temporary directory for traced lambdas and gradients.
        '''
        # Get rid of the temporary directory and all the memory-mapped arrays.
        if hasattr(self, 'tmp_directory'):
            if self.trace_lambdas:
                del self.stage_training_lambdas_truth
                del self.stage_training_lambdas_predicted

                if hasattr(self, 'validation_performance'):
                    del self.stage_validation_lambdas_truth
                    del self.stage_validation_lambdas_predicted

            if self.trace_gradients:
                del self.stage_training_gradients_truth
                del self.stage_training_gradients_predicted

                if hasattr(self, 'validation_performance'):
                    del self.stage_validation_gradients_truth
                    del self.stage_validation_gradients_predicted

            logger.info('Deleting temporary directory (%s) for traced data.'
                        % self.tmp_directory)
            rmtree(self.tmp_directory)
            del self.tmp_directory

    def __str__(self):
        '''
        Return textual representation of the LambdaMART model.
        '''
        return ('LambdaMART(trees=%d, max_depth=%s, max_leaf_nodes=%s, '
                'shrinkage=%.2f, max_features=%s, use_newton_method=%s)'
                % (self.n_estimators,
                   self.max_depth if self.max_leaf_nodes is None else '?',
                   '?' if self.max_leaf_nodes is None else self.max_leaf_nodes,
                   self.shrinkage,
                   'all' if self.max_features is None else self.max_features,
                   self.use_newton_method))
