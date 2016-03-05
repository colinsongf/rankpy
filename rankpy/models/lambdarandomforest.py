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

import numpy as np

from ..externals.joblib import Parallel, delayed

from sklearn.utils import check_random_state

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from ..utils import pickle
from ..utils import unpickle
from ..utils import parallel_helper
from ..utils import _get_partition_indices
from ..utils import _get_n_jobs
from ..utils import aslist

from ..metrics import MetricFactory
from ..metrics._utils import ranksort_queries

from .lambdamart import compute_lambdas_and_weights
from .lambdamart import compute_newton_gradient_steps


logger = logging.getLogger(__name__)


def parallel_build_trees(tree_index, tree, n_trees, metric, queries,
                         query_scales, query_weights, use_newton_method,
                         bootstrap, subsample_queries, subsample_documents,
                         seed, sigma=0.0, validation_queries=None,
                         validation_scores=None):
    '''
    Train a regression tree to optimize the evaluation metric for the specified
    queries using the LambdaMART's 'lambdas'.

    Parameters:
    -----------
    tree_index : int
        The index of the tree, used to log progress.

    tree: DecisionTreeRegressor or ExtraTreeRegressor instance
        The regression tree to train.

    n_trees : int
        The total number of trees (planned) to be trained.

    metric : Metric instance
        The evaluation metric used as a utility function indirectly optimized
        by the tree using LambdaMART's 'lambdas'.

    queries : Queries instance
        The set of queries used for training.

    query_weights : array of floats, shape = [n_queries]
        The weight given to each training query, which is used to
        measure its importance. Queries with 0.0 weight will never
        be used in training.

    query_scales : array of floats, shape = [n_queries]
        The precomputed ideal metric values for the queries.

    use_newton_method : bool
        If True, the terminal node prediction values will be re-estimated
        using Newton-Raphson method.

    bootstrap : bool
        Specify to use bootstrap sample from the queries.

    subsample_queries : float
        The probability of including a query into the training.

    subsample_documents : float
        The probability of including a document into the training.

    seed : int
        Used for initialization of random number generator.

    sigma : float (default=0.)
        The ranking scores for documents are sampled from standard normal,
        this can be used to controll the variance of the scores. By default,
        all ranking scores are 0.

    validation_queries : Queries instance
        The set of queries used for validation.

    validation_scores : array of floats, shape = [n_validation_documents]
        The ranking scores for each document in validation queries.
    '''
    random_state = check_random_state(seed)

    if validation_queries is None:
        logger.info('Started fitting regression tree %d of %d.'
                    % (tree_index, n_trees))
    else:
        logger.info('Started fitting %d-th regression tree.' % tree_index)

    # Pick a random ranking scores from the standard Gaussian
    # with controlled variance
    training_scores = sigma * random_state.randn(queries.document_count())

    # The 1st order derivatives of the implicit loss derived from the metric.
    training_lambdas = np.empty(queries.document_count(), dtype=np.float64)

    # The 2nd order derivatives of the implicit loss derived from the metric.
    training_weights = np.empty(queries.document_count(), dtype=np.float64)

    if subsample_queries != 1.0:
        if query_weights is not None:
            # Need to make copy not to interfere with the outside world.
            query_weights = query_weights.copy()
        else:
            query_weights = np.ones(len(queries), dtype='float64')

        # Subsample the queries.
        query_weights *= random_state.choice([0.0, 1.0],
                                             size=len(query_weights),
                                             p=[1 - subsample_queries, subsample_queries])

    if bootstrap:
        if query_weights is not None:
            # Need to make copy not to interfere with the outside world.
            query_weights = query_weights.copy()
        else:
            query_weights = np.ones(len(queries), dtype='float64')

        nnz_mask = (query_weights > 0.0)
        nnz_count = nnz_mask.sum()

        query_weights[nnz_mask] = np.bincount(random_state.randint(0,
                                                                   nnz_count,
                                                                   nnz_count),
                                              minlength=nnz_count)

    if query_weights is not None:
        document_weights = np.zeros(queries.document_count(), dtype='float64')
        document_weights[queries.qdie[query_weights > 0.0].dmask] = 1.0
    else:
        document_weights = None

    if subsample_documents != 1.0:
        if document_weights is None:
            document_weights = np.ones(queries.document_count(),
                                       dtype='float64')

        # Subsample the documents.
        document_weights *= random_state.choice([0.0, 1.0],
                                                size=len(document_weights),
                                                p=[1 - subsample_documents,
                                                   subsample_documents])

        query_scales = metric.compute_scaling(queries,
                                              query_weights=query_weights,
                                              document_weights=document_weights)

    # Compute LambdaMART lambdas and weights.
    compute_lambdas_and_weights(queries, training_scores, metric,
                                training_lambdas, training_weights,
                                query_scales=query_scales,
                                query_weights=query_weights,
                                document_weights=document_weights,
                                random_state=random_state)

    # Train the regression tree.
    tree.fit(queries.feature_vectors, training_lambdas,
             sample_weight=document_weights,
             check_input=False)

    if use_newton_method:
        # Re-estimate the prediction value in terminal nodes.
        compute_newton_gradient_steps(tree, queries, training_scores, metric,
                                      training_lambdas, training_weights,
                                      query_scales=query_scales,
                                      query_weights=query_weights,
                                      document_weights=document_weights,
                                      random_state=random_state,
                                      recompute=True)

    if validation_queries is not None:
        if validation_scores is not None:
            validation_scores[:] = tree.predict(
                                        validation_queries.feature_vectors,
                                        check_input=False)
        else:
            raise ValueError("'validation_scores' cannot be None when "
                             "'validation_queries' is not None")

    return tree


class LambdaRandomForest(object):
    '''
    LambdaRandomForest learning to rank model.

    Parameters
    ----------
    metric : string, optional (default="NDCG")
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.
            Supported metrics are "NDCG", "WTA", "ERR", with an optional
            suffix "@{N}", where {N} can be any positive integer.

    n_estimators : int, optional (default=100)
        The maximum number of regression trees that will be trained.

    max_depth : int or None, optional (default=None)
        The maximum depth of the regression trees. This parameter is ignored
        if `max_leaf_nodes` is specified (see description of `max_leaf_nodes`).

    max_leaf_nodes : int or None, optional (default=None)
        The maximum number of leaf nodes. If not None, the `max_depth`
        parameter will be ignored. The tree building strategy also changes
        from depth search first to best search first, which can lead to
        substantial decrease in training time.

    max_features : int, float, or None, optional (default=None)
        The maximum number of features that is considered for splitting when
        regression trees are being built. If float is given it is interpreted
        as a percentage. If None, all feature will be used.

    min_samples_split : int, optional (default=2)
        The minimum number of documents required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of documents required to be at a terminal node.

    use_newton_method : bool, optional (default=True)
        Estimate the gradient step in each terminal node of regression
        trees using Newton-Raphson method.

    random_thresholds : bool, optional (default=False)
        If True, extremely randomized trees will be used instead of 'clasic'
        regression tree predictors.

    min_n_estimators: int, optional (default=1)
        The minimum number of regression trees that will be trained regardless
        of the `n_estimators` and `estopping` values. Beware that using this
        option may lead to suboptimal performance of the model.

    subsample_queries : float, optional (default=1.0)
        The probability of considering a query for training.

    subsample_documents : float, optional (default=1.0)
        The probability of including a document into the training.

    estopping : int or None, optional (default=None)
        If the last `estopping` number of trained regression trees did not
        lead to improvement in the metric on the training or validation
        queries (if used), the training is stopped early. If None,
        `n_estimators` is trained.

    n_jobs : int, optional (default=1)
        The number of working sub-processes that will be spawned to train
        regression trees. If -1, the number of CPUs will be used.

    random_state : int or RandomState instance, optional (default=None)
        The random number generator used for internal randomness, such
        as subsampling, etc.
    '''
    def __init__(self, metric='NDCG', n_estimators=100, max_depth=None,
                 max_leaf_nodes=None, max_features=None, min_samples_split=2,
                 min_samples_leaf=1, use_newton_method=True,
                 random_thresholds=False, min_n_estimators=1,
                 subsample_queries=1.0, subsample_documents=1.0,
                 bootstrap=True, sigma=0.0, estopping=None, n_jobs=1,
                 random_state=None):
        self.estimators = []
        self.n_estimators = n_estimators
        self.min_n_estimators = min_n_estimators
        self.metric = metric
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_thresholds = random_thresholds
        self.use_newton_method = use_newton_method
        self.subsample_queries = subsample_queries
        self.subsample_documents = subsample_documents
        self.bootstrap = bootstrap
        self.sigma = sigma
        self.estopping = n_estimators if estopping is None else estopping
        self.n_jobs = _get_n_jobs(n_jobs)
        self.random_state = check_random_state(random_state)

        # Force the use of newer version of scikit-learn.
        if int(sklearn.__version__.split('.')[1]) < 17:
            raise ValueError('LambdaMART is built on scikit-learn '
                             'implementation of regression trees '
                             'version 17. Please, update your'
                             'scikit-learn package before using RankPy.')


    def fit(self, training_queries, training_query_weights=None,
            validation_queries=None, validation_query_weights=None):
        '''
        Train a LambdaRandomForest model on given training queries.
        Optionally, use validation queries for finding an optimal
        number of trees using early stopping.

        Parameters
        ----------
        training_queries : Queries instance
            The set of queries from which the model will be trained.

        training_query_weights : array of floats, shape = [n_queries],
                                 or None
            The weight given to each training query, which is used to
            measure its importance. Queries with 0.0 weight will never
            be used in training.

        validation_queries : Queries instance or None
            The set of queries used for early stopping.

        validation_query_weights : array of floats, shape = [n_queries]
                                   or None
            The weight given to each validation query, which is used to
            measure its importance. Queries with 0.0 weight will never
            be used in validation.

        Returns
        -------
        self : object
            Returns self.
        '''
        metric = MetricFactory(self.metric,
                               queries=aslist(training_queries,
                                              validation_queries),
                               random_state=self.random_state)

        # If the metric used for training is normalized, it is advantageous
        # to precompute the scaling factor for each query in advance.
        training_query_scales = metric.compute_scaling(training_queries,
                                                       query_weights=training_query_weights)

        if validation_queries is None:
            validation_queries = training_queries
            validation_query_scales = training_query_scales.copy()
        else:
            validation_query_scales = metric.compute_scaling(validation_queries,
                                                             query_weights=validation_query_weights)

        # The first row is reserved for the ranking scores computed
        # in the previous fold of regression trees, see below, how
        # the tree ensemble is evaluated, that is why +1.
        validation_ranking_scores = np.zeros((self.n_jobs + 1,
                                              validation_queries.document_count()),
                                             dtype=np.float64)

        logger.info('Training of LambdaRandomForest model has started.')

        estimators = []

        if self.random_thresholds:
            for k in range(self.n_estimators):
                estimators.append(ExtraTreeRegressor(
                                    max_depth=self.max_depth,
                                    max_leaf_nodes=self.max_leaf_nodes,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features=self.max_features,
                                    random_state=self.random_state))
        else:
            for k in range(self.n_estimators):
                estimators.append(DecisionTreeRegressor(
                                    max_depth=self.max_depth,
                                    max_leaf_nodes=self.max_leaf_nodes,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features=self.max_features,
                                    random_state=self.random_state))

        # Best performance and index of the last tree.
        best_performance = -np.inf
        best_performance_k = -1

        # Counts how many trees have been trained since the last
        # improvement on the validation set.
        performance_not_improved = 0

        # Partition the training into a proper number of folds
        # to benefit from the parallelization at best.
        if self.n_estimators > self.n_jobs:
            estimator_indices = np.array_split(
                        np.arange(self.n_estimators, dtype=np.intc),
                        (self.n_estimators + self.n_jobs - 1) / self.n_jobs)
        else:
            estimator_indices = [np.arange(self.n_estimators)]

        for fold_indices in estimator_indices:
            # Train all trees in the current fold...
            fold_estimators = \
                Parallel(n_jobs=self.n_jobs, backend='threading')(
                    delayed(parallel_build_trees, check_pickle=False)(
                        i, estimators[i], self.n_estimators, metric.copy(),
                        training_queries, training_query_scales,
                        training_query_weights, self.use_newton_method,
                        self.bootstrap, self.subsample_queries,
                        self.subsample_documents,
                        self.random_state.randint(1, np.iinfo('i').max),
                        self.sigma, validation_queries,
                        validation_ranking_scores[i - fold_indices[0] + 1])
                    for i in fold_indices)

            self.estimators.extend(fold_estimators)

            # Compute the ranking score of validation queries for every
            # new tree that has been just trained.
            np.cumsum(validation_ranking_scores[:(len(fold_indices) + 1)],
                      out=validation_ranking_scores[:(len(fold_indices) + 1)],
                      axis=0)

            for i, ranking_scores in enumerate(validation_ranking_scores[1:, :]):
                # Get the performance of the current model consisting
                # of `fold_indices[0] + i + 1` number of trees.
                validation_performance = metric.evaluate_queries(
                                            validation_queries, ranking_scores,
                                            scales=validation_query_scales)

                logger.info('#%08d: %s (%s): %11.8f'
                            % (fold_indices[i],
                               'training' if validation_queries is training_queries else 'validation',
                               metric, validation_performance))

                if validation_performance > best_performance:
                    best_performance = validation_performance
                    best_performance_k = fold_indices[i]
                    performance_not_improved = 0
                else:
                    performance_not_improved += 1

                if (performance_not_improved >= self.estopping and
                    self.min_n_estimators <= fold_indices[i] + 1):
                    break

            if (performance_not_improved >= self.estopping and
                self.min_n_estimators <= fold_indices[i] + 1):
                logger.info('Stopping early since no improvement on %s '
                            'queries has been observed for %d iterations '
                            '(since iteration %d)'
                             % ('training' if validation_queries is training_queries else 'validation',
                                self.estopping, best_performance_k + 1))
                break

            # Copy the last ranking scores for the next validation "fold".
            validation_ranking_scores[0, :] = validation_ranking_scores[len(fold_indices), :]

        if validation_queries is not training_queries:
            logger.info('Final model performance (%s) on validation queries: '
                        '%11.8f' % (metric, best_performance))
        else:
            logger.info('Final model performance (%s) on training queries: '
                        '%11.8f' % (metric, best_performance))

        # Make sure the model has the wanted size.
        best_performance_k = max(best_performance_k, self.min_n_estimators - 1)

        # Leave the estimators that led to the best performance,
        # either on training or validation set.
        del self.estimators[best_performance_k + 1:]

        # Correct the number of trees.
        self.n_estimators = len(self.estimators)

        self.best_performance = best_performance

        logger.info('Training of LambdaRandomForest model has finished.')

        return self

    @staticmethod
    def __predict(trees, feature_vectors, output):
        for tree in trees:
            output += tree.predict(feature_vectors, check_input=False)

    def predict(self, queries, compact=True, n_jobs=1):
        '''
        Predict the ranking score for each individual document
        in the given queries.

        If `compact` is set to True then the output will be one
        long 1d array containing the rankings for all the queries
        instead of a list of 1d arrays.

        The compact array can be subsequently index using query
        index pointer array, see `queries.query_indptr`.

        query: Queries instance
            The query whose documents should be ranked.

        compact : boolean, optional (default=False)
            If True, a single array made of concatenated rankings is returned
            If False, the returned rankings will be returned as a list filled
            with rankings of individual queries. .

         n_jobs : int, optional (default=1)
            The number of working threads that will be spawned to compute
            the rankings. If -1, the maximum number of available CPUs will
            be used.
        '''
        if len(self.estimators) == 0:
            raise ValueError('the model has not been trained yet')

        n_jobs = _get_n_jobs(n_jobs)

        predictions = np.zeros(queries.document_count(), dtype=np.float64)

        indices = _get_partition_indices(0, queries.document_count(),
                                         self.n_jobs)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(parallel_helper, check_pickle=False)(
                LambdaRandomForest, '_LambdaRandomForest__predict',
                self.estimators,
                queries.feature_vectors[indices[i]:indices[i + 1]],
                predictions[indices[i]:indices[i + 1]])
            for i in range(indices.size - 1)
        )

        predictions /= len(self.estimators)

        if compact or len(queries) == 1:
            return predictions
        else:
            return np.array_split(predictions, queries.query_indptr[1:-1])

    def predict_rankings(self, queries, compact=False, return_scores=False,
                         n_jobs=1):
        '''
        Predict rankings of the documents for the given queries.

        If `compact` is set to True then the output will be one
        long 1d array containing the rankings for all the queries
        instead of a list of 1d arrays.

        The compact array can be subsequently index using query
        index pointer array, see `queries.query_indptr`.

        query: Queries instance
            The query whose documents should be ranked.

        compact : boolean, optional (default=False)
            If True, a single array made of concatenated rankings is returned
            If False, the returned rankings will be returned as a list filled
            with rankings of individual queries.

        return_scores : boolean, optional (default=False)
            Indicates that the ranking scores, on which the returned rankings
            are based, should be returned as well.

         n_jobs : int, optional (default=1)
            The number of working threads that will be spawned to compute
            the rankings. If -1, the maximum number of available CPUs will
            be used.
        '''
        # Predict the ranking scores for the documents.
        predictions = self.predict(queries, n_jobs)

        rankings = np.zeros(queries.document_count(), dtype=np.intc)

        ranksort_queries(queries.query_indptr, predictions, rankings)

        if compact or len(queries) == 1:
            return (rankings, predictions) if return_scores else rankings
        elif return_scores:
            return (np.array_split(rankings, queries.query_indptr[1:-1]),
                    np.array_split(predictions, queries.query_indptr[1:-1]))
        else:
            return np.array_split(rankings, queries.query_indptr[1:-1])

    def evaluate(self, queries, metric=None, out=None, n_jobs=1):
        '''
        Evaluate the performance of the model on the given queries.

        Parameters
        ----------
        queries : Queries instance
            Queries used for evaluation of the model.

        metric : string or None, optional (default=None)
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.
            Supported metrics are "DCG", NDCG", "WTA", "ERR", with an optional
            suffix "@{N}", where {N} can be any positive integer. If None,
            the model is evaluated with a metric for which it was trained.

        n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        out : array of floats, shape = [n_documents], or None
            If not None, it will be filled with the metric values
            for each query.
        '''
        if metric is None:
            metric = self.metric

        scores = self.predict(queries, n_jobs=_get_n_jobs(n_jobs))

        metric = MetricFactory(metric, queries=aslist(queries),
                               random_state=self.random_state)

        return metric.evaluate_queries(queries, scores, out=out)

    def feature_importances(self, n_jobs=1):
        '''
        Return the feature importances.
        '''
        if len(self.estimators) == 0:
            raise ValueError('the model has not been trained yet')

        importances = Parallel(n_jobs=_get_n_jobs(n_jobs),
                               backend="threading")(
                          delayed(getattr, check_pickle=False)(
                              tree, 'feature_importances_'
                          )
                          for tree in self.estimators
                      )

        return sum(importances) / self.n_estimators

    @classmethod
    def load(cls, filepath):
        '''
        Load the previously saved LambdaRandomForest model
        from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a LambdaRandomForest
            object will be loaded.
        '''
        logger.info("Loading %s object from %s"
                    % (cls.__name__, filepath))
        return unpickle(filepath)

    def save(self, filepath):
        '''
        Save te LambdaRandomForest model into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        logger.info("Saving %s object into %s"
                    % (self.__class__.__name__, filepath))
        pickle(self, filepath)

    def __str__(self):
        '''
        Return textual representation of the LambdaRandomForest model.
        '''
        return ('LambdaRandomForest(trees=%d, max_depth=%s, max_leaf_nodes=%s,'
                ' max_features=%s, use_newton_method=%s, bootstrap=%s)'
                % (self.n_estimators, self.max_depth if self.max_leaf_nodes is None else '?',
                   '?' if self.max_leaf_nodes is None else self.max_leaf_nodes,
                   'all' if self.max_features is None else self.max_features,
                   self.use_newton_method, self.bootstrap))
