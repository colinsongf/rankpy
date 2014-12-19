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

from ..externals.joblib import Parallel, delayed, cpu_count

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from ..utils import parallel_helper
from ..utils import pickle, unpickle
from ..metrics._utils import ranksort

from .lambdamart import compute_lambdas_and_weights
from .lambdamart import _estimate_newton_gradient_steps


logger = logging.getLogger(__name__)


def _parallel_build_trees_shuffle(tree_index, tree, n_trees, queries, metric, use_newton_method, scale_values, bootstrap,
                                  validation=None, validation_scale_values=None, validation_performance=None):
    ''' 
    Train the trees on the specified queries using the LambdaMART's lambdas computed
    using the specified metric.

    Parameters:
    -----------
    tree_index: int
        The index of the tree. Used only for logging progress.

    tree: DecisionTreeRegressor
        The regresion tree to train.

    n_trees: int
        The total number of trees that will be trained.

    queries: Queries object
        The set of queries from which the tree models will be trained.

    metric: Metric
        Specify evaluation metric which will be used as a utility
        function optimized by the tree (through pseudo-responses).

    use_newton_method: bool
        Estimate the gradient step in each terminal node of regression
        tree using Newton-Raphson method.

    scale_values: array of doubles, shape = (n_queries, )
        The precomputed ideal metric value for the specified queries.

    boostrap: bool
        Specify to use bootstrap sample from the lambdas computed
        from `reshuffled` documents.

    validation: Queries
        The set of queries used for validation.

    validation_scale_values: array of doubles, shape = (n_valid_queries, )
        The precomputed scale factors for metric values of the validation queries.

    validation_performance: array of doubles, shape = (n_estimators,)
        The array to store validation performance of each trained tree.
    '''
    # Note, that 0 scores does not mean the order of documents will not change.
    training_scores = np.zeros(queries.document_count(), dtype=np.float64)

    # The pseudo-responses (lambdas) for each document.
    training_lambdas = np.empty(queries.document_count(), dtype=np.float64)

    # The optimal gradient descent step sizes for each document.
    training_weights = np.empty(queries.document_count(), dtype=np.float64)

    logger.info('Started fitting LambdaDecisionTree %d of %d.' % (tree_index, n_trees))

    # Compute the pseudo-responses (lambdas) and gradient step sizes (weights).
    compute_lambdas_and_weights(queries, training_scores, metric, training_lambdas, training_weights,
                                 scale_values=scale_values)
    # Bootstrap?
    if bootstrap:
        n_lambdas = queries.document_count()
        sample_weight = np.bincount(np.random.randint(0, n_lambdas, n_lambdas), minlength=n_lambdas)
    else:
        sample_weight = None

    # Train the regression tree.
    tree.fit(queries.feature_vectors, training_lambdas, sample_weight=sample_weight, check_input=False)

    # Estimate the ('optimal') gradient step sizes using one iteration
    # of Newton-Raphson method.
    if use_newton_method:
       _estimate_newton_gradient_steps(tree, queries, training_lambdas, training_weights)

    if validation is not None:
        validation_performance[tree_index] = metric.evaluate_queries(validation, tree.predict(validation.feature_vectors),
                                                                     scale=validation_scale_values)
    return tree


def _parallel_build_trees_bootstrap(tree_index, tree, n_trees, queries, metric, scale_values, training_lambdas, training_weights,
                                    bootstrap, validation=None, validation_scale_values=None, validation_performance=None):
    ''' 
    Train a regression tree for the specified LambdaMART's lambdas, and if not None,
    use weights to optimize the gradient step in terminal nodes using Newton-Raphson
    method.

    Parameters:
    -----------
    tree_index: int
        The index of the tree. Used only for logging progress.

    tree: DecisionTreeRegressor
        The regresion tree to train.

    n_trees: int
        The total number of trees that will be trained.

    queries: Queries
        The set of queries from which the tree model is being
        trained.

    metric: Metric
        Specify evaluation metric which will be used as a utility
        function (i.e. metric of `goodness`) optimized by the trees.

    scale_values: array of doubles, shape = (n_queries, )
        The precomputed scale factors for metric values of the queries.

    training_lambdas: array of doubles, shape = (n_documents,)
        The precomputed lambdas for every document.

    training_weights: array of doubles, shape = (n_documents,)
        The precomputed weights for every documents.

    boostrap: bool
        Specify to use bootstrap sample from the given lambdas.

    validation: Queries
        The set of queries used for validation.

    validation_scale_values: array of doubles, shape = (n_valid_queries, )
        The precomputed scale factors for metric values of the validation queries.

    validation_performance: array of doubles, shape = (n_estimators,)
        The array to store validation performance of each trained tree.    
    '''
    logger.info('Started fitting LambdaDecisionTree %d of %d.' % (tree_index, n_trees))

    # Bootstrap?
    if bootstrap:
        n_lambdas = queries.document_count()
        sample_weight = np.bincount(np.random.randint(0, n_lambdas, n_lambdas), minlength=n_lambdas)
    else:
        sample_weight = None

    # Train the regression tree.
    tree.fit(queries.feature_vectors, training_lambdas, sample_weight=sample_weight, check_input=False)

    # Estimate the ('optimal') gradient step sizes using one iteration
    # of Newton-Raphson method.
    if training_weights is not None:
       _estimate_newton_gradient_steps(tree, queries, training_lambdas, training_weights)

    if validation is not None:
        validation_performance[tree_index] = metric.evaluate_queries(validation, tree.predict(validation.feature_vectors),
                                                                     scale=validation_scale_values)
    return tree


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

    min_samples_split : int, optional (default is 2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default is 1)
        The minimum number of samples required to be at a leaf node.

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
                 bootstrap=False, seed=None):
        self.estimators = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.use_newton_method = use_newton_method
        self.shuffle = shuffle
        self.bootstrap = bootstrap
        self.n_jobs = max(1, n_jobs if n_jobs >= 0 else n_jobs + cpu_count() + 1)
        self.trained = False
        self.seed = seed

        # `max_leaf_nodes` were introduced in version 15 of scikit-learn.
        if self.max_leaf_nodes is not None and int(sklearn.__version__.split('.')[1]) < 15:
            raise ValueError('cannot use parameter `max_leaf_nodes` with scikit-learn of version smaller than 15')


    def fit(self, metric, queries, validation=None):
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
        training_scale_values = metric.compute_scale(queries)

        if validation is not None:
            validation_scale_values = metric.compute_scale(validation)
            validation_performance = np.zeros(self.n_estimators, dtype=np.float64)
        else:
            validation_scale_values = None
            validation_performance = None

        logger.info('Training of LambdaRandomForest model has started.')

        estimators = []

        for k in range(self.n_estimators):
            estimators.append(DecisionTreeRegressor(max_depth=self.max_depth,
                                                    max_leaf_nodes=self.max_leaf_nodes,
                                                    min_samples_split=self.min_samples_split,
                                                    min_samples_leaf=self.min_samples_leaf,
                                                    max_features=self.max_features))
        if self.shuffle:
            # Using multithreading backend since GIL is being released.
            estimators = Parallel(n_jobs=self.n_jobs, backend='threading')\
                            (delayed(_parallel_build_trees_shuffle, check_pickle=False)
                                          (i, tree, len(estimators), queries, metric, self.use_newton_method, 
                                           training_scale_values, self.bootstrap, validation, validation_scale_values,
                                           validation_performance) for i, tree in enumerate(estimators))
        else:
            # Initial ranking scores.
            training_scores = np.zeros(queries.document_count(), dtype=np.float64)

            # The pseudo-responses (lambdas) for each document.
            training_lambdas = np.empty(queries.document_count(), dtype=np.float64)

            # The optimal gradient descent step sizes for each document.
            training_weights = np.empty(queries.document_count(), dtype=np.float64)

            # Compute the pseudo-responses (lambdas) and gradient step sizes (weights) just once.
            compute_lambdas_and_weights(queries, training_scores, metric, training_lambdas, training_weights,
                                         training_scale_values, n_jobs=self.n_jobs)

            # Not using Newthon-Raphson optimization?
            if not self.use_newton_method:
                training_weights = None

            # Using multithreading backend since GIL is released in the code.
            estimators = Parallel(n_jobs=self.n_jobs, backend='threading')\
                             (delayed(_parallel_build_trees_bootstrap, check_pickle=False)
                                 (i, tree, len(estimators), queries, metric, training_scale_values,
                                  training_lambdas, training_weights, self.bootstrap, validation,
                                  validation_scale_values, validation_performance)
                                  for i, tree in enumerate(estimators))

        # Collect the trained estimators.
        self.estimators.extend(estimators)
        
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

        Parallel(n_jobs=n_jobs, backend="threading")(delayed(parallel_helper, check_pickle=False)
                (LambdaRandomForest, '_LambdaRandomForest__predict', self.estimators,
                 queries.feature_vectors[indices[i]:indices[i + 1]],
                 predictions[indices[i]:indices[i + 1]]) for i in range(indices.size - 1))

        predictions /= len(self.estimators)

        return predictions


    def predict_ranking(self, query):
        ''' 
        Predict the document ranking of the documents for the given query.

        query: Query
            The query whose documents should be ranked.
        '''
        if self.trained is False:
            raise ValueError('the model has not been trained yet')

        predictions = np.zeros(query.document_count(), dtype=np.float64)
        ranking = np.zeros(query.document_count(), dtype=np.intc)

        # Predict the ranking scores for the documents.
        self.__predict(self.estimators, query.feature_vectors, predictions)

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
