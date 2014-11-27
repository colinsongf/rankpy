#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging

import numpy as np

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool 

from functools import partial

from .metrics import DiscountedCumulativeGain

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF

from scipy.special import expit

from utils import pickle, unpickle
from .utils_inner import argranksort


logger = logging.getLogger(__name__)


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

    n_jobs: int, optional (default is 1)
        The number of working sub-processes that will be spawned to compute
        the desired values faster. If -1, the number of CPUs will be used.

    seed: int, optional (default is None)
        The seed for random number generator that internally is used. This
        value should not be None only for debugging.
    '''
    def __init__(self, n_estimators=100, shrinkage=0.1, use_newton_method=True, max_depth=5,
                 max_leaf_nodes=None, estopping=32, max_features=None, n_jobs=1, seed=None):
        self.estimators = []
        self.n_estimators = n_estimators
        self.shrinkage = shrinkage
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.estopping = estopping
        self.use_newton_method=use_newton_method
        np.random.seed(seed)
        if n_jobs != 1:
            n_jobs = None if n_jobs < 0 else n_jobs
            self.training_pool = Pool(processes=n_jobs)
            self.evaluation_pool = ThreadPool(processes=n_jobs)
        else:
            self.training_pool = None
            self.evaluation_pool = None
        self.training_performance  = None
        self.best_performance = None
        self.trained = False


    def fit(self, metric, queries, validation=None, trace=None):
        '''
        Train the LambdaMART model on the specified queries. Optinally, use the
        specified queries for finding an optimal number of trees using validation.

        Parameters:
        -----------
        metric: metrics.AbstractMetric object
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.
            If None, `NDCG@10` will be used.

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
        # Start with random ranking scores. Originally, the randomness was needed to avoid
        # biased evaluation of the rankings because the query documents are kept in sorted
        # order by their relevance scores (see `rankpy.queries.Queries`). But since
        # `utils_inner.argranksort` uses its own randomization (documents with the equal
        # scores are shuffled randomly) this purpose lost its meaning. Now it only serves
        # as a random initialization point as in any other gradient-based type of methdods. 
        self.scores = 0.0001 * np.random.rand(queries.document_count()).astype(np.float64)

        # If the metric used for training is normalized, it is obviously advantageous
        # to precompute the scaling factor for each query in advance.
        training_metric_scale = metric.compute_scale(queries)

        # Scores used for validation data.
        if validation is not None:
            validation_scores = np.zeros(validation.document_count(), dtype=np.float64)
            validation_metric_scale = metric.compute_scale(validation)

        # The pseudo-responses (lambdas) for each document.
        self.lambdas = np.empty(queries.document_count(), dtype=np.float64)

        # The lambdas and predictions may be kept for late analysis.
        if trace is not None:
            # Create temporary directory for traced data.
            TEMP_DIRECTORY_NAME = '.lambdamart.trace.data.tmp'

            logger.info('Created temporary directory (./%s) for traced data.' % TEMP_DIRECTORY_NAME)

            if not os.path.exists(TEMP_DIRECTORY_NAME):
                os.makedirs(TEMP_DIRECTORY_NAME)

            self.trace_lambdas = trace.count('lambdas') > 0
            self.trace_gradients = trace.count('gradients') > 0

            # The pseudo-responses (lambdas) for each document: the true and estimated values.
            if self.trace_lambdas:
                # Use memory mapping to store large matrices.
                self.stage_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
                self.stage_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'lambdas.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))

            # The (loss) gradient steps for each query-document pair: the true and estimated by the regression trees.
            if self.trace_gradients:
                if self.use_newton_method is False:
                    raise ValueError('gradients are computed only if use_newton_method is True')
                # Use memory mapping to store large matrices.
                self.stage_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
                self.stage_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, queries.document_count()))
        else:
            self.trace_lambdas = False
            self.trace_gradients = False

        # The optimal gradient descent step sizes for each document.
        self.weights = np.empty(queries.document_count(), dtype=np.float64)

        # Hope we will be always maximizing :).
        best_performance = -np.inf
        best_performance_k = -1

        # How many iterations the performance has not improved on validation set.
        performance_not_improved = 0

        logger.info('Training of LambdaMART model has started.')

        # Iteratively build a sequence of regression trees.
        for k in range(self.n_estimators):
            # Computes the pseudo-responses (self.lambdas) and gradient
            # step sizes (self.weights) for the current regression tree.
            self.__compute_lambdas_and_weights(queries, metric, training_metric_scale)

            # Build the predictor for the gradients of the loss.
            estimator = DecisionTreeRegressor(max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes,
                                              max_features=self.max_features)

            # Train the regression tree.
            estimator.fit(queries.feature_vectors, self.lambdas)

            # Store the estimated lambdas for later analysis (if wanted).
            if self.trace_lambdas:
                self.stage_lambdas_truth[k, :] = self.lambdas
                self.stage_lambdas_predicted[k, :] = estimator.predict(queries.feature_vectors)

            # Estimate the ('optimal') gradient step sizes using one iteration
            # of Newton-Raphson method.
            if self.use_newton_method:
                self.__estimate_newton_gradient_steps(estimator, queries, self.lambdas, self.weights)

            # Store the true and estimated gradients for later analysis.
            if self.trace_gradients:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.stage_gradients_truth[k, :] = self.lambdas / self.weights
                    self.stage_gradients_truth[k, ~np.isfinite(self.stage_gradients_truth[k, :])] = 0.0
                self.stage_gradients_predicted[k, :] = estimator.predict(queries.feature_vectors)

            # Update the document scores using the new gradient predictor.
            if self.trace_gradients:
                self.scores += self.shrinkage * self.stage_gradients_predicted[k, :]
            else:
                self.scores += self.shrinkage * estimator.predict(queries.feature_vectors)

            # Add the new tree to the company.
            self.estimators.append(estimator)

            training_performance = metric.evaluate_queries(queries, self.scores, scale=training_metric_scale)

            if validation is None:
                logger.info('#%08d: %s (training): %11.8f' % (k + 1, metric, training_performance))

            # If validation queries have been given, estimate the model performance on them
            # and decide whether the training should not be stopped early due to no significant
            # performance improvements.
            if validation is not None:
                validation_scores += self.shrinkage * self.estimators[-1].predict(validation.feature_vectors)
                validation_performance = metric.evaluate_queries(validation, validation_scores, scale=validation_metric_scale)

                logger.info('#%08d: %s (training):   %11.8f  |  (validation):   %11.8f' % (k + 1, metric, training_performance, validation_performance))

                if validation_performance > best_performance:
                    best_performance = validation_performance
                    best_performance_k = k
                    performance_not_improved = 0
                else:
                    performance_not_improved += 1

            elif training_performance > best_performance:
                    best_performance = training_performance
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

        # Get rid of training parameters (for parallelized version, i.e. n_jobs > 1).
        if hasattr(self, 'training_parameters'):
            delattr(self, 'training_parameters')

        # Set these for further inspection.
        self.training_performance = training_performance
        self.best_performance = best_performance

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

        if n_jobs == 1:
            for tree in self.estimators:
                predictions += tree.predict(queries.feature_vectors)
            predictions *= self.shrinkage
        else:
            pool = ThreadPool(processes=None if n_jobs <= 0 else n_jobs)

            predictions = np.zeros(queries.document_count(), dtype=np.float64)

            indices = np.linspace(0, queries.document_count(), pool._processes + 1).astype(np.intc)

            for i in range(indices.size - 1):
                pool.apply_async(LambdaMART.__predict, (self.estimators, self.shrinkage,
                                                        queries.feature_vectors[indices[i]:indices[i + 1]],
                                                        predictions[indices[i]:indices[i + 1]]))
            pool.close()
            pool.join()

        return predictions


    def __estimate_newton_gradient_steps(self, estimator, queries, lambdas, weights):
        '''
        Compute one iteration of Newton-Raphson method to estimate (optimal) gradient
        steps for each terminal node of the given estimator (regression tree).

        Parameters:
        -----------
        estimator: DecisionTreeRegressor
            The regression tree for which the gradient steps are computed.

        queries:
            The queries determine which terminal nodes of the tree the associated
            pseudo-responces (lambdas) and weights fall down to.

        lambdas:
            The current 1st order derivatives of the loss function.

        weights:
            The current 2nd order derivatives of the loss function.
        '''
        # Get the number of nodes (internal + terminal) in the current regression tree.
        node_count = estimator.tree_.node_count
        indices = estimator.tree_.apply(queries.feature_vectors)

        np.copyto(estimator.tree_.value, np.bincount(indices, lambdas, node_count).reshape(-1, 1, 1))

        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(estimator.tree_.value, np.bincount(indices, weights, node_count).reshape(-1, 1, 1), out=estimator.tree_.value)

        # Remove inf's and nas's from the tree.
        estimator.tree_.value[~np.isfinite(estimator.tree_.value)] = 0.0


    def __compute_lambda_and_weight(self, qid, queries, metric, metric_scale=None):
        '''
        Compute the pseudo-responses (`self.lambdas`) and gradient steps sizes (`self.weights`)
        for each document associated with the specified query (`qid`). The pseudo-responses are 
        calculated using the given evaluation metric.

        Note that this method is used only if n_jobs was 1 in initialization, i.e. no paralellized
        method for computation of lambdas and weights is used.

        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The queries from which this LambdaMART model is being trained.

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
        argranksort(self.scores[start:end], document_ranks)

        # The relevance scores of the documents associated with the query `qid`.
        relevance_scores = queries.relevance_scores[start:end]

        # Prepare the output values for accumulation.
        self.lambdas[start:end].fill(0)
        self.weights[start:end].fill(0)

        for i in range(start, end):
            # rstart: the lowest index of a document with lower relevance score than document `i`.
            rstart = queries.query_relevance_strides[qid, relevance_scores[i - start]]

            if rstart >= end:
                break

            # Compute the (absolute) changes in the metric caused by swapping document `i` with all
            # documents `j` (`j` >= rstart) that have lower relevance to the query.
            deltas = metric.compute_delta(i - start, rstart - start, document_ranks, relevance_scores, metric_scale)

            self.scores[rstart:end] -= self.scores[i]

            # rho_i_j = 1.0 / (1.0 + exp(score[i] - score[j]))
            # Note: using name overloading to spare memory.
            weights = expit(self.scores[rstart:end])

            # Restore the scores after computing rho_i_j's.
            self.scores[rstart:end] += self.scores[i]

            # lambda_i_j for all less relevant (than `i`) documents `j`
            lambdas = weights * deltas 
            weights *= (1.0 - weights)
            weights *= deltas

            # lambda_i += sum_{j:relevance[i]>relevance[j]} lambda_i_j 
            self.lambdas[i] += lambdas.sum()
            # lambda_j -= lambda_i_j
            self.lambdas[rstart:end] -= lambdas

            # Weights are invariant in respect to swapping i and j.
            self.weights[i] += weights.sum()
            self.weights[rstart:end] += weights


    def __compute_lambdas_and_weights(self, queries, metric, metric_scale=None):
        '''
        Compute the pseudo-responses (`self.lambdas`) and gradient steps sizes (`self.weights`)
        for each document in the specified set of queries. The pseudo-responses are calculated
        using the given evaluation metric.

        Note that this method decides to run either paralellized computation of the values
        described above (`_paralell_compute_lambda_and_weight`) or single-threaded version
        (`self.__compute_lambda_and_weight`) based on value of parameter `n_jobs` given
        in initialization.

        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The queries from which this LambdaMART model is being trained.

        metric: object implementing metrics.AbstractMetric
            The evaluation metric which will be used as a utility
            function optimized by this model.

        metric_scale: array, shape=(n_queries,), optional (default is None)
            The precomputed ideal metric value for each of the specified
            queries.
        '''
        # Decide wheter to use paralellized code or not.
        if self.training_pool is None:
            for qid in range(queries.query_count()):
                self.__compute_lambda_and_weight(qid, queries, metric, None if metric_scale is None else metric_scale[qid])
        else:
            # TODO: Would be much nicer to use multithreaded backend
            #       instead of child processes here.
            pool = self.training_pool

            # Prepare parameters for the following jobs (only once per training!)
            if not hasattr(self, 'training_parameters'):
                self.training_parameters = []

                # Each process is processing a continuous chunk of queries.
                if queries.query_count() >= pool._processes:
                    self.training_query_sequence = np.linspace(0, queries.query_count(), pool._processes + 1).astype(np.intc)
                else:
                    self.training_query_sequence = np.arange(queries.query_count() + 1, dtype=np.intc)

                # Prepare the appropriate views into the queries data. Every parameter is
                # just a view into needed portion of the query arrays memory, so far...
                for i in range(len(self.training_query_sequence) - 1):
                    qstart = self.training_query_sequence[i]
                    qend = self.training_query_sequence[i + 1]
                    dstart = queries.query_indptr[qstart]
                    dend = queries.query_indptr[qend]
                    query_indptr = queries.query_indptr[qstart:qend + 1]
                    scores = self.scores[dstart:dend]
                    relevance_scores = queries.relevance_scores[dstart:dend]
                    relevance_strides = queries.query_relevance_strides[qstart:qend, :]
                    q_metric_scale = None if metric_scale is None else metric_scale[qstart:qend]
                    self.training_parameters.append((query_indptr, scores, relevance_scores, relevance_strides, metric, q_metric_scale))

            # Execute the computation of lambdas and weights in paralell. This ruins the whole thing (parameters
            # will be copied into child processes) but its just a couple of MBs anyway...
            for pid, (lambdas, weights) in enumerate(pool.imap(_paralell_compute_lambda_and_weight, self.training_parameters, chunksize=1)):
                start = queries.query_indptr[self.training_query_sequence[pid]]
                end   = queries.query_indptr[self.training_query_sequence[pid + 1]]
                np.copyto(self.lambdas[start:end], lambdas)
                np.copyto(self.weights[start:end], weights)


    @classmethod
    def load(cls, filepath):
        '''
        Load the previously saved LambdaMART model from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a LambdaMART object will be loaded.
        '''
        return unpickle(filepath)


    def save(self, filepath):
        '''
        Save te LambdaMART model into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        # TODO: Deal with saving the memory-mapped arrays intelligently!
        self.trace_lambdas = None
        self.trace_gradients = None
        pickle(self, filepath)


    def save_as_text(self, filepath):
        '''
        Save the model into the file in an XML format.
        '''        
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
        return 'LambdaMART(trees=%d, max_depth=%s, max_leaf_nodes=%s, shrinkage=%.2f, max_features=%s, use_newton_method=%s, trained=%s)' % \
               (self.n_estimators, str(self.max_depth) if self.max_leaf_nodes is None else '?', '?' if self.max_leaf_nodes is None else str(self.max_leaf_nodes),
                self.shrinkage, 'all' if self.max_features is None else str(self.max_features), self.use_newton_method, self.trained)


if __name__ == '__main__':
    pass
