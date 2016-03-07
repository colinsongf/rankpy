# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport calloc, free
from libc.math cimport exp, log

from numpy import int32 as INT32
from numpy import float64 as DOUBLE

cdef DOUBLE_t INFINITY = np.inf

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF # alias 2**31 - 1


cdef class AbstractUserModel:
    ''' 
    Defines an abstract base class for user models.
    '''
    cpdef get_clicks(self, object ranked_documents, object labels,
                     int cutoff=2**31-1):
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t n_documents = min(len(ranked_documents), cutoff)
        cdef np.ndarray[INT32_t, ndim=1] clicks_ = np.zeros(n_documents, dtype=INT32)
        cdef np.ndarray ranked_documents_, labels_

        if getattr(ranked_documents, "dtype", None) != INT32 or not ranked_documents.flags.contiguous:
            ranked_documents_ = np.ascontiguousarray(ranked_documents, dtype=INT32)
        else:
            ranked_documents_ = ranked_documents

        if getattr(labels, "dtype", None) != INT32 or not labels.flags.contiguous:
            labels_ = np.ascontiguousarray(labels, dtype=INT32)
        else:
            labels_ = labels

        with nogil:
            self.get_clicks_c(<INT32_t*>ranked_documents_.data, n_documents,
                              <INT32_t*>labels_.data, <INT32_t*>clicks_.data)

        return clicks_

    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents,
                          INT32_t *labels, INT32_t *clicks=NULL) nogil:
        ''' 
        Guts of get_clicks! Need to be reimplemented in the extended class.
        '''
        pass

    cpdef get_expected_reciprocal_rank(self, object ranked_documents,
                                       object labels, int cutoff=2**31-1):
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t n_documents = min(len(ranked_documents), cutoff)
        cdef np.ndarray ranked_documents_, labels_
        cdef DOUBLE_t result

        if getattr(ranked_documents, "dtype", None) != INT32 or not ranked_documents.flags.contiguous:
            ranked_documents_ = np.ascontiguousarray(ranked_documents, dtype=INT32)
        else:
            ranked_documents_ = ranked_documents

        if getattr(labels, "dtype", None) != INT32 or not labels.flags.contiguous:
            labels_ = np.ascontiguousarray(labels, dtype=INT32)
        else:
            labels_ = labels

        with nogil:
            result = self.get_expected_reciprocal_rank_c(<INT32_t*>ranked_documents_.data, n_documents, <INT32_t*>labels_.data)

        return result

    cdef DOUBLE_t get_expected_reciprocal_rank_c(self,
                                                 INT32_t *ranked_documents,
                                                 INT32_t n_documents,
                                                 INT32_t *labels) nogil:
        ''' 
        Guts of self.get_expected_reciprocal_rank_c! Need to be reimplemented
        in the extended class.
        '''
        pass

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels,
                                int cutoff=2**31-1, bint relative=False):
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t n_documents = min(len(ranked_documents), cutoff)
        cdef np.ndarray ranked_documents_, labels_
        cdef DOUBLE_t result

        if getattr(ranked_documents, "dtype", None) != INT32 or not ranked_documents.flags.contiguous:
            ranked_documents_ = np.ascontiguousarray(ranked_documents, dtype=INT32)
        else:
            ranked_documents_ = ranked_documents

        if getattr(labels, "dtype", None) != INT32 or not labels.flags.contiguous:
            labels_ = np.ascontiguousarray(labels, dtype=INT32)
        else:
            labels_ = labels

        with nogil:
            result = self.get_clickthrough_rate_c(<INT32_t*>ranked_documents_.data, n_documents, <INT32_t*>labels_.data, relative)

        return result

    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents,
                                          INT32_t n_documents, INT32_t *labels,
                                          bint relative=False) nogil:
        ''' 
        Guts of self.get_clickthrough_rate! Need to be reimplemented
        in the extended class.
        '''
        pass

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents,
                                           object labels, int cutoff=2**31-1):
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t n_documents = min(len(ranked_documents), cutoff)
        cdef np.ndarray ranked_documents_, labels_
        cdef DOUBLE_t result

        if getattr(ranked_documents, "dtype", None) != INT32 or not ranked_documents.flags.contiguous:
            ranked_documents_ = np.ascontiguousarray(ranked_documents, dtype=INT32)
        else:
            ranked_documents_ = ranked_documents

        if getattr(labels, "dtype", None) != INT32 or not labels.flags.contiguous:
            labels_ = np.ascontiguousarray(labels, dtype=INT32)
        else:
            labels_ = labels

        with nogil:
            result = self.get_last_clicked_reciprocal_rank_c(<INT32_t*>ranked_documents_.data, n_documents, <INT32_t*>labels_.data)

        return result

    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self,
                                                     INT32_t *ranked_documents,
                                                     INT32_t n_documents,
                                                     INT32_t *labels) nogil:
        ''' 
        Guts of self.get_last_clicked_reciprocal_rank! Need to be reimplemented
        in the extended class.
        '''
        pass


cdef class CascadeUserModel(AbstractUserModel):
    def __init__(self, click_proba, stop_proba, abandon_proba=0.0, seed=None):
        ''' 
        Initialize the cascade user model.
        '''
        if len(click_proba) != len(stop_proba):
            raise ValueError('the probability arrays does not have the same '
                             'length: (%d) != (%d)' % (len(click_proba),
                                                       len(stop_proba)))

        if seed == 0:
            raise ValueError('the seed cannot be 0 for technical reasons, '
                             'please, choose different seed, e.g.: 42')

        self.rand_r_state = np.random.randint(1, RAND_R_MAX) if seed is None else seed
        self.click_proba = np.array(click_proba, copy=True, dtype=DOUBLE, order='C')
        self.click_proba_ptr = <DOUBLE_t*> self.click_proba.data
        self.stop_proba = np.array(stop_proba, copy=True, dtype=DOUBLE, order='C')
        self.stop_proba_ptr = <DOUBLE_t*> self.stop_proba.data
        self.abandon_proba = abandon_proba
        self.continue_proba = (1. - self.click_proba * self.stop_proba - 
                               (1 - self.click_proba) * self.abandon_proba)
        self.continue_proba_ptr =  <DOUBLE_t*> self.continue_proba.data

        if (self.click_proba < 0.0).any() or (self.click_proba > 1.0).any():
            raise ValueError('click_proba is not a valid probability vector')

        if (self.stop_proba < 0.0).any() or (self.stop_proba > 1.0).any():
            raise ValueError('click_proba is not a valid probability vector')

        if (abandon_proba < 0.0) or (abandon_proba > 1.0):
            raise ValueError('abandon_proba is not a probability')

    def __reduce__(self):
        return (CascadeUserModel, (self.click_proba, self.stop_proba,
                                   self.abandon_proba, self.rand_r_state))

    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents,
                          INT32_t *labels, INT32_t *clicks=NULL) nogil:
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t i, label, count = 0

        for i in range(n_documents):
            label = labels[ranked_documents[i]]

            if random(&self.rand_r_state) < self.click_proba_ptr[label]:
                if clicks != NULL:
                    clicks[i] = 1
                count += 1
                if random(&self.rand_r_state) < self.stop_proba_ptr[label]:
                    break
            elif random(&self.rand_r_state) < self.abandon_proba:
                break

        # Return the number of clicks.
        return count

    cdef DOUBLE_t get_clickthrough_rate_c(self, 
                                          INT32_t *ranked_documents,
                                          INT32_t n_documents,
                                          INT32_t *labels,
                                          bint relative=False) nogil:
        ''' 
        Guts of self.get_clickthrough_rate! Need to be reimplemented
        in the extended class.
        '''
        cdef int rank, label, n_clicks
        # Probability of getting down to rank i, receiving j clicks, and continuing.
        cdef double *click_count_proba
        # Probability of getting down to rank i, receiving j clicks, and stopping.
        cdef double *click_stop_count_proba
        # Auxiliary varibles to handle the probabilities for the last document in the ranking.
        cdef double stop_proba, abandon_proba
        # The final answer will be here.
        cdef double result

        if relative:
            click_count_proba = <double *> calloc((n_documents + 1) * (n_documents + 2) / 2, sizeof(double))
            click_stop_count_proba = <double *> calloc((n_documents + 1) * (n_documents + 2) / 2, sizeof(double))

            # Compute the probabilities of clicking and continuing.
            # ------------------------------------------------------

            # 1st document must be processed separately.
            label = labels[ranked_documents[0]]

            __cc(click_count_proba, 0, 0)[0] = (1.0 - self.click_proba_ptr[label]) * (1.0 - self.abandon_proba)
            __cc(click_count_proba, 0, 1)[0] = self.click_proba_ptr[label] * (1.0 - self.stop_proba_ptr[label])

            # Notice that ``click_count_proba`` is not needed for the last document (see the next for-loop).
            for rank in range(1, n_documents - 1):
                label = labels[ranked_documents[rank]]

                for n_clicks in range(0, rank + 1):
                    __cc(click_count_proba, rank, n_clicks)[0] += __cc(click_count_proba, rank - 1, n_clicks)[0] * (1.0 - self.click_proba_ptr[label]) * (1.0 - self.abandon_proba)
                    __cc(click_count_proba, rank, n_clicks + 1)[0] += __cc(click_count_proba, rank - 1, n_clicks)[0] * self.click_proba_ptr[label] * (1.0 - self.stop_proba_ptr[label])

            # Compute the probabilities of clicking and stopping.
            # ---------------------------------------------------

            # 1st document must be processed separately.
            label = labels[ranked_documents[0]]

            if n_documents > 1:
                __cc(click_stop_count_proba, 0, 0)[0] = (1.0 - self.click_proba_ptr[label]) * self.abandon_proba
                __cc(click_stop_count_proba, 0, 1)[0] = self.click_proba_ptr[label] * self.stop_proba_ptr[label]
            else:
                __cc(click_stop_count_proba, 0, 0)[0] = 1.0 - self.click_proba_ptr[label]
                __cc(click_stop_count_proba, 0, 1)[0] = self.click_proba_ptr[label]

            for rank in range(1, n_documents):
                label = labels[ranked_documents[rank]]

                # Probability of stopping and/or abandoning at the current document.
                stop_proba, abandon_proba = (<DOUBLE_t> 1.0, <DOUBLE_t> 1.0) if (rank == n_documents - 1) else (self.stop_proba_ptr[label], self.abandon_proba)
                
                for n_clicks in range(0, rank + 1):
                    __cc(click_stop_count_proba, rank, n_clicks)[0] += __cc(click_count_proba, rank - 1, n_clicks)[0] * (1.0 - self.click_proba_ptr[label]) * abandon_proba
                    __cc(click_stop_count_proba, rank, n_clicks + 1)[0] += __cc(click_count_proba, rank - 1, n_clicks)[0] * self.click_proba_ptr[label] * stop_proba

            # Compute the 'relative' CTR, i.e. ratio of clicked documents.
            # ------------------------------------------------------------

            result = 0.0

            # The probability of clicking n_click times while stopping after last click.
            for rank in range(n_documents):
                for n_clicks in range(1, rank + 2):
                    result += n_clicks * __cc(click_stop_count_proba, rank, n_clicks)[0]

            free(click_count_proba)
            free(click_stop_count_proba)

            return result
        else:
            result = 1.0
            for rank in range(n_documents - 1, -1, -1):
                result = (1.0 - self.click_proba_ptr[labels[ranked_documents[rank]]]) * (self.abandon_proba + (1.0 - self.abandon_proba) * result)
            return 1.0 - result
        
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self,
                                                 INT32_t *ranked_documents,
                                                 INT32_t n_documents,
                                                 INT32_t *labels) nogil:
        ''' 
        Guts of self.get_expected_reciprocal_rank_c! Need to be reimplemented
        in the extended class.
        '''
        cdef int rank, label
        cdef double result = 0.0, gamma = 1.0

        for rank in range(n_documents):
            label = labels[ranked_documents[rank]]
            result += gamma * self.click_proba_ptr[label] / (rank + 1)
            gamma *= self.continue_proba_ptr[label]

        return result

    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self,
                                                     INT32_t *ranked_documents,
                                                     INT32_t n_documents,
                                                     INT32_t *labels) nogil:
        ''' 
        Guts of self.get_last_clicked_reciprocal_rank! Need to be reimplemented
        in the extended class.
        '''
        cdef int rank, label
        cdef double result, gamma = 1.0
        cdef double *click_proba_at_rank = <double *> calloc(n_documents, sizeof(double))
        cdef double *not_click_proba_from_rank = <double *> calloc(n_documents + 1, sizeof(double))

        for rank in range(n_documents):
            label = labels[ranked_documents[rank]]
            click_proba_at_rank[rank] = self.click_proba_ptr[label] * gamma
            gamma *= self.continue_proba_ptr[label]

        not_click_proba_from_rank[n_documents] = 1.0
        for rank in range(n_documents - 1, -1, -1):
            label = labels[ranked_documents[rank]]
            not_click_proba_from_rank[rank] += (1.0 - self.click_proba_ptr[label]) * (self.abandon_proba + (1.0 - self.abandon_proba) * not_click_proba_from_rank[rank + 1])

        result = 0.0
        for rank in range(n_documents):
            label = labels[ranked_documents[rank]]
            click_proba_at_rank[rank] = click_proba_at_rank[rank] * (self.stop_proba_ptr[label] + (1.0 - self.stop_proba_ptr[label]) * not_click_proba_from_rank[rank + 1])
            result += click_proba_at_rank[rank] / (rank + 1)

        free(click_proba_at_rank)
        free(not_click_proba_from_rank)

        return result

    property seed:
        def __get__(self):
            return self.rand_r_state

        def __set__(self, v):
            if v == 0:
                raise ValueError('the seed cannot be 0 for technical reasons, '
                                 'please, choose different seed, e.g.: 42')
            self.rand_r_state = v


cdef class PositionBasedModel(AbstractUserModel):
    def __init__(self, click_proba, exam_proba, seed=None):
        ''' 
        Initialize the position-base click model.
        '''
        if seed == 0:
            raise ValueError('the seed cannot be 0 for technical reasons, '
                             'please, choose different seed, e.g.: 42')

        self.rand_r_state = np.random.randint(1, RAND_R_MAX) if seed is None else seed
        self.click_proba = np.array(click_proba, copy=True, dtype=DOUBLE, order='C')
        self.click_proba_ptr = <DOUBLE_t*> self.click_proba.data
        self.exam_proba = np.array(exam_proba, copy=True, dtype=DOUBLE, order='C')
        self.exam_proba_ptr = <DOUBLE_t*> self.exam_proba.data
        self.max_n_documents = self.exam_proba.shape[0]

        if (self.click_proba < 0.0).any() or (self.click_proba > 1.0).any():
            raise ValueError('click_proba is not a valid probability vector')

        if (self.exam_proba < 0.0).any() or (self.exam_proba > 1.0).any():
            raise ValueError('click_proba is not a valid probability vector')

    def __reduce__(self):
        return (PositionBasedModel,
                (self.click_proba, self.exam_proba, self.rand_r_state))

    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents,
                          INT32_t *labels, INT32_t *clicks=NULL) nogil:
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t i, label, count = 0

        for i in range(min(self.max_n_documents, n_documents)):
            label = labels[ranked_documents[i]]

            if random(&self.rand_r_state) < (self.click_proba_ptr[label] *
                                             self.exam_proba_ptr[i]):
                if clicks != NULL:
                    clicks[i] = 1
                count += 1

        # Return the number of clicks.
        return count

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels,
                                int cutoff=2**31-1, bint relative=False):
        return AbstractUserModel.get_clickthrough_rate(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff),
                                            relative=relative)

    cdef DOUBLE_t get_clickthrough_rate_c(self,
                                          INT32_t *ranked_documents,
                                          INT32_t n_documents,
                                          INT32_t *labels,
                                          bint relative=False) nogil:
        ''' 
        Guts of self.get_clickthrough_rate! Need to be reimplemented
        in the extended class.
        '''
        cdef int rank
        # The final answer will be here.
        cdef double result

        result = 1.0
        for rank in range(n_documents):
            result *= (1.0 - 
                       self.click_proba_ptr[labels[ranked_documents[rank]]] *
                       self.exam_proba_ptr[rank])
        return 1.0 - result

    cpdef get_expected_reciprocal_rank(self, object ranked_documents,
                                       object labels, int cutoff=2**31-1):
        return AbstractUserModel.get_expected_reciprocal_rank(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff))

    cdef DOUBLE_t get_expected_reciprocal_rank_c(self,
                                                 INT32_t *ranked_documents,
                                                 INT32_t n_documents,
                                                 INT32_t *labels) nogil:
        ''' 
        Guts of self.get_expected_reciprocal_rank_c! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents,
                                           object labels, int cutoff=2**31-1):
        return AbstractUserModel.get_last_clicked_reciprocal_rank(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff))

    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self,
                                                     INT32_t *ranked_documents,
                                                     INT32_t n_documents,
                                                     INT32_t *labels) nogil:
        ''' 
        Guts of self.get_last_clicked_reciprocal_rank! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    property seed:
        def __get__(self):
            return self.rand_r_state

        def __set__(self, v):
            if v == 0:
                raise ValueError('the seed cannot be 0 for technical reasons, '
                                 'please, choose different seed, e.g.: 42')
            self.rand_r_state = v


cdef class DependentClickModel(AbstractUserModel):
    def __init__(self, click_proba, stop_proba, seed=None):
        ''' 
        Initialize the cascade user model.
        '''
        if seed == 0:
            raise ValueError('the seed cannot be 0 for technical reasons, '
                             'please, choose different seed, e.g.: 42')

        self.rand_r_state = np.random.randint(1, RAND_R_MAX) if seed is None else seed
        self.click_proba = np.array(click_proba, copy=True, dtype=DOUBLE, order='C')
        self.click_proba_ptr = <DOUBLE_t*> self.click_proba.data
        self.stop_proba = np.array(stop_proba, copy=True, dtype=DOUBLE, order='C')
        self.stop_proba_ptr = <DOUBLE_t*> self.stop_proba.data
        self.max_n_documents = self.stop_proba.shape[0]

        if (self.click_proba < 0.0).any() or (self.click_proba > 1.0).any():
            raise ValueError('click_proba is not a valid probability vector')

        if (self.stop_proba < 0.0).any() or (self.stop_proba > 1.0).any():
            raise ValueError('click_proba is not a valid probability vector')

    def __reduce__(self):
        return (DependentClickModel,
                (self.click_proba, self.stop_proba, self.rand_r_state))

    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents,
                          INT32_t *labels, INT32_t *clicks=NULL) nogil:
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t i, label, count = 0

        for i in range(min(self.max_n_documents, n_documents)):
            label = labels[ranked_documents[i]]

            if random(&self.rand_r_state) < self.click_proba_ptr[label]:
                if clicks != NULL:
                    clicks[i] = 1
                count += 1
                if random(&self.rand_r_state) < self.stop_proba_ptr[i]:
                    break

        # Return the number of clicks.
        return count

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels,
                                int cutoff=2**31-1, bint relative=False):
        return AbstractUserModel.get_clickthrough_rate(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff),
                                            relative=relative)

    cdef DOUBLE_t get_clickthrough_rate_c(self,
                                          INT32_t *ranked_documents,
                                          INT32_t n_documents,
                                          INT32_t *labels,
                                          bint relative=False) nogil:
        ''' 
        Guts of self.get_clickthrough_rate! Need to be reimplemented
        in the extended class.
        '''
        cdef INT32_t rank
        # The final answer will be here.
        cdef DOUBLE_t result

        result = 1.0
        for rank in range(n_documents):
            result *= (1.0 - self.click_proba_ptr[labels[ranked_documents[rank]]])
        return 1.0 - result

    cpdef get_expected_reciprocal_rank(self, object ranked_documents,
                                       object labels, int cutoff=2**31-1):
        return AbstractUserModel.get_expected_reciprocal_rank(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff))

    cdef DOUBLE_t get_expected_reciprocal_rank_c(self,
                                                 INT32_t *ranked_documents,
                                                 INT32_t n_documents,
                                                 INT32_t *labels) nogil:
        ''' 
        Guts of self.get_expected_reciprocal_rank_c! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents,
                                           object labels, int cutoff=2**31-1):
        return AbstractUserModel.get_last_clicked_reciprocal_rank(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff))

    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self,
                                                     INT32_t *ranked_documents,
                                                     INT32_t n_documents,
                                                     INT32_t *labels) nogil:
        ''' 
        Guts of self.get_last_clicked_reciprocal_rank! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    property seed:
        def __get__(self):
            return self.rand_r_state

        def __set__(self, v):
            if v == 0:
                raise ValueError('the seed cannot be 0 for technical reasons, '
                                 'please, choose different seed, e.g.: 42')
            self.rand_r_state = v


cdef class ClickChainUserModel(AbstractUserModel):
    def __init__(self, p_attraction, p_continue_noclick, p_continue_click_norel,
                 p_continue_click_rel, seed=None):
        ''' 
        Initialize the cascade user model.
        '''
        if seed == 0:
            raise ValueError('the seed cannot be 0 for technical reasons, '
                             'please, choose different seed, e.g.: 42')

        self.rand_r_state = np.random.randint(1, RAND_R_MAX) if seed is None else seed

        self.p_attraction = np.array(p_attraction, copy=True, dtype=DOUBLE, order='C')
        self.p_attraction_ptr = <DOUBLE_t*> self.p_attraction.data

        self.p_stop_noclick = 1.0 - p_continue_noclick
        self.p_stop_click_norel = 1.0 - p_continue_click_norel
        self.p_stop_click_rel = 1.0 - p_continue_click_rel

    def __reduce__(self):
        return (ClickChainUserModel,
                (self.p_attraction, 1.0 - self.p_stop_noclick,
                 1.0 - self.p_stop_click_norel, 1.0 - self.p_stop_click_rel,
                 self.rand_r_state))

    cdef int get_clicks_c(self, INT32_t *ranked_documents,
                          INT32_t n_documents, INT32_t *labels,
                          INT32_t *clicks=NULL) nogil:
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t i, label, count = 0

        for i in range(n_documents):
            label = labels[ranked_documents[i]]

            if random(&self.rand_r_state) < self.p_attraction_ptr[label]:
                if clicks != NULL:
                    clicks[i] = 1
                count += 1
                if random(&self.rand_r_state) < self.p_attraction_ptr[label]:
                    if random(&self.rand_r_state) < self.p_stop_click_rel:
                        break
                else:
                    if random(&self.rand_r_state) < self.p_stop_click_norel:
                        break
            elif random(&self.rand_r_state) < self.p_stop_noclick:
                break

        # Return the number of clicks.
        return count

    cdef DOUBLE_t get_clickthrough_rate_c(self,
                                          INT32_t *ranked_documents,
                                          INT32_t n_documents,
                                          INT32_t *labels,
                                          bint relative=False) nogil:
        ''' 
        Guts of self.get_clickthrough_rate! Need to be reimplemented
        in the extended class.
        '''
        cdef INT32_t rank
        # The final answer will be here.
        cdef DOUBLE_t result

        result = 1.0
        for rank in range(n_documents - 1, -1, -1):
            result = (1.0 - self.p_attraction_ptr[labels[ranked_documents[rank]]]) * (self.p_stop_noclick + (1.0 - self.p_stop_noclick) * result)
        return 1.0 - result

    cdef DOUBLE_t get_expected_reciprocal_rank_c(self,
                                                 INT32_t *ranked_documents,
                                                 INT32_t n_documents,
                                                 INT32_t *labels) nogil:
        ''' 
        Guts of self.get_expected_reciprocal_rank_c! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self,
                                                     INT32_t *ranked_documents,
                                                     INT32_t n_documents,
                                                     INT32_t *labels) nogil:
        ''' 
        Guts of self.get_last_clicked_reciprocal_rank! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    property seed:
        def __get__(self):
            return self.rand_r_state

        def __set__(self, v):
            if v == 0:
                raise ValueError('the seed cannot be 0 for technical reasons, '
                                 'please, choose different seed, e.g.: 42')
            self.rand_r_state = v

cdef class UserBrowsingModel(AbstractUserModel):
    def __init__(self, p_attraction, p_examination, seed=None):
        ''' 
        Initialize the cascade user model.
        '''
        if seed == 0:
            raise ValueError('the seed cannot be 0 for technical reasons, please,'
                             ' choose different seed, e.g.: 42')

        self.rand_r_state = np.random.randint(1, RAND_R_MAX) if seed is None else seed

        self.p_attraction = np.array(p_attraction, copy=True, dtype=DOUBLE, order='C')
        self.p_attraction_ptr = <DOUBLE_t*> self.p_attraction.data

        self.p_examination = np.array(p_examination, copy=True, dtype=DOUBLE, order='C')

        if self.p_examination.shape[0] != self.p_examination.shape[1]:
            raise ValueError('the p_examination must be a square matrix (%d != %d)' % (self.p_examination.shape[0], self.p_examination.shape[1]))

        self.p_examination_ptr = <DOUBLE_t*> self.p_examination.data

        self.max_n_documents = self.p_examination.shape[0]

    def __reduce__(self):
        return (UserBrowsingModel,
                (self.p_attraction, self.p_examination, self.rand_r_state))


    cpdef get_clicks(self, object ranked_documents, object labels,
                     int cutoff=2**31-1):
        return AbstractUserModel.get_clicks(self, ranked_documents, labels, min(self.max_n_documents, cutoff))


    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=NULL) nogil:
        ''' 
        Simulate clicks on the specified ranked list of documents.
        '''
        cdef INT32_t rank, label, count = 0
        cdef INT32_t curr_click_rank_offset = 0
        cdef INT32_t prev_click_rank = self.max_n_documents - 1

        for rank in range(n_documents):
            label = labels[ranked_documents[rank]]

            if random(&self.rand_r_state) < (self.p_attraction_ptr[label] * self.p_examination_ptr[curr_click_rank_offset + prev_click_rank]):
                if clicks != NULL:
                    clicks[rank] = 1
                count += 1
                prev_click_rank = rank

            curr_click_rank_offset += self.max_n_documents

        # Return the number of clicks.
        return count

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels,
                                int cutoff=2**31-1, bint relative=False):
        return AbstractUserModel.get_clickthrough_rate(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff),
                                            relative=relative)

    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents,
                                          INT32_t n_documents, INT32_t *labels,
                                          bint relative=False) nogil:
        ''' 
        Guts of self.get_clickthrough_rate! Need to be reimplemented
        in the extended class.
        '''
        cdef int rank, label
        # The final answer will be here.
        cdef DOUBLE_t result = 1.0
        cdef INT32_t no_prev_click_offset = self.max_n_documents - 1

        for rank in range(n_documents):
            result *= (1.0 - self.p_attraction_ptr[labels[ranked_documents[rank]]] * self.p_examination_ptr[no_prev_click_offset])
            no_prev_click_offset += self.max_n_documents

        return 1.0 - result

    cpdef get_expected_reciprocal_rank(self, object ranked_documents,
                                       object labels, int cutoff=2**31-1):
        return AbstractUserModel.get_expected_reciprocal_rank(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff))

    cdef DOUBLE_t get_expected_reciprocal_rank_c(self,
                                                 INT32_t *ranked_documents,
                                                 INT32_t n_documents,
                                                 INT32_t *labels) nogil:
        ''' 
        Guts of self.get_expected_reciprocal_rank_c! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents,
                                           object labels, int cutoff=2**31-1):
        return AbstractUserModel.get_last_clicked_reciprocal_rank(
                                            self, ranked_documents, labels,
                                            min(self.max_n_documents, cutoff))

    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self,
                                                     INT32_t *ranked_documents,
                                                     INT32_t n_documents,
                                                     INT32_t *labels) nogil:
        ''' 
        Guts of self.get_last_clicked_reciprocal_rank! Need to be reimplemented
        in the extended class.
        '''
        # TODO: Implement this!!!
        return -1.0

    property seed:
        def __get__(self):
            return self.rand_r_state

        def __set__(self, v):
            if v == 0:
                raise ValueError('the seed cannot be 0 for technical reasons, '
                                 'please, choose different seed, e.g.: 42')
            self.rand_r_state = v

        
# =============================================================================
# Utils
# =============================================================================


cdef inline unsigned int rand_r(unsigned int *seed) nogil:
    seed[0] ^= <unsigned int> (seed[0] << 13)
    seed[0] ^= <unsigned int> (seed[0] >> 17)
    seed[0] ^= <unsigned int> (seed[0] << 5)

    return (seed[0] & <unsigned int> RAND_R_MAX)


cdef inline double random(unsigned int *random_state) nogil:
    ''' 
    Generate a random double in [0, 1].
    '''
    return (<double> rand_r(random_state) / <double> RAND_R_MAX)


cdef inline double * __cc(double *a, int i, int j) nogil:
    return &a[i * (i + 3) / 2 + j + 1]


