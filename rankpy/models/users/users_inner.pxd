cimport numpy as np
np.import_aray()

ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp    SIZE_t
ctypedef np.npy_uint32  UINT32_t
ctypedef np.npy_int32   INT32_t


cdef class AbstractUserModel:
    ''' 
    Defines an abstract base class for user models.
    '''
    cpdef get_clicks(self, object ranked_documents, object labels, int cutoff=?)
    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=?) nogil

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels, int cutoff=?, bint relative=?)
    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, bint relative=?) nogil

    cpdef get_expected_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil


cdef class CascadeUserModel(AbstractUserModel):
    ''' 
    Defines a simulator of a user browsing the document list top-down
    and clicking on documents based on their relevance
    '''
    cdef unsigned int rand_r_state

    cdef public np.ndarray click_proba
    cdef public np.ndarray stop_proba
    cdef public np.ndarray continue_proba
    cdef public DOUBLE_t   abandon_proba

    cdef DOUBLE_t *click_proba_ptr
    cdef DOUBLE_t *stop_proba_ptr
    cdef DOUBLE_t *continue_proba_ptr

    cpdef get_clicks(self, object ranked_documents, object labels, int cutoff=?)
    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=?) nogil

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels, int cutoff=?, bint relative=?)
    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, bint relative=?) nogil

    cpdef get_expected_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil


cdef class PositionBasedModel(AbstractUserModel):
    ''' 
    Defines a simulator of a user browsing the document list top-down
    and clicking on documents based on their relevance
    '''
    cdef unsigned int rand_r_state

    cdef public np.ndarray click_proba
    cdef public np.ndarray exam_proba

    cdef DOUBLE_t *click_proba_ptr
    cdef DOUBLE_t *exam_proba_ptr

    cdef int max_n_documents
    
    cpdef get_clicks(self, object ranked_documents, object labels, int cutoff=?)
    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=?) nogil

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels, int cutoff=?, bint relative=?)
    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, bint relative=?) nogil

    cpdef get_expected_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil


cdef class DependentClickModel(AbstractUserModel):
    ''' 
    Defines a simulator of a user browsing the document list top-down
    and clicking on documents based on their relevance
    '''
    cdef unsigned int rand_r_state

    cdef public np.ndarray click_proba
    cdef public np.ndarray stop_proba

    cdef DOUBLE_t *click_proba_ptr
    cdef DOUBLE_t *stop_proba_ptr

    cdef int max_n_documents

    cpdef get_clicks(self, object ranked_documents, object labels, int cutoff=?)
    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=?) nogil

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels, int cutoff=?, bint relative=?)
    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, bint relative=?) nogil

    cpdef get_expected_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil


cdef class ClickChainUserModel(AbstractUserModel):
    ''' 
    Defines a simulator of a user browsing the document list top-down
    and clicking on documents based on their relevance
    '''
    cdef unsigned int rand_r_state

    cdef public np.ndarray p_attraction
    cdef public DOUBLE_t   p_stop_noclick
    cdef public DOUBLE_t   p_stop_click_norel
    cdef public DOUBLE_t   p_stop_click_rel

    cdef DOUBLE_t *p_attraction_ptr

    cpdef get_clicks(self, object ranked_documents, object labels, int cutoff=?)
    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=?) nogil

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels, int cutoff=?, bint relative=?)
    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, bint relative=?) nogil

    cpdef get_expected_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil


cdef class UserBrowsingModel(AbstractUserModel):
    ''' 
    Defines a simulator of a user browsing the document list top-down
    and clicking on documents based on their relevance
    '''
    cdef unsigned int rand_r_state

    cdef public np.ndarray p_attraction
    cdef public np.ndarray p_examination

    cdef DOUBLE_t *p_attraction_ptr
    cdef DOUBLE_t *p_examination_ptr

    cdef int max_n_documents

    cpdef get_clicks(self, object ranked_documents, object labels, int cutoff=?)
    cdef int get_clicks_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, INT32_t *clicks=?) nogil

    cpdef get_clickthrough_rate(self, object ranked_documents, object labels, int cutoff=?, bint relative=?)
    cdef DOUBLE_t get_clickthrough_rate_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels, bint relative=?) nogil

    cpdef get_expected_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_expected_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil

    cpdef get_last_clicked_reciprocal_rank(self, object ranked_documents, object labels, int cutoff=?)
    cdef DOUBLE_t get_last_clicked_reciprocal_rank_c(self, INT32_t *ranked_documents, INT32_t n_documents, INT32_t *labels) nogil
