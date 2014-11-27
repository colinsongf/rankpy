cimport numpy as np

from cython cimport view
np.import_array()

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t   INT_t

cpdef argranksort(DOUBLE_t[:] scores, INT_t[:] ranks)

cpdef argranksort_ext(DOUBLE_t[:] scores, INT_t[:] ranks, INT_t[:] queries_offsets)

cpdef ranksort_relevance_labels(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] out)

cpdef ranksort_relevance_labels_ext(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] queries_offsets, INT_t[:] out)
