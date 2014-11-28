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

cimport numpy as np

from cython cimport view
np.import_array()

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t   INT_t

cpdef argranksort(DOUBLE_t[:] scores, INT_t[:] ranks)

cpdef argranksort_ext(DOUBLE_t[:] scores, INT_t[:] ranks, INT_t[:] queries_offsets)

cpdef ranksort_relevance_labels(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] out)

cpdef ranksort_relevance_labels_ext(DOUBLE_t[:] scores, INT_t[:] labels, INT_t[:] queries_offsets, INT_t[:] out)
