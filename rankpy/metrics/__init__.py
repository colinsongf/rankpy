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


'''
Information Retrieval Evaluation Metrics
'''

from .metrics import MetricFactory

from .metrics import MeanPrecision
from .metrics import WinnerTakesAll
from .metrics import ClickthroughRate
from .metrics import MeanReciprocalRank
from .metrics import MeanAveragePrecision
from .metrics import ExpectedReciprocalRank
from .metrics import DiscountedCumulativeGain


__all__ = [
    'MetricFactory',
    'MeanPrecision',
    'WinnerTakesAll',
    'ClickthroughRate',
    'MeanReciprocalRank',
    'MeanAveragePrecision',
    'ExpectedReciprocalRank',
    'DiscountedCumulativeGain'
]
