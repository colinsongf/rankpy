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
Models for simulation of user click behaviour.
'''

from .users_inner import CascadeUserModel
from .users_inner import PositionBasedModel
from .users_inner import DependentClickModel
from .users_inner import ClickChainUserModel
from .users_inner import UserBrowsingModel

__all__ = [
    'CascadeUserModel',
    'PositionBasedModel',
    'DependentClickModel',
    'ClickChainUserModel',
    'UserBrowsingModel',
]
