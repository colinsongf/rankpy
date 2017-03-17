"""
Compatibility layer for Python 3/Python 2 single codebase
"""

try:
    _basestring = str
    _bytes_or_unicode = (str, str)
except NameError:
    _basestring = str
    _bytes_or_unicode = (bytes, str)