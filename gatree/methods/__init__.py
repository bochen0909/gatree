"""
GATree methods module.

This module contains the different implementations of GATree for various machine learning tasks:
- GATreeClassifier: For classification tasks
- GATreeClustering: For clustering tasks  
- GATreeRegressor: For regression tasks
"""

from .gatreeclassifier import GATreeClassifier
from .gatreeclustering import GATreeClustering
from .gatreeregressor import GATreeRegressor

__all__ = ['GATreeClassifier', 'GATreeClustering', 'GATreeRegressor']