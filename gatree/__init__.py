__version__ = '0.2.0'

# Import main classes
from .methods import GATreeClassifier, GATreeRegressor, GATreeActionSelector, GATreeContinuousActionSelector

# Import utilities
from .utils import format_decision_path, print_decision_path

__all__ = [
    'GATreeClassifier', 
    'GATreeRegressor', 
    'GATreeActionSelector', 
    'GATreeContinuousActionSelector',
    'format_decision_path',
    'print_decision_path'
]
