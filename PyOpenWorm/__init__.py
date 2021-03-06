# -*- coding: utf-8 -*-

"""
PyOpenWorm
==========

OpenWorm Unified Data Abstract Layer.

Classes
-------

.. autoclass:: Network
   :members:
.. autoclass:: Neuron
   :members:
.. autoclass:: Worm
   :members:
.. autoclass:: Muscle
   :members:
"""

__version__ = '0.0.1'
__author__ = 'Stephen Larson'

from .configure import Configure,Configureable,ConfigValue,BadConf
from .data import Data,DataUser,propertyTypes
from .dataObject import DataObject
from .cell import Cell
from .network import Network
from .neuron import Neuron
from .worm import Worm
from .evidence import Evidence,EvidenceError
from .muscle import Muscle
from .quantity import Quantity
from .my_neuroml import NeuroML
from .relationship import Relationship
from .connection import Connection
