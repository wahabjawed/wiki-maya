"""
This module implements a set of
:class:`~maya.Datasource`
processors that represent the input data for extracting
:class:`~maya.Feature` values.  Just like
:class:`~maya.Feature` and other
:class:'~maya.Dependent' processors,
:class:`~maya.Datasource` processors are tended to
be :func:`~maya.dependencies.solve`'d as dependencies. The
provided datasources are split conceptually into a set of modules.  Currently,
there is one module: :mod:`~maya.datasources.revision_oriented`.

Meta-datasources
++++++++++++++++
Meta-Features are classes that extend :class:`~maya.Datasource` and
implement common operations on :class:`~maya.Datasource` like
:class:`~maya.datasources.meta.filters.filter` and
:class:`~maya.datasources.meta.mappers.map`.
See :mod:`maya.datasources.meta` for the full list.

Base classes
++++++++++++
.. automodule:: maya.datasources.datasource




"""
from .datasource import Datasource

__all__ = [Datasource]
