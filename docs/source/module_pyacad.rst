#####################################
AutoCAD Interface (pyfacade.pyacad)
#####################################

.. py:module:: pyacad

Module ``pyfacade.pyacad`` encapsulates a class as :ref:`AutoCAD API <class-Acad>`, including some utility methods
for quick tasks. Based on that, a subclass is implemented as :ref:`Quick Tools to Setup 2D Frame Model <class-CADFrame>`
interactively through AutoCAD drawing for ``pyfacade. pyeng``

.. _class-Acad:

1. AutoCAD API
=========================

.. autoclass:: pyfacade.pyacad.Acad
    :members:
    :exclude-members: multitry
    :member-order: bysource

    .. py:decoratormethod:: multitry(limit)

        Force function to be called again when being rejected by application.

        :param limit: int, attempting times before raising except.


.. note::

    Universal keyword *'BYLAYER'* and *'BYBLOCK'* are acceptable strings for arguments *color* and *ltype* of
    method ``setcolor``, ``setlinetype``, ``addline``, ``addcurve``, ``addrect``, ``addcircle``, ``fillhatch``,
    ``addleader``, ``addtext``, ``addmtext``, ``addtable``, ``addpolygon`` and ``freedraw``.

----

**Table of Color Name**

Below are strings of valid *Color Name* recognized by method ``setcolor``, ``setlayer``, ``addline``, ``addcurve``,
``addrect``, ``addcircle``, ``fillhatch``, ``addleader``, ``addtext``, ``addmtext``, ``addtable``, ``addpolygon``
and ``freedraw``.

===========      ======================
Color Name        Equivalent RGB Tuple
===========      ======================
'red'             (255, 0, 0)
'yellow'          (255, 255, 0)
'green'           (0, 255, 0)
'cyan'            (0, 255, 255)
'blue'            (0, 0, 255)
'magenta'         (255, 0, 255)
'gray'            (128, 128, 128)
===========      ======================

-----

.. _class-CADFrame:

2. Quick Tools to Setup 2D Frame Model
=========================================

.. autoclass:: pyfacade.pyacad.CADFrame
    :members:
    :member-order: bysource

