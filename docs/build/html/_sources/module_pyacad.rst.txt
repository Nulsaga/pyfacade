#####################################
AutoCAD Interface (pyfacade.pyacad)
#####################################

.. py:module:: pyacad

Module ``pyfacade.pyacad`` encapsulates a class as AutoCAD API, including some utility methods
for quick tasks.

.. autoclass:: pyfacade.pyacad.Acad
    :members:
    :member-order: bysource

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

----

Also a subclass is implemented as quick tools for ``pyfacade. pyeng`` to acquire information from AutoCAD drawing
interactively

.. autoclass:: pyfacade.pyacad.CADFrame
    :members:
    :member-order: bysource


