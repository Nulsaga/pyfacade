#####################################
MathCAD Interface (pyfacade.pymcad)
#####################################

.. py:module:: pymcad

Module ``pyfacade.pymcad`` provides two Classes working with MathCAD on implement level in both statical and
interactive way.

* Class ``Xmcd`` - :ref:`.xmcd file parser. <class-Xmcd>`
* Class ``Mathcad`` - :ref:`MathCAD API. <class-Mathcad>`

.. _class-Xmcd:
.. autoclass:: pyfacade.pymcad.Xmcd
    :members:
    :member-order: bysource

.. _class-Mathcad:
.. autoclass:: pyfacade.pymcad.Mathcad
    :members:
    :member-order: bysource

.. autoclass:: pyfacade.pymcad.Worksheet
    :members:
    :member-order: bysource

.. autoclass:: pyfacade.pymcad.Region
    :members:
    :member-order: bysource

----

**Table of Color Name**

Below are strings of *Color Name* recognized by method ``addtext``, ``addmath``, ``addcompare``, ``addsolve`` of
class ``pymcad.Xmcd``, and the corresponding *Hex Code* .

===========      ===========
Color Name        Hex Code
===========      ===========
'red'             #ff0000
'maroon'          #800000
'pink'            #ff8080
'violet'          #ff80ff
'magenta'         #ff00ff
'orange'          #ff8000
'yellow'          #ffff80
'lime'            #80ff80
'green'           #00ff00
'aqua'            #80ffff
'blue'            #0000ff
'gray'            #c0c0c0
===========      ===========
