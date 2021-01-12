##############################################
Solver for 2D-frame Question (pyfacade.pyeng)
##############################################

.. py:module:: pyeng

Module ``pyfacade.pyeng`` provides full-functional methods for 2D beam model analysis, which are encapsulated to 2 classes:

* Class ``Beam2`` - :ref:`2D beam element. <class-Beam2>`
* Class ``Bsolver`` - :ref:`FEM solver for beam questions. <class-Bsolver>`

.. _class-Beam2:
.. autoclass:: pyfacade.pyeng.Beam2
    :members:
    :member-order: bysource

.. note::
    Methods ``fill_GK``, ``fill_GR`` and ``solve`` are manual operating apis kept for Interactive Mode.
    For general purpose of a FEM study, they are automatically called by running ``Bsolver.solve()``

.. _class-Bsolver:
.. autoclass:: pyfacade.pyeng.Bsolver
    :members:
    :member-order: bysource





