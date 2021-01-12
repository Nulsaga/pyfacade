#######################################################
Toolbox for Structural Calculation (pyfacade.strcals)
#######################################################

.. py:module:: strcals

Module ``pyfacade.strcals`` provides some utility functions based on other modules of this package,
for structural calculation and verification works.

.. autofunction:: pyfacade.strcals.eqslds

.. autofunction:: pyfacade.strcals.biabend

.. autofunction:: pyfacade.strcals.combsec

.. autofunction:: pyfacade.strcals.check_transoms

   Refer to explanations to :ref:`Namedtuples for Output of Transom Check  <transom-output>`

.. autofunction:: pyfacade.strcals.check_mullions

    Refer to explanations to :ref:`Namedtuples for Output of Mullion Check  <mullion-output>`

.. autofunction:: pyfacade.strcals.check_mullions_varsec

    Refer to explanations to :ref:`Namedtuples for Output of Mullion Check  <mullion-output>`

.. autofunction:: pyfacade.strcals.build_frame

.. autofunction:: pyfacade.strcals.load_frame

    Refer to explanations to :ref:`Namedtuples for Output of Loading Frame  <load-frame>`

----

.. _transom-output:

**Namedtuples for Output of Transom Check**

+ ``Transom_summary``, including 3 fields:

    + **max_stress**: a tuple in the form of (max. stress in alum. member, max. stress in steel member),
      presents max. stress on the entire model produced by most critical factored load combinations.
    + **max_deflection_wl**: a tuple in the form of (max. deflection in x-axis, max deflection in y-axis),
      presents max. deflection of the entire model produced by nominal wind load.
    + **max_deflection_dl**: a tuple in the form of (max. deflection in x-axis, max deflection in y-axis),
      presents max. deflection of the entire model produced by nominal dead load.

+ ``Transom_output``, including 4 fields:

    + **shear**: dict of max shear (shear in x-axis, shear in y-axis) due to factored load combinations on each section.
      Unit=N.
    + **stress**: dict of max stress (compressive, tensile) due to factored load combinations on each sections. Unit=MPa.
    + **deflection_wl**: dict of (max deflection in x-axis, max deflection in y-axis) due to nominal wind load
      combinations on each section. Unit=mm.
    + **deflection_dl**: dict of (max deflection in x-axis, max deflection in y-axis) due to nominal dead load
      combinations on each section. Unit=mm.

----

.. _mullion-output:

**Namedtuples for Output of Mullion Check**

+ ``Mullion_summary``, including 2 fields,

    + **max_stress**: a tuple in the form of (max. stress in alum. member, max. stress in steel member),
      presents max. stress on the entire model produced by most critical factored load combinations.
    + **max_deflection**: a tuple in the form of (max. deflection in x-axis, max deflection in y-axis),
      presents max. deflection of the entire model produced by most critical nominal load combination.

+ ``Mullion_model``, including 5 fields:

    + **N**: overall matrix of non-factored max axial force due to load cases
    + **V**: overall matrix of non-factored max shear force due to load cases
    + **Mn**: overall matrix of non-factored bending moment at support due to load cases
    + **Mp**: overall matrix of non-factored bending moment in span due to load cases
    + **D**: overall matrix of deflection due to load cases

+ ``Mullion_verify``, including 6 fields:

    + **axial**: dict of factored (max compression, max tension) on each section. Unit=N.
    + **shear**: dict of (max shear in x-axis, max shear in y-axis) due to factored load combinations on each section.
      Unit=N.
    + **moment_n**: dict of factored moments pair (about x-axis, about y-axis) causing max. compressive bending stress
      under load combinations on each section. Unit=N*mm.
    + **moment_p**: dict of factored moments pair (about x-axis, about y-axis) causing max. tensile bending stress
      under load combinations on each section. Unit=N*mm.
    + **stress**: dict of max stress (compressive, tensile) due to factored load combinations on each sections. Unit=MPa.
    + **deflection**: dict of (max deflection in x-axis, max deflection in y-axis) due to nominal load combinations on
      each section. Unit=mm.

+ ``Mullion_critical``, including 4 fields:

    + **N**: *location* occurs (max compression, max tension)
    + **V**: *location* occurs (max shear in x-axis, max shear in y-axis)
    + **S**: *location* occurs (max compressive stress, max tensile stress)
    + **D**: *location* occurs (max deflection in x-axis, max deflection in y-axis)

    Where, location is a tuple in the form of (relative_position_on_member, member_number).

----

.. _load-frame:

**Namedtuples for Output of Loading Frame**

+ ``Frame_model``, including 9 fields:

    + **nodes**: nested list, coordinates of all nodes in model. Unit = mm.
    + **beams**: nested list, beam sets of index pair of start and end nodes of each beam.
    + **A**: list, section area of each beam. Unit = mm :superscript:`2`
    + **I**: list, moment of inertia of each beam. Unit = mm :superscript:`4`
    + **E**: list, modulus of elasticity of each beam. Unit = N/mm :superscript:`2`
    + **restrain**: dict, definition of structural restrain.
    + **release**: dict, definition of release condition of beams.
    + **udl**: dict, applied uniformly distributed load on beams.
    + **pl**: dict, applied concentrated load on nodes.