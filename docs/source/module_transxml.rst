#############################################
MathCAD XML Constructor (pyfacade.transxml)
#############################################

.. py:module:: transxml

1. Module Introduction
=========================

Module ``pyfacade.transxml`` provides some utility functions for translating mathematical expressions to
*Extensible Markup Language* (XML) in specific form recognized by MathCAD.

.. autofunction:: pyfacade.transxml.xml_define

.. autofunction:: pyfacade.transxml.xml_eval

.. autofunction:: pyfacade.transxml.xml_ind

.. autofunction:: pyfacade.transxml.xml_func

.. autofunction:: pyfacade.transxml.xml_sub

.. autofunction:: pyfacade.transxml.to_matrix

.. autofunction:: pyfacade.transxml.xml_prog

.. note::

    Paragraph returned by ``to_matrix`` and ``xml_prog`` can't be directly inserted to MAthCAD XML file as elements.
    Instead, they play roles like normal string of expression, which need to be translated by calling ``xml_define``,
    ``xml_eval`` or ``xml_ind``.


Alternatively, Class ``Xexpr`` is provided as quick access working with math expression string.

.. autoclass:: pyfacade.transxml.Xexpr
    :members:
    :member-order: bysource
    :special-members: __bool__



2. Syntax of Math Expression
=============================
String can be operated and translated by this module must comply with syntax stated in this section.

2.1. Plain Expression
-------------------------
2.1.1. Basic
^^^^^^^^^^^^^^
A string of plain expression contains number, variable, function, or some of formers concatenated by recognized math symbols.
Below are some samples of valid plain expression string::

    '12'  # simply a number

    '5*X+Y'  # number and variables concatenated by math symbols

    '(sin(0.5*@p)-m^2)//2'  # expression contains function and escaped character

Here are basic rules to write plain expression:

    1. No blank in string allowed.
    #. Always use Parentheses ``'()'`` instead of Square Bracket ``'[]'`` or Curly Braces ``'{}'``.
    #. A item with Negative Sign ``'-'`` after any math symbols must be surrounded by parentheses.
    #. To input variable with subscription, add a Dot ``'.'`` between variable name and its subscription. To input Multi-subscriptions, separate them by Dots.
    #. To input index of a array, add a Underline ``'_'`` between variable name and the index. To input indices of multi-dimensional matrix , separate them by Underlines.
    #. To input greek letter, escape the corresponding roman letter with initial symbol ``'@'``. Refer to :ref:`Correspondence Table <table-greek>`.

2.1.2. Symbols and Keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below lists math symbols supported by this module.

=======   ===========================
Symbol     Meaning
=======   ===========================
\+         plus
\-         minus
\*         multiplication in default style
\*\*       multiplication shown as '×'
//         division
^          power
#          n-th Root
!          factorial
`\\\\`     square root
\|         absolute value
==         equal to
<          less than
<=         less than or equal to
>          greater than
>=         greater than or equal to
!=         not equal to
=======   ===========================

.. note:: As *unary operators*, ``'\\'`` and ``'|'`` should be followed by number or expression surrounded with parentheses to indicate its application extent.

Keywords for advanced math operators is shown below.

============================================  ===============================================================================
  Keyword                                       Equivalent Math Operator
============================================  ===============================================================================
**sigma** (x,f(x),a,b)                          :math:`\sum_{a}^{b} f(x)`
**iterp** (x,f(x),a,b)                          :math:`\prod_{a}^{b} f(x)`
**integral** (x,f(x),a,b)                       :math:`\int_{a}^{b} f(x)`
**matrix** (x,y,val_1,val_2,val_3,...)          | A matrix of x rows and y columns, and filled with val_1, val_2, val_3...
                                                | in row-order.
                                                | Positions without value provided are left to 0.
**determ** (M)                                  Determinate of matrix M, :math:`\left | M \right |`
**transpose** (M)                               Transpose of matrix M, :math:`M^{T}`
**crossp** (M1,M2)                              Cross product of matrix M1 and M2, :math:`M1\times M2`
============================================  ===============================================================================

.. _table-greek:

**The Greek and Roman Characters Correspondence Table**

======    ======     ======     ======
  Greek Letter         Roman Letter
----------------     -----------------
upper     lower      upper      lower
======    ======     ======     ======
Α          α          A          a
Β          β          B          b
Γ          γ          G          g
Δ          δ          D          d
Ε          ε          E          e
Ζ          ζ          Z          z
Η          η          H          h
Θ          θ          Q          q
Ι          ι          I          i
Κ          κ          K          k
Λ          λ           L          l
Μ          μ           M          m
Ν          ν          N          n
Ξ          ξ          X          x
Ο          ο          O          o
Π          π           P          p
Ρ          ρ          R          r
Σ          σ          S          s
Τ          τ          T          t
Υ          υ          U          u
Φ          φ          F          f
Χ          χ          C          c
Ψ          ψ          Y          y
Ω          ω          W          w
======    ======     ======     ======


2.1.3. Functions
^^^^^^^^^^^^^^^^^^
Functions, either built-in functions of MathCAD or custom-defined functions, can be included in math expression by simply
state the function name followed with parentheses, in which necessary arguments is listed and separated by comma, such like
``'tan(x+2)'``, ``'max(10,15)'``, ``'myfunc(a,b,c)'``.

.. warning::

    The validity of functions in expression will **NOT** be checked when being translated into XML for MathCAD. It is
    no problem to insert expression contains undefined function or function with improper number of arguments into MathCAD,
    as long as its syntax is correct.


2.2. Programming Block
-------------------------
Programming block can be constructed by ``transxml.xml_prog`` according to a group of statements using programming
keywords listed as below.

=======================  ===================================  ===================================
 Operation                 Keyword                              Translated Code in MathCAD
=======================  ===================================  ===================================
IF Statement             **pgif** (condition,statement)        *statement* **if** *condition*
ELSE Statement           **pgelse** (statement)                *statement* **otherwise**
FOR loop                 **pgfor** (i,a,s,b,statement)         | **for** *i* **∈** a,s..b
                                                               |     *statement*
WHILE loop               **pgwhile** (condition,statement)     | **while** *condition*
                                                               |     *statement*
Local DEFINE statement   **locd** (v,expression)                *v* **←** *expression*
BREAK statement          **pgbreak**                                **break**
CONTINUE statement       **pgcontinue**                             **continue**
RETURN action            **pgreturn** (expression)              **return** *expression*
Logical AND              **lgand** (bool1,bool2)                 *bool1* **˄** *bool2*
Logical OR               **lgor** (bool1,bool2)                  *bool1* **˅** *bool2*
Logical NOT              **lgnot** (bool)                             **¬** *bool*
Logical COMPARING        **lgcp** (sign,a,b) [*]_                   *a* **sign** *b*
=======================  ===================================  ===================================

Statements can be nested to build a complex programming block.

.. [*]
    Acceptable comparing sign are ``'=='``, ``'<'``, ``'<='``, ``'>'``, ``'>='``, ``'!='``