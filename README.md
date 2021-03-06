![logo](docs/source/title_logo.png)

# pyfacade
[![Documentation Status](https://readthedocs.org/projects/pyfacade/badge/?version=latest)](https://pyfacade.readthedocs.io/en/latest/?badge=latest)

_**A handy toolbox for Designers and Engineers.**_

--------
## Introduction

`Pyfacade` is a Python package which aims to provide easy automatic interfaces to design and engineering software for general
purpose. Meanwhile, it encapsulates highly integrated utility functions for various professional tasks, such as *Local Buckling Analysis for
Aluminum Profile*, *Biaxial Unsymmetrical Bending Calculator for Combined Sections*, and *Lightweight FME Solver for 2D Frames*, etc.


The package is composed by below interdependent modules:

* `pyacade`  - API for _AutoCAD 2019_
* `transxml`  - XML Constructor for _MathCAD R15_
* `pymcad` - API for _MathCAD R15_
* `pyeng` - Solver for Structural Model
* `strcals` - Utilities for Quick Structural Checking


Read the full documentations on [ReadTheDocs](https://pyfacade.readthedocs.io/en/latest/)

-------
## Dependencies

The package partially relies on below third party modules:

* `pywin32` - for accessing to and interacting with Windows applications.
* `numpy` - for universal usage in matrix calculations and data process.
* `scipy` - for solving equations.
* `pandas` - for data input and output.
* `matplotlib` - for figure output.
* `lxml` - as XML parser by class 'pymcad.Xmcd'
* `geomdl` - for dealing with *Spline* curve by class 'pyacad.Acad'.


-------
## Example Codes

#### 1. Work with AutoCAD.
   
```python

import pyfacade.pyacad as pac
 
dwg = pac.Acad()  # launch AutoCAD and create a instant
rect = dwg.addrect((0,0,0), (50,100,0))  # create a rectangle on active drawing
regs = dwg.makeregion([rect])  # make region from created rectangle
sec_prop = dwg.getsecprop(regs[0])  # calculate the section properties of created rectangle

```

#### 2. Work with MathCAD.

Assume that we have an empty MathCAD XML Document *MyWorksheet.xmcd* in the folder *"C:/Temp/"*

```python

import pyfacade.pymcad as pmc
import pyfacade.transxml as tx

mw = pmc.Xmcd("C:/Temp/MyWorksheet.xmcd")  # create instance of the MathCAD file.

# Add some plain text and math expressions into the file
mw.addtext("Speed:", row=50, color="blue")
mw.addmath("v", row=50, expression="5*m*s^(-1)")
mw.addtext("Time:", row=70, color="blue")
mw.addmath("t", row=70, expression="1*min", tag="var_time")
mw.addtext("Distance:", row=95, color="green")
mw.addmath("s", row=95, expression="v*t", highlight=True, bgc="yellow", evaluate=True, unit="m")
mw.write("C:/Temp/MyWorksheet.xmcd")  # save the changes

# Now Let's see what has been created by above operations
mc_app = pmc.Mathcad()  # launch the application and create a instance
sht=mc_app.worksheet("C:/Temp/MyWorksheet.xmcd")  # open the file we just created
print(sht.getvalue("s"))   # print the final value of variable "s"

# ...And do some modifications
exp = sht.region("var_time")  # get the math region through its tag
exp.xml = tx.xml_define('t', "0.1*hr")  # re-write its expression
print(sht.getvalue("s"))   # print the final value of variable "s" again.

sht.close() # save and close the file

```


