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

.. py:class:: Acad((file_name=None, visible=True, hanging=0.5)

    AutoCAD Automation API.

    :param file_name: str, file name (and path) to open with application. Activate the specified file
                        if it is opened already. Otherwise, try to open the file from specified path.
    :param visible: bool, show AutoCAD Application UI after launching.
    :param hanging: float, time in seconds to wait when each time application is found busy.

    .. py:method:: visible
        :property:

        Visibility of application

    .. py:method:: drawing
        :property:

        Name of operating drawing

    .. py:staticmethod:: findacad()

        Show *Windows Handle* of AutoCAD application.

        :return: list of int, handle of application windows.

    .. py:method:: drawinglist(fullname=False)

        Get a name list of drawings opened by application.

        :param fullname: bool, get the full path of the drawings.
        :return: list of str.

    .. py:method:: new(template="acadiso.dwt")

        Create a new drawing and set it as operating target.

        :param template: str, template name used to create new drawing.
        :return: None

    .. py:method:: current()

        Set the active drawing as operating target.

        :return: str. name of the active drawing.

    .. py:method:: prompt(message)

        Show a message in *Command Line* of AutoCAD UI.

        :param message: str, content of message.
        :return: None

    .. py:method:: marco(macro_name)

        Run pre-defined Macro of the drawing.

        :param macro_name: str. name of the Macro.
        :return: None

    .. py:method:: update()

        Refresh the drawing.

        :return: None

    .. py:method:: isbusy()

        Check if application is busy now

        :return: bool.

    .. py:method:: showwindow()

        Bring the application window to the most front.

        :return: None.

    .. py:method:: showdwg(hanging=0.5)

        Show the operating drawing on screen.

        :param hanging: float, time in seconds of each waiting when application is busy.
        :return: None

    .. py:staticmethod:: point(x, y, z=0)

        Define a point coordinates array.

        :param x: float, X-coordinate of the point.
        :param y: float, Y-coordinate of the point.
        :param z: float, Z-coordinate of the point.
        :return: ``win32com.client.VARIANT`` for VBA COM usage.

    .. py:class:: UCS()

        A user defined orthogonal coordinate system.

        :param origin: 3-element array-like, global coordinates of origin point.
        :param point_x: 3-element array-like, global coordinates of a point lying on local x-direction.
                        By default, it follows WCS x-direction.
        :param point_ref: 3-element array-like, global coordinates of a point lying on positive y side of local
                          xy plane. By default, it use WCS y-direction.

        .. py:method:: o
            :property:

            Coordinates of origin point. Read-only.

        .. py:method:: x
            :property:

            Unit vector of x-direction. Read-only.

        .. py:method:: y
            :property:

            Unit vector of y-direction. Read-only.

        .. py:method:: z
            :property:

            Unit vector of z-direction. Read-only.

        .. py:method:: m3
            :property:

            [3x3] matrix of direction transformation from UCS to WCS. Read-only.

        .. py:method:: m4
            :property:

            [4x4] matrix of homogenous coordinate transformation from UCS to WCS. Read-only.

        .. py:method:: toucs(point)

            Translate a WCS coordinates to UCS.

            :param point: 3-element array-like, WCS coordinates of a point in the form of (X,Y,Z).
            :return: numpy.ndarray, corresponding coordinate on UCS in the form of (x,y,z).

        .. py:method:: fromucs(point)

            Translate a UCS coordinates to WCS.

            :param point: 3-element array-like, UCS coordinates of a point in the form of (x,y,z).
            :return: numpy.ndarray, corresponding coordinate on WCS in the form of (X,Y,Z).

    .. py:method:: get(intype="point", ref_pnt=None, prompt="Specify a point from drawing: ", bits=0, keyword="", default="")

        Get information from drawing.

        :param intype: str, type of requested information.

            * **'point'** - WCS coordinate of a point. Select from drawing, or input from keyboard.

            * **'angle'** - angel in radians between specified direction and *Base Angle* according to System Variable *ANGBASE*.

                        Specify two points to indicate the direction by selecting from drawing or
                        keyboard input of their coordinates. Alternatively, a number in unit of *degree* can be input
                        directly by keyboard.

            * **'distance'** - distance between two specified points.

                        Specify two points by selecting from drawing or keyboard input of their coordinates.
                        Alternatively, a number as distance can be input directly by keyboard.

            * **'orientation'** - angel in radians between specified direction and *World X-direction*.

            * **'corner'** -  WCS coordinate of a point as corner of a Rectangle formed by itself and *ref_pnt*.

                        Specify by selecting from drawing or keyboard input of its coordinate.

        :param ref_pnt: list of float, coordinate of reference point in the form of [x,y,z].
                        If information required is *'angle'*, *'distance'* or *'orientation'*, this reference point is
                        taken as the first point specified.
        :param prompt: str, prompt message.
        :param bits: int, input filter.

                    | 0-No filtering
                    | 1-Disallows NULL input, i.e. [return] or [space].
                    | 2-Disallows input of zero.
                    | 4-Disallows negative values.
        :param keyword: str. keywords to be recognized when receiving input from keyboard, separate each keyword by
                        blank.
        :param default: value to be returned when received keyword is NULL. Invalid when *bits* = 1.
        :return: float or tuple of float according to required information type, or str if keyword is inputted.

    .. py:method:: kbinput(intype="real", prompt="", bits=0, keyword="", space=False, default="")

        Get keyboard input interactively.

        :param intype: str, requested input type. 'integer', 'real', 'string' or 'keyword'.
        :param prompt: str, prompt message.
        :param bits: int, input filter. Invalid when *intype* = 'string'.

                | 0-No filtering
                | 1-Disallows NULL input, i.e. [return] or [space]
                | 2-Disallows input of zero
                | 4-Disallows negative values
        :param keyword: str, keywords to be recognized when receiving input from keyboard, separate each keyword by
                        blank. Invalid when *intype* = 'string'.
        :param space: bool, spaces are allowed when requested input is string.
        :param default: value to be returned when received keyword is NULL. Invalid when *bits* = 1.
        :return: int, float or str according to requested input type.

    .. py:method:: pick(objtype=None, prompt="Select a entity from drawing: ", keyword="")

        Get a *AutoCAD Entity* from drawing by mouse-click selecting.

        :param objtype: str, type of requested entity either in formal name (e.g. 'AcDbLine') or short name
                        (e.g. 'line'). Any type of entity will be acceptable if not specified.
        :param prompt: str, prompt message.
        :param keyword: str,  keywords to be recognized when receiving input from keyboard, separate each keyword by
                        blank.
        :return: selected entity object or keyword.

    .. py:method:: select(objtype=None, prompt="Select entities from drawing: ")

        Get a selection set by window selecting on drawing.

        :param objtype: str, type of requested entity either in formal name (e.g. 'AcDbLine') or short name
                        (e.g. 'line'). Any type of entity will be acceptable if not specified.
        :param prompt: str, prompt message.
        :return: a list of selected entities.

    .. py:method:: byid(obj_id)

        Get a *AutoCAD Entity* by its object ID.

        :param obj_id: int, AUtoCAD object ID.
        :return: entity object.

    .. py:method:: readtable(table_obj, title=True, header=True, index=True)

        Read data from a table in drawing

        :param table_obj: AcDbTable object
        :param title: bool, ignore the first row of table
        :param header: bool, read the second row of table as header of each column
        :param index: bool, read the first column of table as index of each row
        :return: pandas.DataFrame

    .. py:method:: command(comds)

        Send a commands list to AutoCAD.

        :param comds: list of str, commends to be executed.
        :return: None

    .. py:method:: setcolor(color, dwg_obj, *dwg_objs)

        Set color of entity objects.

        :param color: str as color name, or a tuple of 3 int as RGB value.
        :param dwg_obj: entity object to be set color.
        :param dwg_objs: other entity objects to be set color.
        :return: None

    .. py:method:: setlinetype(ltype, dwg_obj, *dwg_objs, scale=None, lib="acadiso.lin")

        Set line type of entity objects.

        :param ltype: str, name of line type.
        :param dwg_obj: entity object to be set line type.
        :param dwg_objs: other entity objects to be set line type.
        :param scale: float, line type scale. Remain unchanged if not specified.
        :param lib: str, library that specified line type is loaded from if it has not been loaded yet.
        :return: None

    .. py:method:: setlayer(self, layer_name, color=None, ltype=None, lweight=None, plottable=None, hidden=None, freeze=None, lock=None, activate=False)

        Change settings of a layer

        :param layer_name: str, name of operating layer, if layer name is not existing, a new layer will be created
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. Remain unchanged if not specified.
        :param lweight: int, weight of lines when plotted. Remain unchanged if not specified.
                        below is correspondence of valid values:

                        =============  ===================================================
                        by layer        -1
                        by block        -2
                        default         -3
                        line weight     | 0,5,9,13,15,18,20,25,30,35,40,50,53,60,70,80,90
                                        | 100,106,120,140,158,200,211
                        =============  ===================================================

        :param plottable: bool, print objects belongs to this layer when plotting. Remain unchanged if not specified.
        :param hidden: bool, hide objects belongs to this layer. Remain unchanged if not specified.
        :param freeze: bool, freeze objects belongs to this layer. Remain unchanged if not specified.

               .. note:: Active layer can **NOT** be frozen.

        :param lock: bool, lock objects belongs to this layer. Remain unchanged if not specified.
        :param activate: bool, activate operated layer.
        :return: operating layer object

    .. py:method:: addline(point_1, point_2, *other_points, close=False, polyline=False, line_width=0, color=None, ltype=None, scale=None, layer=None)

        Draw straight lines on drawing.

        :param point_1: list of float [x,y,z], coordinate of first point.
        :param point_2: list of float [x,y,z], coordinate of second point.
        :param other_points: list of float, coordinate of other points, [x,y,z]...
        :param close: bool, make a line between last point and first point when more than 2 points are provided.
        :param polyline: bool, make a continuous polyline instead of separated lines.
        :param line_width: float, global width of polyline. Only valid when *ployline* = True.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw lines on. If not given, draw lines on current layer.
        :return: created polyline object, or a list of line objects.

    .. py:method:: addcurve(*fitpoints, start_tan=[0, 0, 0], end_tan=[0, 0, 0], color=None, ltype=None, scale=None, layer=None)

        Draw Nurbs curve as spline passing through specified fit points.

        :param fitpoints: nested list of float [[x1,y1,z1],[x2,2,z2]...], represents coordinate of fit points.
        :param start_tan: list of float, 3D vector of tangency at start point
        :param end_tan: list of float, 3D vector of tangency at end point
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw curve on. If not given, draw curve on current layer.
        :return: created spline object.

    .. py:method:: addrect(corner_1, corner_2, line_width=0, color=None, ltype=None, scale=None, layer=None)

        Shortcut of drawing a rectangle by defining its 2 opposite corners.

        :param corner_1: list of float [x,y,z], coordinate of one corner point.
        :param corner_2: list of float [x,y,z], coordinate of the opposite corner point.
        :param line_width: float, global width of polyline.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw rectangle on. If not given, draw rectangle on current layer.
        :return: created of polyline object.

    .. py:method:: addcircle(center, radius, color=None, ltype=None, scale=None, layer=None)

        Draw a circle by defining center point and radius.

        :param center: list of float [x,y,z], coordinate of center point.
        :param radius: float, radius of circle.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw rectangle on. If not given, draw rectangle on current layer.
        :return: created circle object.

    .. py:method:: addnode(point, *other_points, layer=None)

        Mark nodes at specified locations.

        :param point: list of float [x,y,z], coordinate to mark node at.
        :param other_points: 3-element lists [x,y,z], other coordinates of to mark nodes at.
        :param layer: str, name of layer to mark node on. If not given, mark node on current layer.
        :return: created node object, or a list of node objects.

    .. py:method:: fillhatch(outerloops, innerloops=[], pattern='ANSI31', angle=0.0, scale=1.0, asso=False, color=None, layer=None)

        Fill the entities by specified pattern.

        :param outerloops: nested list of objects, outer boundary of hatch filling, in the form of
                            [[object group 1], [object group 2], [object group 3]...], objects in
                            each group should form a simple closed boundary.
        :param innerloops: nested list of objects, inner boundary of hatch filling, in the form of
                            [[object group 1], [object group 2], [object group 3]...], objects in
                            each group should form a simple closed boundary.
        :param pattern: str, name of hatch pattern.
        :param angle: float, angle of hatch in radians.
        :param scale: float, scale of hatch.
        :param asso: bool, associate hatch with boundaries.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param layer: str, name of layer to create hatch on. If not given, create hatch on current layer.
        :return: created hatch object.

    .. py:method:: adddim(point_1, point_2, offset, measure_dir=None, dimstyle=None, layer=None)

        Add a aligned or rotated dimension annotation on x-y plane measuring the distance between 2 points.

        :param point_1: list of float [x,y,z], coordinates of first point.
        :param point_2: list of float [x,y,z], coordinates of second point.
        :param offset: float, offset distance of dimension annotation from measured line.
        :param measure_dir: list of float [vx,vy,vz], direction vector which measurement align with. If not given, align
                            with the line connecting two measured points.
        :param dimstyle: str, name of dimension style. Use current active style if not specified.
        :param layer: str, name of layer to create annotation on. If not given, create annotation on current layer.
        :return: created dimension object.

    .. py:method:: addleader(point_1, point_2, *other_points, style=None, ltype=None, scale=None, color=None, arrow=None, headsize=None, layer=None, spline=False)

        Add a straight or curved leader line on drawing.

        :param point_1: list of float [x,y,z], coordinate of base point which the leader points at.
        :param point_2: list of float [x,y,z], coordinate of second point leader line passes.
        :param other_points: list of float [x,y,z], coordinate of other point leader line passes.
        :param style: str, name of dimension style. Use current active style if not specified.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use color defined by
                      dimension style
        :param arrow: int, index of arrow type, from 0 to 19. If not given, use arrow type defined by dimension style.

                        ======    ===============================
                        Index       Arrow Type
                        ======    ===============================
                        0           Closed filled
                        1           Closed blank
                        2           Closed
                        3           Dot
                        4           Architectural tick
                        5           Oblique
                        6           Open
                        7           Origin indicator
                        8           Origin indicator 2
                        9           Right angle
                        10          Open 30
                        11          Dot small
                        12          Dot blank
                        13          Dot small blank
                        14          Box
                        15          Box filled
                        16          Datum triangle
                        17          Datum triangle filled
                        18          Integral
                        19          No arrow
                        ======    ===============================

        :param headsize: float, size of arrow. If not given, use arrow size defined by dimension style.
        :param layer: str, name of layer to create annotation on. If not given, create leader line on current layer.
        :param spline: bool, draw leader line as curved spline.
        :return: created leader object.

    .. py:method:: insertblock(insert_pnt, block_name, scale=(1.0, 1.0, 1.0), rotation=0, dynamic_prop=None, attr=None, layer=None)

        Insert a block to current drawing.

        :param insert_pnt: list of float [x,y,z], coordinates of insert point.
        :param block_name: str, path and file name of inserted block including '.dwg' extension.
        :param scale: list or tuple of float (sx,sy,sz), scale of inserting block in x, y and z direction.
                      for Dynamic Block, only uniform scaling is allowed.
        :param rotation: float, rotation angle about insert point in radians.
        :param dynamic_prop: dict, customized properties for dynamic block. Keys of dict can be int as numeric index or
                            str as property name.
        :param attr: dict, attributes of block. Keys of dict can be int as numeric index or str as tag name.
        :param layer: str, name of layer to insert block to. If not given, insert block to current layer.
        :return: inserted block object.

    .. py:method:: makeregion(objects=None, layer=None, del_source=True)

        Create region from selected or provided objects.

        :param objects: list of objects. If not given, interactive selecting on screen will be requested.
        :param del_source: bool, delete the source objects after region being created.
        :param layer: str, name of layer to create generated region on. If not given, create region on current layer.
        :return: list of created regions.

    .. py:method:: textstyle(style_name, font_file, bigfont_file='', bold=False, italic=False, regen=True, activate=False)

        Define a text style.

        :param style_name: str, name of text style, existing text style with same name will be overwritten.
        :param font_file: str, path and name of font file.
        :param bigfont_file: str, path and  name of big font file.
        :param bold: bool, bold font style.
        :param italic: bool, italic font style.
        :param regen: bool, regenerate the drawing. Modification on existing text style will only be shown after
                      regeneration
        :param activate: bool, set the new defined text style as the active one.
        :return: None

    .. py:method:: addtext(content, insert_point, height, style=None, align_type="Left", rotation=0, color=None, layer=None)

        Add single-line text on drawing.

        :param content: str, content of text.
        :param insert_point: list of float [x,y,z], coordinate of reference point for text alignment.
        :param height: float, height of text.
        :param style: str, text style. Use current active style if not specified.
        :param align_type: str, type of alignment. One of below:

                *'Left', 'Center', 'Right', 'Middle', 'TopLeft', 'TopCenter', 'TopRight', 'MiddleLeft', 'MiddleCenter',
                'MiddleRight', 'BottomLeft', 'BottomCenter', 'BottomRight'*

        :param rotation: float, rotation angle about reference point in radians.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param layer: str, name of layer to add text to. If not given, add text to current layer.
        :return: created text object

    .. py:method:: addmtext(content, insert_point, height, width, style=None, align_type="TopLeft", rotation=0, color=None, layer=None)

        Add multi-line text zone on drawing.

        :param content: str, content of text.
        :param insert_point:  list of float [x,y,z], coordinate of reference point for text alignment.
        :param height: float, height of text.
        :param width: float, width of text zone.
        :param style: str, text style, Use current active style if not specified.
        :param align_type: str, type of alignment. One of below:

                *'TopLeft', 'TopCenter', 'TopRight', 'MiddleLeft', 'MiddleCenter', 'MiddleRight', 'BottomLeft',
                'BottomCenter', 'BottomRight'*

        :param rotation: float, rotation angle about reference point in radians.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param layer: str, name of layer to add multi-line text to. If not given, add multi-line text to current layer.
        :return: added mtext object

    .. py:method:: addtable(table_data, insert_point, row_height, col_width, layer=None, title=None, show_index=False, index=[], \
                            show_header=False, headers=[], acc=2, title_textheight=None, header_textheight=None, \
                            main_textheight=None, title_style=None, header_style=None, main_style=None, \
                            title_align=None, header_align=None, main_align=None, cell_style=None, cell_color=None, merge_empty=0)

        Insert a Table on drawing according to input data.

        :param table_data: array-like, Series or DataFrame, data to be shown in table.
        :param insert_point: list of float [x,y,z], coordinate of reference point for text alignment.
        :param row_height: float or list of float, height of rows. If a list is provided, height is specified
                           respectively from top to bottom.
        :param col_width: float or list of float, width of columns. If a list is provided, width is specified
                          respectively from left to right.
        :param layer: str, name of layer to insert table to. If not given, insert to current layer.
        :param title: str, title of table.
        :param show_index: bool, show the index for each row of data.
        :param index: list of str, name of indices. If not given, indices will be read from input data as first option,
                      when failed, the default indices such as *'R1', 'R2', 'R3'* will be applied.
        :param show_header: bool, show the header for each column of data.
        :param headers: list of str, name of headers. If not given, headers will be read from input data as first
                        option, when failed, the default header such as *'C1', 'C2', 'C3'* will be applied.
        :param acc: int. number of decimal place to be shown in table if input value is a float.
        :param title_textheight: float, text height of title. USe default height if not specified.
        :param header_textheight: float, text height of header. USe default height if not specified.
        :param main_textheight: float, text height of main data part. USe default height if not specified.
        :param title_style: str. style name of title text. Use current active style if not specified.
        :param header_style:  str. style name of header text. Use current active style if not specified.
        :param main_style: str. style name of main data text. Use current active style if not specified.
        :param title_align: str. alignment of title text. One of below:

                *'TopLeft', 'TopCenter', 'TopRight', 'MiddleLeft', 'MiddleCenter', 'MiddleRight', 'BottomLeft',
                'BottomCenter', 'BottomRight'*

                Use default alignment if not specified.

        :param header_align: str. alignment of header text. Use default alignment if not specified. Valid value is same
                             as title_align.
        :param main_align: str. alignment of main data text. Use default alignment if not specified. Valid value is same
                             as title_align.
        :param cell_style: dict. text style of individual cell, in the form of {(row, col):'TextStyle', ...}
        :param cell_color: dict. text color of individual cell, in the form of {(row, col):'ColorName', ...} or
                            {(row, col):(r,g,b),...}
        :param merge_empty: int. option for merging empty cells to adjacent cells.

                            | 0- do not merge
                            | 1- merge to left cell
                            | 2 - merge to upper cell

        :return: created table object

    .. py:method:: save(full_file_name=None, version=2013)

        Save the drawing file.

        :param full_file_name: str. path and file name to save the drawing as, including extension *'.dwg'* or *'.dxf'* .
                               If not given, save the drawing in-place.
        :param version: int. version of AutoCAD file. Supported version are: *2000, 2004, 2010, 2013, 2018*
        :return: None.

    .. py:method:: close()

        Discard changes and close the drawing.

        Operating drawing will be shifted to next active drawing.

        :return: bool, return True if all the drawings have been closed.


    .. py:method:: getvector(unit=True)

        Create a vector by specifying two point on drawing.

        :param unit: bool, return unit vector.
        :return: numpy.ndarray represents a 3-dimensional vector.

    .. py:staticmethod:: vangle(vector, rad=True)

        Get projected orientation of a vector on *X-Y Plane*, in range from 0 to pi.

        :param vector: array-like [vx, vy ,vz], the vector
        :param rad: return the orientation angle in radians. When False, return the angle in degree
        :return: float, the solved angle.

    .. py:staticmethod:: tolerate(var_1, var_2, dim=3, acc=1e-6)

        Check whether two points or vectors are equivalent within tolerance.

        :param var_1: array-like of float, the coordinate of first point, or the first vector.
        :param var_2: array-like of float, the coordinate of second point, or the second vector.
        :param dim: int. number of dimensions to compare.
        :param acc: float. allowable tolerance.
        :return: bool.

    .. py:staticmethod:: rotatevec(vector, angle)

        Rotate a vector by specified angle.

        :param vector: array-like of float [vx, vy, vz], the vector to be rotated.
        :param angle: float, rotating angle in radians. take counter-clockwise as positive.
        :return: array-like of float, the rotated vector.

    .. py:method:: addpolygon(insert_point, n, radius, start_angle=None, line_width=0, color=None, ltype=None, scale=None, layer=None)

        Shortcut to drawing a polygon.

        :param insert_point: list of float [x,y,z], coordinates of center of polygon.
        :param n: int. number of sides.
        :param radius: float, length of sides.
        :param start_angle: float, orientation angle of the line connecting center and the first corner.
        :param line_width: float, global width of polyline.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw polygon on. If not given, draw polygon on current layer.
        :return: created polyline object.

    .. py:method:: freedraw(polyline=False, line_width=0, color=None, ltype=None, scale=None, layer=None)

        Shortcut to continuously draw lines or a polyline in interactive way.

        :param polyline: bool, draw a polyline instead of individual lines.
        :param line_width: float, global width of polyline. Only valid when *polyline* = True.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw on. If not given, draw on current layer.
        :return: created polyline object, or list of created lines.

    .. note::

        Universal keyword *'BYLAYER'* and *'BYBLOCK'* are acceptable strings for arguments *color* and *ltype* of
        method ``setcolor``, ``setlinetype``, ``addline``, ``addcurve``, ``addrect``, ``addcircle``, ``fillhatch``,
        ``addleader``, ``addtext``, ``addmtext``, ``addtable``, ``addpolygon`` and ``freedraw``.


    .. py:method:: locate(self, block_name=None, ucs=None)

        Get the coordinate of insert point of selected blocks.

        :param block_name: str, name of the block to be located. If not given, a sample block will be requested to be
                            selected from drawing.
        :param ucs: ``Acad.UCS`` object, return the coordinate of insert points in this UCS. If not given, return the
                     coordinates in WCS.
        :return: list of tuple.

    .. py:method:: readsecprop(sort=1, sec_name=None, file_name=None)

        Read section properties data from selected tables in drawing.

        :param sort: int, method of sorting selected tables.

                        | 0: no sorting.
                        | 1: vertical priority, from upper to lower.
                        | 2: horizontal priority, from left to right.

        :param sec_name: list of str, section name used as header of each column of data. If not given, Use default name
                        such as 'Section_01', 'Section_02' etc.
        :param file_name: str, file name to for data export, with extension of '.csv' or '.json'. If not
                        specified, exporting procedure will be skipped.
        :return: pandas.DataFrame

        .. warning::

            This experimental method only works on table in specific form.


    .. py:method:: getplasec(region_obj, axis_angle=0, acc=1e-6)

        Calculate plastic section modulus about a specified axis of a region.

        :param region_obj: Region object.
        :param axis_angle: float, orientation angle of specified axis in radians.
        :param acc: float, allowable tolerance when finding the center line of the section.
        :return: float, calculated plastic section modulus.


    .. py:method:: getsecprop(region_obj, file_name=None)

        Analyze section properties of a region.

        :param region_obj: Region object.
        :param file_name: str, file name to for data export, with extension of '.json'. If not specified, exporting
                          procedure will be skipped.
        :return: dict. keys correspondence is shown below:

                ========  ==================================================
                'A'         area of cross section
                'Ix'        moment of inertia about x-axis
                'Iy'        moment of inertia about y-axis
                'Ixy'       product moment of inertia
                'Zx'        elastic section modulus about x-axis
                'Zy'        elastic section modulus about y-axis
                'Sx'        plastic section modulus about x-axis
                'Sy'        plastic section modulus about y-axis
                'I1'        moment of inertia about major axis
                'I2'        moment of inertia about minor axis
                'alpha'     angle between major axis and x-axis, in radians
                'Z1'        elastic section modulus about major axis
                'Z2'        elastic section modulus about minor axis
                'S1'        plastic section modulus about major axis
                'S2'        plastic section modulus about minor axis
                ========  ==================================================

    .. py:method:: getboundary(region_obj, spl_sub=100, file_name=None)

        Get vector from centroid of a section to its boundary corner and arcs

        :param region_obj: Region object.
        :param spl_sub: int, numbers of subdivided segments of a spline boundary.
        :param file_name: str, file name for data exporting, with extension of '.json'. If not specified, exporting
                          procedure will be skipped.
        :return: tuple in the form of:

                | ([vector_to_corner_1, vector_to_corner_2, ... ],
                | [(vector_to_center_arc1, (start_angle_arc1, end_angle_arc1), radius_arc1 ),
                | (vector_to_center_arc2, (start_angle_arc2, end_angle_arc2), radius_arc2 ),
                | ... ]
                | )

                each vector is a tuple of float (vx, vy).

    .. py:classmethod:: boundalong(boundary_nodes, boundary_arc, direction_vector)

        Measure the distance from centroid of section to its boundary in specified direction.

        :param boundary_nodes: list of vector from centroid to boundary corner.
                               each vector is a tuple of float, (vx, vy).
        :param boundary_arc: list of tuple states geometrical information of boundary arc, in the form of:

                            | [(vector_to_center_arc1, (start_angle_arc1, end_angle_arc1), radius_arc1 ),
                            | (vector_to_center_arc2, (start_angle_arc2, end_angle_arc2), radius_arc2 ), ... ]

        :param direction_vector: array-like, vector indicating the direction of measurement.
        :return: tuple of float, (max. negative distance, max. positive distance). Here negative distance is measured
                along the opposite direction.

    .. py:method:: dimregion(region_obj, direction_vector=(1, 0, 0), spl_sub=10, dim_offset=-10, outside=True, dimstyle=None)

        Add dimension annotation marking overall size of a region along specified direction.

        :param region_obj: Region object.
        :param direction_vector: array-like, vector indicating the direction of measurement.
        :param spl_sub: int, numbers of subdivided segments of a spline boundary.
        :param dim_offset: float, offset distance of dimension annotation from reference point. reference point is the
                            most left point on measured object When *'outside'* = False, and is the most outside point
                            on boundary when *'outside'* = True.
        :param outside: bool, always put the dimension annotation outside of measured object.
        :param dimstyle: str, name of dimension style. If not given, current active style will be used.
        :return: created dimension object.

    .. py:method:: seclib(file_name, sort=1, sec_name=None, spl_sub=100, update=True)

        Select a group of regions and output their object ID, section properties and boundary information to .csv or .json file

        :param file_name: str, file name of output with extension of '.csv' or '.json'.
        :param sort: int, method of sorting selected regions.

                        | 0: no sorting.
                        | 1: vertical priority, from upper to lower.
                        | 2: horizontal priority, from left to right.

        :param sec_name: list of str, section name used as header of each column of data. If not given, Use default name
                        such as 'Section_01', 'Section_02' etc.
        :param spl_sub: int, numbers of subdivided segments when obtain boundary node of a spline.
        :param update: bool, partially update the output file if it is existing.
                       If True, data column with new section name will be inserted to the existing file, while data
                       column with duplicated section name will be renewed. Otherwise, the whole existing file will be
                       overwritten.
        :return: None

    .. py:method:: lbelem(spl_sub=10, dim_offset=-10, dimstyle=None, num_block="NUM.dwg", num_scale=1.0, file_name=None, update=True)

        Mark out elements on selected regions and get elements geometry information.

        This is a quick tool for primary task producing structural data used by ``pyfacade.pymcad.Xmcd.addblock_lbcheck``

        :param spl_sub: int, numbers of subdivided segments when obtain boundary node of a spline.
        :param dim_offset: float, offset distance of dimension annotation from measured element.
        :param dimstyle: str, name of applied dimension style.
        :param num_block: str or None, insert a numbering mark next to element. If None is given, no numbering mark
                            will be inserted.
        :param num_scale: float, scale of inserted numbering mark.
        :param file_name: str, path and name of .json file for data output. If not given, the exporting procedure will
                            be skipped.
        :param update: bool, partially update the output file if it is existing. If True, only the element data on
                       section with duplicated object ID will be renewed. Otherwise, the whole file will be overwritten.
        :return: dict, recorded element information of regions, in the form of:

                        | {'object id 1': {'1': elem_data,'2': elem_data, ... },
                        |  'object id 2': {'1': elem_data,'2': elem_data, ...},
                        |  ... }

                Where *elem_data* represents a dict with keys and corresponding values listed as below:

                ===========  ===================================================================================
                'length'      float, length of element.
                'thks'        list of float, element thickness at intersection.
                'rp'          list of float, coordinates of reference point.
                'slope'       float, orientation of element.
                'bx'          | list of float, boundary of element in x-direction,
                              | relative to section centroid.
                'by'          | list of float, boundary of element in y-direction,
                              | relative to section centroid.
                'Ie'          | float, moment of inertia of element about axis along its
                              | longitudinal side.
                'type'        | int. indicator for type element:
                              | 'G' - element under stress gradient
                              | 'U' - element under Uniform compressive stress
                              | 'R' - **reinforced** element under Uniform compressive stress
                              | 'X' - stress gradient along X-aix
                              | 'Y' - stress gradient along Y-aix
                ===========  ===================================================================================

    .. py:method:: search(criteria=None)

        Search objects from drawing according to specified criteria.

        :param criteria: one-argument function which receives drawing entities and return a boolean.
        :return: list of found objects.

    .. py:method:: replaceblock(block_name, replac_by, offset=(0., 0., 0.), new_scale=None, new_rotation=None, \
            new_dynamic_prop=None, new_attr=None, offset_in_scale=True, inherit=True)

        Replace blocks with specified name by another block.

        :param block_name: str, name of block to be replaced.
        :param replac_by: str, path and name of new block with extension '.dwg' use to replace.
        :param offset: array-like of float (dx, dy, dz), offset distance in 3 directions from the new block to replaced
                        one.
        :param new_scale: float, scale of new block. If not given, keep same scale as replaced block.
        :param new_rotation: float, rotation angle of new block about insert point in radians.  If not given, keep
                            same rotation angle as replaced block.
        :param new_dynamic_prop: dict, customized properties for new dynamic block. Keys of dict can be int as numeric
                               index or str as property name. If not given and *inherit* = True, the new block will try
                               to use same properties as replaced one.
        :param new_attr: dict, attributes of new block. Keys of dict can be int as numeric index or str as tag name.
                           If not given and *inherit* = True, the new block will try to use same attributes as
                           replaced one.
        :param offset_in_scale: bool, Also scale the *offset* extent according to the scale of new block.
        :param inherit: bool, inherit dynamic block properties or block attributes from replaced block if corresponding
                            value for new block is not specified.
        :return: None

    .. py:decoratormethod:: multitry(limit)

        Force function to be called again when being rejected by application.

        :param limit: int, attempting times before raising except.

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

.. py:class:: CADFrame(file_name=None, geoacc=4)

    Subclass of ``pyacad.Acad``, extends for interactively acquiring structural information of 2D frame.

    Acquired data is used to build up analysis model by Module ``pyfacade.pyeng``.

    :param file_name: str, file name (and path) to open with application. Activate the specified file
            if it is opened already. Otherwise, try to open the file from specified path. When successfully opened,
            prompts will be shown in the command line of AutoCAD, asking for selecting and specifying the basic frame.
    :param geoacc: int. number of decimal place to be kept when outputting the nodes' coordinates.

    .. py:method:: nodes
        :property:

        List of nodes' coordinates. Read-only.

    .. py:method:: beams
        :property:

        List of beam set, indicating index pair of start and end nodes of each beam. Read-only.

    .. py:method:: node_num(rng=None, color=None)

        Show number of nodes on drawing.

        :param rng: list on int, indices of nodes require showing number. If not give, show numbers of all nodes.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :return: list of added text objects.

    .. py:method:: beam_num(rng=None, color=None)

        Show number of beams on drawing.

        :param rng: list on int, indices of beams require showing number. If not give, show numbers of all beams.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :return: list of added text objects.

    .. py:method:: set_restrain()

        Specify restrain conditions on node from drawing.

        :return: dict. Definition of structural restrain in the form of
                {node_no:[res_condition_x, res_condition_y, res_condition_rotate], ...}, where:
                0=released and 1=restrained.

    .. py:method:: set_release()

        Specify end release of beams from drawing.

        :return: dict. Definition of release conditions of beams in the form of
                {beam_no:[(axial_condition_start, rotation_condition_start), (axial_condition_end, rotation_condition_end)],...},
                where: 0=fixed and 1=released.

    .. py:method:: set_udl()

        Specify uniform distributed load on beams from drawing.

        :return: dict. Definition of applied *Uniformly Distributed Load* on model in the form of
                {beam_no:(axial_force, transverse_force),...}. Unit=N/mm.

    .. py:method:: set_pointload()

        Specify point load on nodes from drawing.

        :return: dict. Definition of applied *Concentrated Load* on model in the form of
                {node:[Fx, Fy, Mz]...}. Unit=N or N*mm.

    .. py:method:: set_E(lib=None)

        Assign modulus of elasticity for each beams.

        :param lib: dict of modulus of elasticity, unit = N/mm :superscript:`2`. If provided, application will request
                    specifying key of dict for each beam. Otherwise, non-zero positive value will be requested.
        :return: list. Assigned *Modulus of Elasticity* of beams, unit = N/mm :superscript:`2`.

    .. py:method:: set_A(lib=None)

        Assign section area for each beams.

        :param lib: dict of section area, unit = mm :superscript:`2`. If provided, application will request specifying
                    key of dict for each beam. Otherwise, non-zero positive value will be requested.
        :return: list. Assigned *Section Area* of beams, unit = mm :superscript:`2`.

    .. py:method:: set_I(lib=None)

        Assign moment of inertia for each beams.

        :param lib: dict of moment of inertia, unit = mm :superscript:`4`. If provided, application will request
                    specifying key of dict for each beam. Otherwise, non-zero positive value will be requested.
        :return: list. Assigned *Moment of Inertia* of beams, unit = mm :superscript:`4`.



