# -*- coding: utf-8 -*-
"""
Created on Wed June 03 11:36:40 2020
Last Update on 22/12/2020.

    + Fixing the bug of calling being reject due to application is busy.
    + Update the Docstrings.
    + Add method Acad.addleader. Add Sub-class CADFrame for 2D frame

@author: qi.wang
"""

import win32com.client
from win32com.client import gencache, constants
import win32gui
import pythoncom
import pywintypes
import re
import numpy as np
from numpy import pi, sqrt, sin, cos
import pandas as pd
from geomdl import BSpline
import json
import csv
import time
from functools import wraps

gencache.EnsureModule('{E2077CF2-3573-4E66-B1DC-01118675056D}', 0, 1, 0)

# Dict: AutoCAD File Type Enumerate
Filetype = {"ac2000_dwg": 12,
            "ac2004_dwg": 24,
            "ac2010_dwg": 48,
            "ac2013_dwg": 60,
            "ac2018_dwg": 64,
            "ac2000_dxf": 13,
            "ac2004_dxf": 25,
            "ac2010_dxf": 49,
            "ac2013_dxf": 61,
            "ac2018_dxf": 65,
            }

# Dict: RGB Color Mapping
Color_RGB = {"red": (255, 0, 0),
             "yellow": (255, 255, 0),
             "green": (0, 255, 0),
             "cyan": (0, 255, 255),
             "blue": (0, 0, 255),
             "magenta": (255, 0, 255),
             "gray": (128, 128, 128)}


# Class: Toolbox via AutoCAD API
class Acad():
    """AutoCAD Automation API.

    :param file_name: str, file name (and path) to open with application. Activate the specified file
                        if it is opened already. Otherwise, try to open the file from specified path.
    :param visible: bool, show AutoCAD Application UI after launching.
    :param hanging: float, time in seconds to wait when each time application is found busy.
    """

    def __init__(self, file_name=None, visible=True, hanging=0.5):
        # launch AutoCad Application
        if not Acad.findacad():  # if AutoCAD is not running
            print("Launching AutoCAD Program...", end="")
            new_app = True
        else:
            print("Connect to AutoCAD Program...", end="")
            new_app = False
        self.__app = win32com.client.Dispatch("AutoCAD.Application")

        # a pending system to ensure the application has been launched before further action
        attp = 0
        while True:
            try:
                if visible and new_app:  # only maximize the window when first launch
                    win32gui.ShowWindow(self.__app.HWND, 3)
                self.__app.Visible = visible
            except pywintypes.com_error:
                if attp < 5:  # try 5 times
                    print("...", end='')
                    time.sleep(hanging)  # pending 0.5 second
                    attp += 1
                else:
                    raise
            else:
                if new_app:
                    print(
                        f"\nAutoCAD ver. {self.__app.Version} has been launched successfully from <{self.__app.FullName}>")
                else:
                    print("Done!")
                break

        if file_name:  # open / activate requested drawing
            # self.__dwg = None
            for f in self.__app.Documents:
                if f.Name == file_name or f.FullName == file_name:
                    self.__dwg = f
                    self.__dwg.Activate()
                    while self.isbusy():
                        time.sleep(hanging)
                    break
            else:  # requested drawing in not found in opened files
                try:
                    self.__dwg = self.__app.Documents.Open(file_name)
                    print(f"Drawing <{file_name}> has been loaded successfully")
                except pywintypes.com_error:
                    raise OSError(f"Fail to load {file_name}")
        else:
            if self.__app.Documents.Count:  # if there's any drawing being opened already
                self.__dwg = self.__app.ActiveDocument
            else:
                self.new()
        time.sleep(0.5)

    @staticmethod
    def findacad():
        """Show *Windows Handle* of AutoCAD application.

        :return: list of int, handle of application windows.
        """
        w = []

        def findacad(hwnd, w):
            if r"Autodesk AutoCAD" in win32gui.GetWindowText(hwnd):
                w.append(hwnd)

        win32gui.EnumWindows(findacad, w)
        return w

    @property  # access for debug only
    def _app(self):
        return self.__app

    @property  # access for debug only
    def _doc(self):
        return self.__dwg

    @property
    def visible(self):
        """Visibility of application"""
        return self.__app.Visible

    @visible.setter
    def visible(self, isvisible):
        self.__app.Visible = isvisible
        if isvisible:
            win32gui.ShowWindow(self.__app.HWND, 3)

    def drawinglist(self, fullname=False):
        """Get a name list of drawings opened by application.

        :param fullname: bool, get the full path of the drawings.
        :return: list of str.
        """
        if fullname:
            return [d.FullName for d in self.__app.Documents]
        else:
            return [d.Name for d in self.__app.Documents]

    @property
    def drawing(self):
        """Name of operating drawing"""
        return self.__dwg.Name

    @drawing.setter
    def drawing(self, file_name):
        # switch operating drawing
        self.__dwg = None
        for f in self.__app.Documents:
            if f.Name == file_name or f.FullName == file_name:
                self.__dwg = f
                self.__dwg.Activate()
                break
        if not self.__dwg:  # requested drawing in not found in opened files
            try:
                self.__dwg = self.__app.Documents.Open(file_name)
                print(f"Drawing <{file_name}> has been loaded successfully")
            except:
                print(f"Fail to load {file_name}")
                raise

    def new(self, template="acadiso.dwt"):
        """Create a new drawing and set it as operating target.

        :param template: str, template name used to create new drawing.
        :return: None
        """
        self.__dwg = self.__app.Documents.Add(template)

    def current(self):
        """Set the active drawing as operating target.

        :return: str. name of the active drawing.
        """
        try:
            self.__dwg = self.__app.ActiveDocument
            return self.__dwg.Name
        except pywintypes.com_error as er:
            if er.excepinfo[5] == -2145320900:  # No active drawing
                print("No drawing is open currently.")
                self.__dwg = None
            else:
                raise

    def prompt(self, message):
        """Show a message in *Command Line* of AutoCAD UI.

        :param message: str, content of message.
        :return: None
        """
        self.__dwg.Utility.Prompt(message)

    def marco(self, macro_name):
        """Run pre-defined Macro of the drawing.

        :param macro_name: str. name of the Macro.
        :return: None
        """
        self.__app.RunMacro(macro_name)

    def update(self):
        """Renew the drawing.

        :return: None
        """
        self.__app.Update()

    def isbusy(self):
        """Check if application is busy now.

        :return: bool.
        """
        try:
            return not self.__app.GetAcadState().IsQuiescent
        except pywintypes.com_error as er:
            if er.args[0] == -2147418111:  # Call was rejected by callee
                # print("Call was rejected by callee")  # for debug
                return True
            else:
                raise

    def showwindow(self):
        """Bring the application window to the most front.

        :return: None.
        """
        if self.__app.WindowState == 2:  # if window is minimized
            self.__app.WindowState = 3  # maximize the window
        if win32gui.GetForegroundWindow() != self.__app.HWND:  # if AutoCAD application is not current active
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys('^')  # send a keyboard event: ctrl
            win32gui.SetForegroundWindow(self.__app.HWND)  # activate application window

    def showdwg(self, hanging=0.5):
        """ Show the operating drawing on screen.

        :param hanging: float, time in seconds of each waiting when application is busy.
        :return: None
        """
        self.showwindow()
        self.__dwg.Activate()
        while self.isbusy():  # hanging until application is ready for operation
            time.sleep(hanging)
        time.sleep(0.1)

    @staticmethod
    def point(x, y, z=0):
        """Define a point coordinates array.

        :param x: float, X-coordinate of the point.
        :param y: float, Y-coordinate of the point.
        :param z: float, Z-coordinate of the point.
        :return: ``win32com.client.VARIANT`` for VBA COM usage.
        """
        # translate a 3d array into VT_R8 form for VBA COM
        return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (x, y, z))

    @staticmethod
    class UCS():
        """A user defined orthogonal coordinate system.

        :param origin: 3-element array-like, global coordinates of origin point.
        :param point_x: 3-element array-like, global coordinates of a point lying on local x-direction.
                        By default, it follows WCS x-direction.
        :param point_ref: 3-element array-like, global coordinates of a point lying on positive y side of local
                          xy plane. By default, it use WCS y-direction.
        """
        def __init__(self, origin, point_x=None, point_ref=None):
            self.__origin = np.asarray(origin)
            x = np.asarray(point_x) - self.__origin if point_x else np.array([1, 0, 0])
            y = np.asarray(point_ref) - self.__origin if point_ref else np.array([0, 1, 0])
            self.__vx = x / np.linalg.norm(x)  # unit vector of x
            vy = y / np.linalg.norm(y)  # supposed unit vector of y
            if self.__vx @ vy == 0:  # if vx and vy is orthogonal
                self.__vy = vy
            else:  # build a orthogonal system according vector x and z
                vz = np.cross(self.__vx, vy)
                self.__vy = np.cross(vz, self.__vx) / np.linalg.norm(vz)
            self.__vz = np.cross(self.__vx, self.__vy)  # unit vector of z
            self.__m3 = np.column_stack([self.__vx, self.__vy, self.__vz])
            self.__m4 = np.vstack([np.column_stack([self.__m3, self.__origin]), np.array([0, 0, 0, 1])])

        @property
        def o(self):
            """Coordinates of origin point. Read-only."""
            return self.__origin

        @property
        def x(self):
            """Unit vector of x-direction. Read-only."""
            return self.__vx

        @property
        def y(self):
            """Unit vector of y-direction. Read-only."""
            return self.__vy

        @property
        def z(self):
            """Unit vector of z-direction. Read-only."""
            return self.__vz

        @property
        def m3(self):
            """[3x3] matrix of direction transformation from UCS to WCS. Read-only."""
            return self.__m3

        @property
        def m4(self):
            """[4x4] matrix of homogenous coordinate transformation from UCS to WCS. Read-only."""
            return self.__m4

        def toucs(self, point):
            """Translate a WCS coordinates to UCS.

            :param point: 3-element array-like, WCS coordinates of a point in the form of (X,Y,Z).
            :return: numpy.ndarray, corresponding coordinate on UCS in the form of (x,y,z).
            """
            pt_wcs = np.append(np.asarray(point), 1)
            pt_ucs = np.linalg.inv(self.__m4) @ pt_wcs
            return pt_ucs[:3]

        def fromucs(self, point):
            """Translate a UCS coordinates to WCS.

            :param point: 3-element array-like, UCS coordinates of a point in the form of (x,y,z).
            :return: numpy.ndarray, corresponding coordinate on WCS in the form of (X,Y,Z).
            """
            pt_ucs = np.append(np.asarray(point), 1)
            pt_wcs = self.__m4 @ pt_ucs
            return pt_wcs[:3]

    def get(self, intype="point", ref_pnt=None, prompt="Specify a point from drawing: ", bits=0, keyword="",
            default=""):
        """Get information from drawing.

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
        """

        # initialize input condition
        self.__dwg.Utility.InitializeUserInput(bits, keyword)
        self.showdwg()

        if intype in ["point", "angle", "distance", "orientation", "corner"]:
            func = "Get" + intype.title()
            try:
                if ref_pnt:
                    return getattr(self.__dwg.Utility, func)(self.point(*ref_pnt), prompt)
                else:
                    if intype == "corner":
                        return getattr(self.__dwg.Utility, func)(self.point(0, 0, 0), Prompt=prompt)
                    else:
                        return getattr(self.__dwg.Utility, func)(Prompt=prompt)

            except pywintypes.com_error as er:
                if er.excepinfo[5] == -2145320928:
                    print("A keyword has been received")
                    kw = self.__dwg.Utility.GetInput()  # get input keyword
                    return kw if kw else default
                else:
                    print("Canceled by user")

        else:
            raise ValueError("Unrecognized input type.")

    def kbinput(self, intype="real", prompt="", bits=0, keyword="", space=False, default=""):
        """Get keyboard input interactively.

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
        """

        self.__dwg.Utility.InitializeUserInput(bits, keyword)
        self.showdwg()

        try:
            if intype == "integer":
                return self.__dwg.Utility.GetInteger(Prompt=prompt)
            elif intype == "real":
                return self.__dwg.Utility.GetReal(Prompt=prompt)
            elif intype == "string":
                # this type receives any inputs as regular string while ignore settings about keyword
                s = self.__dwg.Utility.GetString(HasSpaces=int(space), Prompt=prompt)
                return s if s else default  # replace '' by default
            elif intype == "keyword":
                if keyword:  # must provide acceptable keyword list
                    reval = self.__dwg.Utility.GetKeyword(Prompt=prompt)
                    return reval if reval else default
                else:
                    raise ValueError("Keyword has not been defined.")
            else:
                raise ValueError("Unrecognized input type.")

        except pywintypes.com_error as er:
            if er.excepinfo[5] == -2145320928:
                print("A keyword has been received")
                kw = self.__dwg.Utility.GetInput()  # get input keyword
                return kw if kw else default
            else:
                print("Canceled by user")

    def pick(self, objtype=None, prompt="Select a entity from drawing: ", keyword=""):
        """Get a *AutoCAD Entity* from drawing by mouse-click selecting.

        :param objtype: str, type of requested entity either in formal name (e.g. 'AcDbLine') or short name
                        (e.g. 'line'). Any type of entity will be acceptable if not specified.
        :param prompt: str, prompt message.
        :param keyword: str,  keywords to be recognized when receiving input from keyboard, separate each keyword by
                        blank.
        :return: selected entity object or keyword.
        """

        self.__dwg.Utility.InitializeUserInput(1, keyword)  # the blank input is block.
        self.showdwg()
        print("Operating from AutoCAD...")

        try:
            self.prompt(prompt)  # send prompt
            entity, pkpnt = self.__dwg.Utility.GetEntity()
            while objtype and (not re.search(objtype.lower(), entity.ObjectName.lower())):
                # check if the selected entity has requested object type if specified
                self.prompt(f"Selected entity is not '{objtype}', please re-select: \n")
                entity, pkpnt = self.__dwg.Utility.GetEntity()  # re-select
            return entity
        except pywintypes.com_error as er:
            if er.excepinfo[5] == -2145320928:
                print("A keyword has been received.")
                return self.__dwg.Utility.GetInput()  # return keyword
            else:
                print("Nothing has been selected.")

    def select(self, objtype=None, prompt="Select entities from drawing: "):
        """Get a selection set by window selecting on drawing.

        :param objtype: str, type of requested entity either in formal name (e.g. 'AcDbLine') or short name
                        (e.g. 'line'). Any type of entity will be acceptable if not specified.
        :param prompt: str, prompt message.
        :return: a list of selected entities.
        """

        # create a temporary selection set
        try:
            temps = self.__dwg.SelectionSets.Add("temp_selection_set")
        except pywintypes.com_error as er:
            if er.excepinfo[5] == -2145320851:  # temp_selection_set is existing
                for s in self.__dwg.SelectionSets:
                    if s.Name == "temp_selection_set":
                        s.Clear()
                        temps = s
            else:
                raise

        self.showdwg()

        # select objects from drawing and add them into temporary selection set
        self.prompt(prompt)
        temps.SelectOnScreen()

        # filter selection and return as a list
        selected = []
        if temps:
            for item in temps:
                if objtype:
                    if re.search(objtype.lower(), item.ObjectName.lower()):
                        selected.append(self.__dwg.ObjectIdToObject(item.ObjectID))  # record object get via Object ID
                else:
                    selected.append(self.__dwg.ObjectIdToObject(item.ObjectID))

        temps.Delete()  # delete the temporary selection set before quit
        return selected

    def byid(self, obj_id):
        """Get a *AutoCAD Entity* by its object ID.

        :param obj_id: int, AUtoCAD object ID.
        :return: entity object.
        """
        try:
            return self.__dwg.ObjectIdToObject(obj_id)
        except pywintypes.com_error:
            raise ValueError("Invalid Object ID.")

    def readtable(self, table_obj, title=True, header=True, index=True):
        """Read data from a table in drawing

        :param table_obj: AcDbTable object
        :param title: bool, ignore the first row of table
        :param header: bool, read the second row of table as header of each column
        :param index: bool, read the first column of table as index of each row
        :return: pandas.DataFrame
        """
        # set table size
        rows = table_obj.Rows
        cols = table_obj.Columns
        start_row = 1 if title else 0
        start_col = 1 if index else 0

        # set headers
        if header:
            headers = [table_obj.GetCellValue(start_row, c) for c in range(start_col, cols)]
            start_row += 1
        else:
            headers = None  # use default headers

        # set indexes
        if index:
            indexes = [table_obj.GetCellValue(r, 0) for r in range(start_row, rows)]
        else:
            indexes = None

        # read table
        data = []
        for row in range(start_row, rows):
            data.append([table_obj.GetCellValue(row, c) for c in range(start_col, cols)])
        df = pd.DataFrame(data, index=indexes, columns=headers)

        return df.where(df.notna(), None)  # replace nan by None

    def command(self, comds):
        """Send a commands list to AutoCAD.

        :param comds: list of str, commends to be executed.
        :return: None
        """
        self.__dwg.SendCommand("\n".join(comds) + "\n")

    def setcolor(self, color, dwg_obj, *dwg_objs):
        """Set color of entity objects.

        :param color: str as color name, or a tuple of 3 int as RGB value.
        :param dwg_obj: entity object to be set color.
        :param dwg_objs: other entity objects to be set color.
        :return: None
        """

        c = self.__app.GetInterfaceObject(f"AutoCAD.AcCmColor.{self.__app.Version[:2]}")
        if color == "BYLAYER":  # set color by layer
            c.ColorMethod = 192
        elif color == "BYBLOCK":  # set color by block
            c.ColorMethod = 193
        else:
            if type(color) == str:  # translate color name to RGB value
                color = Color_RGB[color]
            c.SetRGB(*color)
        for obj in [dwg_obj] + list(dwg_objs):
            try:
                obj.TrueColor = c
            except AttributeError:
                print(f"Can't set color of Object <{obj.ObjectName}>")
                continue

    def setlinetype(self, ltype, dwg_obj, *dwg_objs, scale=None, lib="acadiso.lin"):
        """Set line type  of entity objects.

        :param ltype: str, name of line type.
        :param dwg_obj: entity object to be set line type.
        :param dwg_objs: other entity objects to be set line type.
        :param scale: float, line type scale. Remain unchanged if not specified.
        :param lib: str, library that specified line type is loaded from if it has not been loaded yet.
        :return: None
        """
        if ltype not in [lt.Name for lt in self.__dwg.Linetypes]:  # load line type if it is not loaded
            self.__dwg.Linetypes.Load(ltype, lib)

        for l in [dwg_obj] + list(dwg_objs):
            try:
                l.Linetype = ltype
                if scale:
                    l.LinetypeScale = scale
            except AttributeError:
                print(f"Can't set line type to <{l.ObjectName}>")
                continue

    def setlayer(self, layer_name, color=None, ltype=None, lweight=None, plottable=None,
                 hidden=None, freeze=None, lock=None, activate=False):
        """Change settings of a layer.

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
        """

        # create a layer, existing layer with duplicated name will be overwritten
        layer = self.__dwg.Layers.Add(layer_name)

        if activate:
            self.__dwg.ActiveLayer = layer

        if color:
            self.setcolor(color, layer)
        if ltype:
            self.setlinetype(ltype, layer)
        if lweight is not None:
            layer.Lineweight = lweight
        if plottable is not None:
            layer.Plottable = plottable
        if hidden is not None:
            layer.LayerOn = not hidden
        if freeze is not None:
            try:
                layer.Freeze = freeze
            except pywintypes.com_error as er:
                if er.excepinfo[5] == -2145386348:
                    print("Can't freeze current active layer")
                else:
                    raise
        if lock is not None:
            layer.Lock = lock

        return layer

    def addline(self, point_1, point_2, *other_points, close=False, polyline=False, line_width=0, color=None,
                ltype=None, scale=None, layer=None):
        """Draw straight lines on drawing.

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
        """

        pts = [point_1, point_2] + list(other_points)
        if close and other_points:
            pts.append(point_1)

        if polyline:  # draw 2D polyline
            vl = []
            for p in pts:
                vl.extend(p[:2])
            vl = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, vl)  # vertices list
            pl = self.__dwg.ModelSpace.AddLightWeightPolyline(vl)
            if line_width:
                pl.ConstantWidth = line_width
            if color:
                self.setcolor(color, pl)
            if ltype or scale:
                self.setlinetype(ltype if ltype else self.__dwg.ActiveLinetype.Name, pl, scale=scale)
            if layer:
                pl.Layer = layer
            return pl

        else:  # draw separated lines
            lines = []
            for i in range(len(pts) - 1):
                lines.append(self.__dwg.ModelSpace.AddLine(self.point(*pts[i]), self.point(*pts[i + 1])))
            if color:
                self.setcolor(color, *lines)
            if ltype or scale:
                self.setlinetype(ltype if ltype else self.__dwg.ActiveLinetype.Name, *lines, scale=scale)
            if layer:
                for l in lines:
                    l.Layer = layer
            return lines

    def addcurve(self, *fitpoints, start_tan=[0, 0, 0], end_tan=[0, 0, 0], color=None, ltype=None, scale=None,
                 layer=None):
        """Draw Nurbs curve as spline passing through specified fit points.

        :param fitpoints: Nested list of float like [[x1,y1,z1], [x2,y2,z2]...], represents coordinate of fit points.
        :param start_tan: list of float, 3D vector of tangency at start point
        :param end_tan: list of float, 3D vector of tangency at end point
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw curve on. If not given, draw curve on current layer.
        :return: created spline object.
        """
        if len(fitpoints) >= 3:  # at least 3 points
            fp = []
            for p in fitpoints:
                fp.extend(p)
            fp = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, fp)  # fit point list
            st = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, start_tan)  # tan at start point
            et = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, end_tan)  # tan at end point
            spl = self.__dwg.ModelSpace.AddSpline(fp, st, et)
            if color:
                self.setcolor(color, spl)
            if ltype or scale:
                self.setlinetype(ltype if ltype else self.__dwg.ActiveLinetype.Name, spl, scale=scale)
            if layer:
                spl.Layer = layer
            return spl
        else:
            raise ValueError("Not enough fit points. At least provide 3 points for fitting a curve.")

    def addrect(self, corner_1, corner_2, line_width=0, color=None, ltype=None, scale=None, layer=None):
        """Shortcut of drawing a rectangle by defining its 2 opposite corners.

        :param corner_1: list of float [x,y,z], coordinate of one corner point.
        :param corner_2: list of float [x,y,z], coordinate of the opposite corner point.
        :param line_width: float, global width of polyline.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw rectangle on. If not given, draw rectangle on current layer.
        :return: created of polyline object.
        """
        pt1 = corner_1
        pt2 = [corner_2[0], corner_1[1], corner_1[2]]
        pt3 = [corner_2[0], corner_2[1], corner_1[2]]
        pt4 = [corner_1[0], corner_2[1], corner_1[2]]
        return self.addline(pt1, pt2, pt3, pt4, line_width=line_width, close=True, polyline=True, color=color,
                            ltype=ltype, scale=scale, layer=layer)

    def addcircle(self, center, radius, color=None, ltype=None, scale=None, layer=None):
        """Draw a circle by defining center point and radius.

        :param center: list of float [x,y,z], coordinate of center point.
        :param radius: float, radius of circle.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw rectangle on. If not given, draw rectangle on current layer.
        :return: created circle object.
        """
        center = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, center)
        cir = self.__dwg.ModelSpace.AddCircle(center, radius)
        if color:
            self.setcolor(color, cir)
        if ltype or scale:
            self.setlinetype(ltype if ltype else self.__dwg.ActiveLinetype.Name, cir, scale=scale)
        if layer:
            cir.Layer = layer

        return cir

    def addnode(self, point, *other_points, layer=None):
        """Mark nodes at specified locations.

        :param point: list of float [x,y,z], coordinate to mark node at.
        :param other_points: 3-element lists [x,y,z], other coordinates of to mark nodes at.
        :param layer: str, name of layer to mark node on. If not given, mark node on current layer.
        :return: created node object, or a list of node objects.
        """
        if other_points:
            nds = []
            for p in [point] + list(other_points):
                nd = self.__dwg.ModelSpace.AddPoint(win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, p))
                if layer:
                    nd.Layer = layer
                nds.append(nd)
            return nds
        else:
            nd = self.__dwg.ModelSpace.AddPoint(win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, point))
            if layer:
                nd.Layer = layer
            return nd

    def fillhatch(self, outerloops, innerloops=[], pattern='ANSI31', angle=0.0, scale=1.0, asso=False,
                  color=None, layer=None):
        """Fill the entities by specified pattern.

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
        """

        ha = self.__dwg.ModelSpace.AddHatch(constants.acHatchPatternTypePreDefined, pattern, asso)
        # ha.AppendOuterLoop(win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, outerloop))
        for olp in outerloops:
            ha.AppendOuterLoop(win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, olp))
        for ilp in innerloops:
            ha.AppendInnerLoop(win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, ilp))
        ha.PatternAngle = angle
        ha.PatternScale = scale
        if color:
            self.setcolor(color, ha)
        if layer:
            ha.Layer = layer

        return ha

    def adddim(self, point_1, point_2, offset, measure_dir=None, dimstyle=None, layer=None):
        """Add a aligned or rotated dimension annotation on x-y plane measuring the distance between 2 points.

        :param point_1: list of float [x,y,z], coordinates of first point.
        :param point_2: list of float [x,y,z], coordinates of second point.
        :param offset: float, offset distance of dimension annotation from measured line.
        :param measure_dir: list of float [vx,vy,vz], direction vector which measurement align with. If not given, align
                            with the line connecting two measured points.
        :param dimstyle: str, name of dimension style. Use current active style if not specified.
        :param layer: str, name of layer to create annotation on. If not given, create annotation on current layer.
        :return: created dimension object.
        """

        pt1 = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, point_1)
        pt2 = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, point_2)

        if measure_dir:
            da = self.vangle(measure_dir)
            if abs(measure_dir[0]) < 1e-8:  # vertical dim:
                if measure_dir[1] >= 0:
                    po = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                                 (-offset + point_1[0], point_1[1], 0))
                else:
                    po = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                                 (offset + point_1[0], point_1[1], 0))
                # print(f"po vertical case={po.value}")
            else:
                b = measure_dir[1] / measure_dir[0]
                po = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (1 + point_1[0],
                                                                                    offset * sqrt(1 + b ** 2) + b +
                                                                                    point_1[1],
                                                                                    0))
                # print(f"po non-vertical case={po.value}")
            dim = self.__dwg.ModelSpace.AddDimRotated(pt1, pt2, po, da)

        else:
            if abs(point_1[0] - point_2[0]) < 1e-8:  # vertical dim:
                po = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                             (-offset + point_1[0], point_1[1], 0))


            else:
                b = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
                po = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, (1 + point_1[0],
                                                                                    offset * sqrt(1 + b ** 2) + b +
                                                                                    point_1[1],
                                                                                    0))
            dim = self.__dwg.ModelSpace.AddDimAligned(pt1, pt2, po)

        if dimstyle:
            dim.StyleName = dimstyle
        if layer:
            dim.Layer = layer

        return dim

    def addleader(self, point_1, point_2, *other_points, style=None, ltype=None, scale=None, color=None, arrow=None,
                  headsize=None, layer=None, spline=False):
        """Add a straight or curved leader line on drawing.

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
        """
        pts = list(point_1) + list(point_2)
        for p in other_points:
            pts.extend(p)
        pts = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, pts)
        lds = self.__dwg.ModelSpace.AddLeader(pts, None, 3 if spline else 2)  # leader with arrow, no attached object

        if style:
            lds.StyleName = style
        if ltype or scale:
            self.setlinetype(ltype if ltype else self.__dwg.ActiveLinetype.Name, lds, scale=scale)
        if color:
            lds.DimensionLineColor = 0  # Line color is ByBlock
            self.setcolor(color, lds)  # set the True Color of leader block
        if arrow:
            lds.ArrowheadType = arrow  # arrow type index 0~19
        if headsize:
            lds.ArrowheadSize = headsize
        if layer:
            lds.Layer = layer

        return lds

    def insertblock(self, insert_pnt, block_name, scale=(1.0, 1.0, 1.0), rotation=0, dynamic_prop=None, attr=None,
                    layer=None):
        """ Insert a block to current drawing.

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
        """
        pt = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, insert_pnt)
        blk = self.__dwg.ModelSpace.InsertBlock(pt, block_name, scale[0], scale[1], scale[2], rotation)

        # set customize properties for dynamic block
        if dynamic_prop and blk.IsDynamicBlock:
            dp = blk.GetDynamicBlockProperties()
            name = [p.PropertyName for p in dp]
            for i, v in dynamic_prop.items():
                if type(i) == str:  # identify by property name
                    dp[name.index(i)].Value = v
                else:  # identify by property item number
                    dp[i].Value = v

        # set attributes of block
        if attr and blk.HasAttributes:
            ar = blk.GetAttributes()
            tag = [a.TagString for a in ar]
            for i, v in attr.items():
                if type(i) == str:  # identify by tag name
                    ar[tag.index(i)].TextString = v
                else:  # identify by attributes item number
                    ar[i].TextString = v

        if layer:
            blk.Layer = layer

        return blk

    def makeregion(self, objects=None, layer=None, del_source=True):
        """Create region from selected or provided objects.

        :param objects: list of objects. If not given, interactive selecting on screen will be requested.
        :param del_source: bool, delete the source objects after region being created.
        :param layer: str, name of layer to create generated region on. If not given, create region on current layer.
        :return: list of created regions.
        """
        if not objects:  # when object list is not provided, select on screen
            objects = self.select(prompt="Select objects:")

        try:
            reg = self.__dwg.ModelSpace.AddRegion(
                win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, objects))
        except pywintypes.com_error:
            raise RuntimeError("Fail to create region from specified objects.")

        if layer:
            for r in reg:
                r.Layer = layer

        if del_source:
            for item in objects:
                item.Delete()

        return reg

    def textstyle(self, style_name, font_file, bigfont_file='', bold=False, italic=False, regen=True, activate=False):
        """Define a text style.

        :param style_name: str, name of text style, existing text style with same name will be overwritten.
        :param font_file: str, path and name of font file.
        :param bigfont_file: str, path and  name of big font file.
        :param bold: bool, bold font style.
        :param italic: bool, italic font style.
        :param regen: bool, regenerate the drawing. Modification on existing text style will only be shown after
                      regeneration
        :param activate: bool, set the new defined text style as the active one.
        :return: None
        """
        stl = self.__dwg.TextStyles.Add(style_name)
        stl.fontFile = font_file
        if font_file[-3:] == "shx" and bigfont_file:  # only applicable to shx font
            stl.BigFontFile = bigfont_file
        ft = stl.GetFont()
        stl.SetFont(ft[0], bold, italic, ft[3], ft[4])
        if regen:
            self.__dwg.Regen(constants.acAllViewports)
        if activate:
            self.__dwg.ActiveTextStyle = stl

    def addtext(self, content, insert_point, height, style=None, align_type="Left", rotation=0, color=None, layer=None):
        """Add single-line text on drawing.

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
        """

        bpt = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, insert_point)
        txt = self.__dwg.ModelSpace.AddText(content, bpt, height)
        if style:
            txt.StyleName = style
        if align_type != "Left":  # if align type is not default
            txt.Alignment = getattr(constants, f"acAlignment{align_type}")
            txt.TextAlignmentPoint = bpt  # use TextAlignmentPoint for alignment type other than 'Left'
        if rotation:
            txt.Rotation = rotation
        if color:
            self.setcolor(color, txt)
        if layer:
            txt.Layer = layer
        return txt

    def addmtext(self, content, insert_point, height, width, style=None, align_type="TopLeft", rotation=0, color=None,
                 layer=None):
        """ Add multi-line text zone on drawing.

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
        """

        bpt = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, insert_point)
        mtxt = self.__dwg.ModelSpace.AddMText(bpt, width, content)
        mtxt.Height = height
        if style:
            mtxt.StyleName = style
        if align_type != "TopLeft":  # if align type is not default
            mtxt.AttachmentPoint = getattr(constants, f"acAttachmentPoint{align_type}")
            mtxt.InsertionPoint = bpt
        if rotation:
            mtxt.Rotation = rotation
        if color:
            self.setcolor(color, mtxt)
        if layer:
            mtxt.Layer = layer

        return mtxt

    def addtable(self, table_data, insert_point, row_height, col_width, layer=None, title=None, show_index=False,
                 index=[],
                 show_header=False, headers=[], acc=2, title_textheight=None, header_textheight=None,
                 main_textheight=None, title_style=None, header_style=None, main_style=None,
                 title_align=None, header_align=None, main_align=None, cell_style=None, cell_color=None, merge_empty=0):
        """Insert a table on drawing according to input data.

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
        """

        bpt = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, insert_point)

        # determine the shape of input data
        try:
            dims = table_data.ndim
        except AttributeError:  # not a DataFrame, Series, or Numpy Array
            nr = len(table_data)
            nc = 1
        else:
            if dims == 1:
                nr, = table_data.shape
                nc = 1
            elif dims == 2:
                nr, nc = table_data.shape
            else:
                raise TypeError("Unsupported data type.")

        # determine the table size
        start_row = 2  # initial case: table is created with title and column name
        start_col = 1 if show_index else 0

        # create a table draft
        const_rh, const_cw = False, False
        try:
            rh = max(row_height)
        except TypeError:
            rh = row_height
            const_rh = True
        try:
            cw = max(col_width)
        except TypeError:
            cw = col_width
            const_cw = True
        tb = self.__dwg.ModelSpace.AddTable(bpt, nr + start_row, nc + start_col, rh, cw)

        # set text style
        if title_style:
            tb.SetTextStyle(2, title_style)
        if header_style:
            tb.SetTextStyle(4, header_style)
        if main_style:
            tb.SetTextStyle(1, main_style)

        # set text height
        if title_textheight:
            tb.SetTextHeight(2, title_textheight)
        if header_textheight:
            tb.SetTextHeight(4, header_textheight)
        if main_textheight:
            tb.SetTextHeight(1, main_textheight)

        # set alignment
        if title_align:
            tb.SetAlignment(2, getattr(constants, f"ac{title_align}"))
        if header_align:
            tb.SetAlignment(4, getattr(constants, f"ac{header_align}"))
        if main_align:
            tb.SetAlignment(1, getattr(constants, f"ac{main_align}"))

        # set title
        if title:
            tb.SetCellValue(0, 0, title)  # write table title
        else:
            tb.DeleteRows(0, 1)  # delete the title cell
            start_row -= 1  # table without title

        # set header
        if show_header:
            if not headers:  # column name is not provided
                try:
                    headers = table_data.columns  # read columns from data frame
                except AttributeError:
                    headers = [f'C{n}' for n in range(nc)]  # use default column name
            for i in range(nc):
                tb.SetCellValue(start_row - 1, i + start_col, headers[i])
        else:
            tb.DeleteRows(1, 1)  # delete the column name cell
            start_row -= 1  # table without column name

        # set index
        if show_index:
            if not index:  # index is not provided
                if isinstance(table_data, (pd.DataFrame, pd.Series)):
                    index = table_data.index  # read index from data frame or series
                else:
                    index = [f'R{n}' for n in range(nr)]  # use default index
            for i in range(nr):
                tb.SetCellValue(i + start_row, 0, index[i])

        # fill table
        def fillcell(row, col, val):
            if pd.notna(val):  # val is not None or NaN
                tb.SetCellValue(row, col, val)
                if isinstance(val, (int, np.integer)):
                    tb.SetCellDataType(row, col, 1, 0)  # acLong type
                elif isinstance(val, (float, np.float)):
                    tb.SetCellDataType(row, col, 2, 0)  # acDouble type
                    tb.SetCellFormat(row, col, f'%lu2%pr{acc}')  # set decimal place
                elif isinstance(val, str):
                    tb.SetCellDataType(row, col, 4, 0)  # acString type
                else:
                    tb.SetCellDataType(row, col, 512, 0)  # acGeneral type
            else:
                if merge_empty == 1 and col > start_col:  # merge to left cell horizontally
                    if tb.IsMergedCell(row, col - 1)[0]:
                        left = tb.IsMergedCell(row, col - 1)[3]  # the left bound of merged cell.
                    else:
                        left = col - 1  # the left cell
                    tb.MergeCells(row, row, left, col)
                elif merge_empty == 2 and row > start_row:  # merge to upper cell horizontally
                    if tb.IsMergedCell(row - 1, col)[0]:
                        upper = tb.IsMergedCell(row - 1, col)[1]  # the upper bound of merged cell.
                    else:
                        upper = row - 1  # the left cell
                    tb.MergeCells(upper, row, col, col)

        if isinstance(table_data, pd.DataFrame):  # if input is a DataFrame
            for r in range(nr):
                for c in range(nc):
                    fillcell(r + start_row, c + start_col, table_data.iloc[r, c])
        else:  # Series, array, list, tuple
            if nc == 1:  # one column only
                for r in range(nr):
                    fillcell(r + start_row, start_col, table_data[r])
            else:  # multiple columns
                for r in range(nr):
                    for c in range(nc):
                        fillcell(r + start_row, c + start_col, table_data[r, c])

        # set individual row height and column width
        if not const_rh:
            for i in range(len(row_height)):
                tb.SetRowHeight(i, row_height[i])
        if not const_cw:
            for i in range(len(col_width)):
                tb.SetColumnWidth(i, col_width[i])

        # set individual text style
        if cell_style:
            for (row, col), stl in cell_style.items():
                tb.SetTextStyle2(row, col, 0, stl)

        # set individual cell color
        if cell_color:
            c = self.__app.GetInterfaceObject(f"AutoCAD.AcCmColor.{self.__app.Version[:2]}")
            for (row, col), color in cell_color.items():
                if type(color) == str:  # translate color name to RGB value
                    color = Color_RGB[color]
                c.SetRGB(*color)
                tb.SetContentColor2(row, col, 0, c)

        if layer:
            tb.Layer = layer

        return tb

    def save(self, full_file_name=None, version=2013):
        """Save the drawing file.

        :param full_file_name: str. path and file name to save the drawing as, including extension *'.dwg'* or *'.dxf'* .
                               If not given, save the drawing in-place.
        :param version: int. version of AutoCAD file. Supported version are: *2000, 2004, 2010, 2013, 2018*
        :return: None.
        """
        if full_file_name:
            try:
                ftype = getattr(constants, f'ac{version}_{full_file_name[-3:]}')
            except KeyError:
                raise ValueError("Unsupported file type")
            self.__dwg.SaveAs(full_file_name, SaveAsType=ftype)
            print(f"File has been saved as <{full_file_name}>")

        else:  # no file name specified
            self.__dwg.Save()
            print("File has been saved.")

    def close(self):
        """Discard changes and close the drawing.

        Operating target will be shifted to next active drawing.

        :return: bool, flag states if all the drawings have been closed.
        """
        n = self.__dwg.Name
        self.__dwg.Close(SaveChanges=False)  # always discard changes when close
        print(f"{n} is closed.")
        new_dwg = self.current()
        if new_dwg:
            print(f"{new_dwg} is now being operated.")  # switch to new active drawing
            return False
        else:
            return True

    # region =============================== TOOLBOX ======================================

    def getvector(self, unit=True):
        """Create a vector by specifying two point on drawing.

        :param unit: bool, return unit vector.
        :return: numpy.ndarray represents a 3-dimensional vector.
        """
        base_pt = self.get(prompt="Specify base point of vector, default is (0,0,0):", default=(0, 0, 0))
        if base_pt:  # base point is specified
            v_pt = self.get(ref_pnt=base_pt, prompt="Specify the second point of vector:", bits=1)
            if v_pt:
                vector = np.array(v_pt) - np.array(base_pt)
                if unit:
                    return vector / np.linalg.norm(vector)
                else:
                    return vector

    # get the plane orientation in domain 0 to 2*pi (0deg to 360deg) of a vector
    @staticmethod
    def vangle(vector, rad=True):
        """Get projected orientation of a vector on *X-Y Plane*, in range from 0 to 2*pi.

        :param vector: array-like [vx, vy ,vz], the vector
        :param rad: return the orientation angle in radians. When False, return the angle in degree
        :return: float, the solved angle.
        """
        x, y, *_ = vector
        if y >= 0:
            ang = np.arccos(x / sqrt(x ** 2 + y ** 2))
        else:
            ang = pi * 2 - np.arccos(x / sqrt(x ** 2 + y ** 2))
        if rad:
            return ang
        else:
            return ang / pi * 180

    @staticmethod
    def tolerate(var_1, var_2, dim=3, acc=1e-6):
        """Check whether two points or vectors are equivalent within tolerance.

        :param var_1: array-like of float, the coordinate of first point, or the first vector.
        :param var_2: array-like of float, the coordinate of second point, or the second vector.
        :param dim: int. number of dimensions to compare.
        :param acc: float. allowable tolerance.
        :return: bool.
        """
        return all([abs(var_1[i] - var_2[i]) <= acc for i in range(dim)])

    @staticmethod
    def rotatevec(vector, angle):
        """Rotate a vector by specified angle.

        :param vector: array-like of float [vx, vy, vz], the vector to be rotated.
        :param angle: float, rotating angle in radians. take counter-clockwise as positive.
        :return: array-like of float, the rotated vector.
        """
        v = np.asarray(vector)
        rm = np.array([[cos(angle), -sin(angle), 0],
                       [sin(angle), cos(angle), 0],
                       [0, 0, 1]])
        return rm @ v

    def addpolygon(self, insert_point, n, radius, start_angle=None, line_width=0, color=None, ltype=None, scale=None,
                   layer=None):
        """Shortcut to drawing a polygon.

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
        """
        alpha = pi / n  # half of inner angle
        if start_angle is None:
            start_angle = alpha - pi * 0.5
        pts = []
        for a in np.arange(start_angle, start_angle + pi * 2, 2 * alpha):
            tm = np.array([[cos(a), -sin(a), 0, insert_point[0]],
                           [sin(a), cos(a), 0, insert_point[1]],
                           [0, 0, 1, insert_point[2]],
                           [0, 0, 0, 1]])
            pt_wcs = tm @ np.array([radius, 0, 0, 1])
            pts.append(pt_wcs[:3])
        # for p in pts:
        #     print(p)
        return self.addline(*pts, close=True, polyline=True, line_width=line_width, color=color, ltype=ltype,
                            scale=scale, layer=layer)

    def freedraw(self, polyline=False, line_width=0, color=None, ltype=None, scale=None, layer=None):
        """Shortcut to continuously draw lines or a polyline in interactive way.

        :param polyline: bool, draw a polyline instead of individual lines.
        :param line_width: float, global width of polyline. Only valid when *polyline* = True.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :param ltype: str, name of line type. If not given, use current active type.
        :param scale: float, line type scale. If not given, use default scale = 1.
        :param layer: str, name of layer to draw on. If not given, draw on current layer.
        :return: created polyline object, or list of created lines.
        """

        pts = [self.get(prompt='Specify the first point of polyline:')]
        lines = []
        close = False
        while True:
            pt = self.get(ref_pnt=pts[-1], prompt='Specify the next point of polyline, or [Close]:', bits=0,
                          keyword="c")
            if not pt:  # user quit
                break
            elif pt == 'c':
                if len(pts) >= 3:
                    lines.append(self.addline(pts[-1], pts[0], line_width=line_width, color=color,
                                              ltype=ltype, scale=scale, layer=layer)[0])
                    close = True
                    break
                else:
                    self.prompt("Insufficient points to form a closed shape.")
            else:
                lines.append(self.addline(pts[-1], pt, line_width=line_width, color=color, ltype=ltype, scale=scale,
                                          layer=layer)[0])
                pts.append(pt)

        # create polyline if requested
        if polyline:
            pl = self.addline(*pts, close=close, polyline=True, line_width=line_width, color=color, ltype=ltype,
                              scale=scale, layer=layer)
            for l in lines:
                l.Delete()
            return pl
        else:
            return lines

    # read blocks location at insert point
    def locate(self, block_name=None, ucs=None):
        """Get the coordinate of insert point of selected blocks.

        :param block_name: str, name of the block to be located. If not given, a sample block will be requested to be
                            selected from drawing.
        :param ucs: ``Acad.UCS`` object, return the coordinate of insert points in this UCS. If not given, return the
                     coordinates in WCS.
        :return: list of tuple.
        """
        if not block_name:
            sample = self.pick(objtype='block', prompt="select a sample")
            if not sample:
                print("no block name or block sample has been specified")
                return []
            else:
                block_name = sample.Name
        blks = self.select(objtype='block', prompt="select blocks:")
        if ucs:
            return [list(ucs.toucs(b.InsertionPoint)) for b in blks if b.Name == block_name]
        else:
            return [b.InsertionPoint for b in blks if b.Name == block_name]

    def readsecprop(self, sort=1, sec_name=None, file_name=None):
        """Read section properties data from selected tables in drawing.

        This method is experimental and only works on table in specific form.

        :param sort: int, method of sorting selected tables.

                        | 0: no sorting.
                        | 1: vertical priority, from upper to lower.
                        | 2: horizontal priority, from left to right.

        :param sec_name: list of str, section name used as header of each column of data. If not given, Use default name
                        such as 'Section_01', 'Section_02' etc.
        :param file_name: str, file name to for data export, with extension of '.csv' or '.json'. If not
                        specified, exporting procedure will be skipped.
        :return: pandas.DataFrame
        """

        tb_filter = lambda x: (x.Rows == 21) and ("section properties" in str(x.GetCellValue(0, 0)).lower())
        tbs = list(filter(tb_filter, self.select(objtype='table', prompt="select tables of section properties")))

        if not tbs:
            print("No valid table has been selected.")
            return None

        if sort == 1:  # sort selected table from left to right, then upper to lower
            def sf(x):
                p = getattr(x, "InsertionPoint")
                return [-p[1], p[0], p[2]]

            tbs.sort(key=sf)
        elif sort == 2:  # sort selected table from upper to lower, then left to right
            def sf(x):
                p = getattr(x, "InsertionPoint")
                return [p[0], -p[1], p[2]]

            tbs.sort(key=sf)

        # symbol of section properties
        indexes = ['A', 'P', 'Bx', 'By', 'Ix', 'Iy', 'Ixy', 'rx', 'ry', 'Zx', 'Zy', 'Sx', 'Sy',
                   'I1', 'I2', 'alpha', 'Z1', 'Z2', 'S1', 'S2']

        if not sec_name or len(sec_name) != len(tbs):  # define default section name
            sec_name = [f"Section_{n + 1:0>2}" for n in range(len(tbs))]

        props = pd.DataFrame([], index=indexes)  # create a empty data summary
        for i in range(len(tbs)):
            df = self.readtable(tbs[i], header=False)  # read table
            df.index = indexes  # rename index
            for r, d in df.iterrows():  # merge two columns into one
                if d[1]:
                    d[0] = [d[0], d[1]]
            props[sec_name[i]] = df.iloc[:, 0]  # add to data summary

        if file_name:  # export to file
            if file_name[-3:] == 'csv':
                props.to_csv(file_name)
                print(f"Properties of {len(tbs)} Sections have been exported to <{file_name}>")
            elif file_name[-4:] == 'json':
                props.to_json(file_name)
                print(f"Properties of {len(tbs)} Sections have been exported to <{file_name}>")
            else:
                raise ValueError("Unsupported export file type.")

        return props

    def getsecprop(self, region_obj, file_name=None):
        """Analyze section properties of a region.

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
                'S2'        elastic section modulus about minor axis
                ========  ==================================================

        """
        if region_obj.ObjectName != 'AcDbRegion':
            raise TypeError("input drawing entity must be <'AcDbRegion'>")

        print("Analysing Section Properties...")
        # read section properties
        Ar = region_obj.Area
        cp_x, cp_y = region_obj.Centroid
        Ix_, Iy_ = region_obj.MomentOfInertia
        Ixy_ = -region_obj.ProductOfInertia  # reverse sign mistake from AutoCAD2019
        (x_min, y_min, z_min), (x_max, y_max, z_max) = region_obj.GetBoundingBox()

        # section properties about x-y
        Ix = Ix_ - Ar * cp_y ** 2
        Iy = Iy_ - Ar * cp_x ** 2
        Ixy = Ixy_ - Ar * cp_x * cp_y
        Zx = Ix / max(cp_y - y_min, y_max - cp_y)
        Zy = Iy / max(cp_x - x_min, x_max - cp_x)
        Sx = self.getplasec(region_obj)
        Sy = self.getplasec(region_obj, axis_angle=pi / 2)

        # section properties about principal axis
        I_1, I_2 = region_obj.PrincipalMoments
        PriD = region_obj.PrincipalDirections
        if (abs(Ix - Iy) < 1e-6) and (abs(Ixy) < 1e-6):  # case of infinite principal axis
            alpha = 0
        else:
            if I_1 < I_2:
                alpha = -pi / 2 if PriD[2] == 0 else np.arctan(-PriD[3] / PriD[2])
            else:
                alpha = pi / 2 if PriD[0] == 0 else np.arctan(-PriD[1] / PriD[0])

        if I_1 < I_2:
            I_1, I_2 = I_2, I_1  # always let 1_1 as major axis.

        if alpha:  # rotate region to position so that major axis is horizontal
            region_obj.Rotate(self.point(cp_x, cp_y), -alpha)

        (x_min_, y_min_, z_min_), (x_max_, y_max_, z_max_) = region_obj.GetBoundingBox()
        Z_1 = I_1 / max(cp_y - y_min_, y_max_ - cp_y)
        Z_2 = I_2 / max(cp_x - x_min_, x_max_ - cp_x)
        S_1 = self.getplasec(region_obj)
        S_2 = self.getplasec(region_obj, axis_angle=pi / 2)

        if alpha:  # rotate back
            region_obj.Rotate(self.point(cp_x, cp_y), alpha)

        props = {'A': Ar, 'Ix': Ix, 'Iy': Iy, 'Ixy': Ixy, 'Zx': Zx, 'Zy': Zy, 'Sx': Sx, 'Sy': Sy,
                 'I1': I_1, 'I2': I_2, 'alpha': alpha, 'Z1': Z_1, 'Z2': Z_2, 'S1': S_1, 'S2': S_2}

        if file_name:  # save as json file
            with open(file_name, 'w') as f:
                json.dump(props, f)

        return props

    def getplasec(self, region_obj, axis_angle=0, acc=1e-6):
        """Calculate plastic section modulus about a specified axis of a region.

        :param region_obj: Region object.
        :param axis_angle: float, orientation angle of specified axis in radians.
        :param acc: float, allowable tolerance when finding the center line of the section.
        :return: float, calculated plastic section modulus.
        """

        if region_obj.ObjectName != 'AcDbRegion':
            raise TypeError("input drawing entity must be <'AcDbRegion'>")

        cp_x, cp_y = region_obj.Centroid
        if axis_angle:  # rotate region to position so that axis is horizontal
            region_obj.Rotate(self.point(cp_x, cp_y), -axis_angle)

        ar = region_obj.Area

        (x_min, y_min, z_min), (x_max, y_max, z_max) = region_obj.GetBoundingBox()
        oper = constants.acIntersection
        y_bottom = y_min

        while True:
            y = (y_min + y_max) / 2
            cutbox = \
                self.makeregion([self.addrect((x_min - 1, y_bottom - 1, z_min), (x_max + 1, y, z_max))],
                                del_source=True)[0]
            lower_half = region_obj.Copy()
            lower_half.Boolean(oper, cutbox)
            # print(f'difference: {lower_half.Area} - {0.5*ar} = {lower_half.Area - 0.5*ar}')
            if lower_half.Area - 0.5 * ar > acc:  # lower half is larger
                y_max = y
            elif lower_half.Area - 0.5 * ar < -acc:  # lower half is smaller
                y_min = y
            else:  # tolerance is within requested
                break
            lower_half.Delete()

        lower_cx, lower_cy = lower_half.Centroid
        upper_cy = y_bottom + (2 * (cp_y - y_bottom) - (lower_cy - y_bottom))

        lower_half.Delete()

        if axis_angle:  # rotate back
            region_obj.Rotate(self.point(cp_x, cp_y), axis_angle)

        return (upper_cy - lower_cy) * ar * 0.5

    def getboundary(self, region_obj, spl_sub=100, file_name=None):
        """Get vector from centroid of a section to its boundary corner, arcs or ellipse.

        :param region_obj: Region object.
        :param spl_sub: int, numbers of subdivided segments of a spline boundary.
        :param file_name: str, file name for data exporting, with extension of '.json' or '.csv'. If not specified,
                          exporting procedure will be skipped.
        :return: tuple in the form of:

                | ([vector_to_corner_1, vector_to_corner_2, ... ],
                | [(vector_to_center_arc1, (start_angle_arc1, end_angle_arc1), radius_arc1 ),
                | (vector_to_center_arc2, (start_angle_arc2, end_angle_arc2), radius_arc2 ),
                | ... ],
                | [(vector_to_center_ellipse1, (start_angle, end_angle), (major_radius, minor_radius),major_axis_vector),
                | (vector_to_center_ellipse2, (start_angle, end_angle), (major_radius, minor_radius),major_axis_vector),
                | ...],
                | )

                each vector is a tuple of float (vx, vy).
        """

        if region_obj.ObjectName != 'AcDbRegion':
            raise TypeError("input drawing entity must be <'AcDbRegion'>")

        # get centroid of region as base point of vector
        cp_x, cp_y = region_obj.Centroid

        # explode region into drawing elements
        elem = region_obj.Explode()

        # record the boundary vector
        bv = []
        bv_arc = []
        bv_ell = []
        for item in elem:
            if item.ObjectName == 'AcDbLine':
                # record ends vector of line
                ev_1 = (item.StartPoint[0] - cp_x, item.StartPoint[1] - cp_y)
                ev_2 = (item.EndPoint[0] - cp_x, item.EndPoint[1] - cp_y)
                if ev_1 not in bv:
                    bv.append(ev_1)
                if ev_2 not in bv:
                    bv.append(ev_2)
                # delete recorded item
                item.Delete()
            elif item.ObjectName == 'AcDbArc':
                # record ends vectors of arc
                ev_1 = (item.StartPoint[0] - cp_x, item.StartPoint[1] - cp_y)
                ev_2 = (item.EndPoint[0] - cp_x, item.EndPoint[1] - cp_y)
                if ev_1 not in bv:
                    bv.append(ev_1)
                if ev_2 not in bv:
                    bv.append(ev_2)
                # record center vectors, angel range and radius of arc
                cv = (item.Center[0] - cp_x, item.Center[1] - cp_y)
                radi = item.Radius
                sa = item.StartAngle
                ea = item.EndAngle if item.EndAngle >= sa else item.EndAngle + pi * 2  # make end angle always larger
                bv_arc.append((cv, (sa, ea), radi))
                # delete recorded item
                item.Delete()

            elif item.ObjectName == 'AcDbEllipse':
                # record ends vectors of ellipse
                ev_1 = (item.StartPoint[0] - cp_x, item.StartPoint[1] - cp_y)
                ev_2 = (item.EndPoint[0] - cp_x, item.EndPoint[1] - cp_y)
                if ev_1 not in bv:
                    bv.append(ev_1)
                if ev_2 not in bv:
                    bv.append(ev_2)
                # record center vectors, angel range, major/minor radius and major axis direction of ellipse
                cv = (item.Center[0] - cp_x, item.Center[1] - cp_y)
                maj_radi = item.MajorRadius
                min_radi = item.MinorRadius
                maj_axis = item.MajorAxis
                threshold = round(self.vangle(item.MinorAxis, rad=False) - self.vangle(item.MajorAxis, rad=False))
                if threshold == 90 or threshold == -270:  # ellipse anticlockwise
                    sa = (item.StartAngle + self.vangle(item.MajorAxis)) % (2*pi)
                    _ea = (item.EndAngle + self.vangle(item.MajorAxis)) % (2*pi)
                    ea = _ea if _ea >= sa else _ea + pi * 2  # make end angle always larger
                elif threshold == -90 or threshold == 270:  # ellipse clockwise
                    sa = (2*pi - item.EndAngle + self.vangle(item.MajorAxis)) % (2*pi)
                    _ea = (2*pi - item.StartAngle + self.vangle(item.MajorAxis)) % (2*pi)
                    ea = _ea if _ea >= sa else _ea + pi * 2  # make end angle always larger
                else:
                    raise ValueError('Invalid Principal Axis.')
                bv_ell.append((cv, (sa, ea), (maj_radi, min_radi), maj_axis[:2]))
                # delete recorded item
                item.Delete()

            elif item.ObjectName == 'AcDbSpline':
                # get ends and intermediate vectors of spline
                ctrl_pnt = item.ControlPoints
                curv = BSpline.Curve()  # create geomdl.SPline instance
                curv.degree = item.Degree
                curv.ctrlpts = [ctrl_pnt[i:i + 3] for i in range(0, len(ctrl_pnt), 3)]
                curv.knotvector = item.Knots
                curv.delta = 1 / spl_sub  # delta of interpolate in domain [0,1]
                bv.extend(list(map(lambda p: (p[0] - cp_x, p[1] - cp_y), curv.evalpts)))
                # delete recorded item
                item.Delete()

        if file_name:   # export to file
            if file_name[-3:] == 'csv':
                with open(file_name, "w") as f:
                    csvwriter = csv.writer(f, lineterminator='\n')
                    for row in bv:
                        csvwriter.writerow(row + (0, 0, 0, 0, 0, 0))
                    for row in bv_arc:
                        cv, se, r = row
                        csvwriter.writerow(cv + se + tuple([r, 0]) + (0, 0))
                    for row in bv_ell:
                        cv, se, rs, rv = row
                        csvwriter.writerow(cv + se + rs + rv)

            elif file_name[-4:] == 'json':  # save as json file
                with open(file_name, 'w') as f:
                    json.dump({'node': bv, 'arc': bv_arc, 'ellipse': bv_ell}, f)

        return bv, bv_arc, bv_ell

    @classmethod
    def boundalong(cls, boundary_nodes, boundary_arc, boundary_ellipse, direction_vector, *direction_vectors):
        """Measure the distance from centroid of section to its boundary in specified direction.

        :param boundary_nodes: list of vector from centroid to boundary corner.
                               each vector is a tuple of float, (vx, vy).
        :param boundary_arc: list of tuple states geometrical information of boundary arc, in the form of:

                            | [(vector_to_center_arc1, (start_angle_arc1, end_angle_arc1), radius_arc1 ),
                            | (vector_to_center_arc2, (start_angle_arc2, end_angle_arc2), radius_arc2 ), ... ]
        :param boundary_ellipse: list of tuple states geometrical information of boundary ellipse, in the form of:

              | [(vector_to_center_ellipse1, (start_angle, end_angle), (major_radius, minor_radius),major_axis_vector),
              | (vector_to_center_ellipse2, (start_angle, end_angle), (major_radius, minor_radius),major_axis_vector),
              | ...]

        :param direction_vector: array-like, vector indicating the direction of measurement.
        :return: tuple of float, (max. negative distance, max. positive distance). Here negative distance is measured
                along the opposite direction.
        """
        dv = np.asarray([direction_vector]+list(direction_vectors))[:, :2]  # [(x1,y1),(x2,y2),...]
        dv = dv / np.linalg.norm(dv, axis=1).reshape((-1, 1))  # normalized
        # TODO: do not flatten, recorde the distance by group of different direction
        dist = (np.array(boundary_nodes) @ dv.T).flatten().tolist()  # distance from nodes to centroid in measuring direction

        beta = np.array([cls.vangle(d) for d in dv])  # angle between measuring direction and global +x
        for cv, (angle_start, angle_end), r in boundary_arc:
            dist.extend((dv[np.logical_or((angle_start < beta) & (beta < angle_end),
                                          (angle_start < beta+2*pi) & (beta+2*pi < angle_end))] @ cv + r).tolist())
            dist.extend((dv[np.logical_or((angle_start < beta-pi) & (beta-pi < angle_end),
                                          (angle_start < beta+pi) & (beta+pi < angle_end))] @ cv - r).tolist())

        for cv, (angle_start, angle_end), (r1, r2), mav in boundary_ellipse:
            mav_norm = mav / np.linalg.norm(mav)  # normalize the vector
            eigv = np.column_stack([mav_norm, np.array([[0, -1], [1, 0]]) @ mav_norm])  # eigenvector of tilted ellipse
            lam = np.array([[1/r1**2, 0], [0, 1/r2**2]])  # eigenvalue of tilted ellipse
            elp = eigv @ lam @ eigv.T  # matrix of tilted ellipse
            # vector from ellipse center to the furthest point on ellipse
            vp = dv @ eigv @ np.array([[0, 1], [-(r2/r1)**2, 0]]) @ eigv.T @ np.array([[0, 1], [-1, 0]])
            vp = vp / np.linalg.norm(vp, axis=1).reshape((-1, 1))  # normalized
            # vps = np.vstack([vp, -vp])  # incl. opposite direction
            # beta = np.array([cls.vangle(p) for p in vps])
            # nvp = vps[np.logical_or((angle_start < beta) & (beta < angle_end),
            #                         (angle_start < beta+2*pi) & (beta+2*pi < angle_end))]
            # rv= nvp / np.sqrt(np.diag(nvp @ elp @ nvp.T)).reshape(-1, 1)

            for i, nvp in enumerate(np.vstack([vp, -vp])):  # incl. opposite direction
                beta = cls.vangle(nvp)
                if angle_start < beta < angle_end or angle_start < beta + 2 * pi < angle_end:
                    rv = nvp / np.sqrt(nvp @ elp @ nvp)  # radius vector to farest point
                    # print((np.array(cv) + rv) @ dv[i % len(dv)])
                    dist.append((np.array(cv) + rv) @ dv[i % len(dv)])

        return min(dist), max(dist)

    def dimregion(self, region_obj, direction_vector=(1, 0, 0), spl_sub=10, dim_offset=-10, outside=True,
                  dimstyle=None):
        """Add dimension annotation marking overall size of a region along specified direction.

        :param region_obj: Region object.
        :param direction_vector: array-like, vector indicating the direction of measurement.
        :param spl_sub: int, numbers of subdivided segments of a spline boundary.
        :param dim_offset: float, offset distance of dimension annotation from reference point. reference point is the
                            most left point on measured object When *'outside'* = False, and is the most outside point
                            on boundary when *'outside'* = True.
        :param outside: bool, always put the dimension annotation outside of measured object.
        :param dimstyle: str, name of dimension style. If not given, current active style will be used.
        :return: created dimension object.
        """

        dv = np.asarray(direction_vector[:2])  # (vx, vy)
        dv = dv / np.linalg.norm(dv)  # unitized
        bv, bv_arc, bv_ell = self.getboundary(region_obj=region_obj, spl_sub=spl_sub)
        dist = np.array(bv) @ dv  # distance from each recorded node to centroid

        cp = np.array(region_obj.Centroid)
        pt_min = cp + bv[np.argmin(dist)]  # dim point 1#
        pt_max = cp + bv[np.argmax(dist)]  # dim point 2#

        beta = self.vangle(dv)  # angle between measuring direction and global +x
        for cv, (angle_start, angle_end), r in bv_arc:
            if angle_start < beta < angle_end or angle_start < beta + 2 * pi < angle_end:
                if np.array(cv) @ dv + r > max(dist):
                    pt_max = cp + cv + r * dv  # overwrite dim point 2#
            if angle_start < beta - pi < angle_end or angle_start < beta + pi < angle_end:
                if np.array(cv) @ dv - r < min(dist):
                    pt_min = cp + cv - r * dv  # overwrite dim point 1#

        for cv, (angle_start, angle_end), (r1, r2), mav in bv_ell:
            mav_norm = mav / np.linalg.norm(mav)
            eigv = np.column_stack([mav_norm, np.array([[0, -1], [1, 0]]) @ mav_norm])
            lam = np.array([[1/r1**2, 0], [0, 1/r2**2]])
            elp = eigv @ lam @ eigv.T  # matrix of tilted ellipse
            vp = eigv @ np.array([[0, 1], [-(r2/r1)**2, 0]]) @ eigv.T @ np.array([[0, 1], [-1, 0]]) @ dv
            for nvp in [vp / np.linalg.norm(vp), -vp / np.linalg.norm(vp)]:  # normalized, incl. opposite direction
                beta = self.vangle(nvp)
                if angle_start < beta < angle_end or angle_start < beta + 2 * pi < angle_end:
                    rv = nvp / np.sqrt(nvp @ elp @ nvp)
                    if (np.array(cv) + rv) @ dv > max(dist):
                        pt_max = cp + cv + rv  # overwrite dim point 2#
                    elif (np.array(cv) + rv) @ dv < min(dist):
                        pt_min = cp + cv + rv  # overwrite dim point 1#

        hv = np.array(
            [[0, -1], [1, 0]]) @ dv  # h-direction: perpendicular with measuring direction, 90deg anticlockwise
        # todo: update application of .boundalong
        bottom, top = self.boundalong(bv, bv_arc, bv_ell, hv)  # boundary along h-direction
        h = (pt_min - cp) @ hv  # distance from centroid to pt_min
        to_top = top - h  # distance from pt_min to top boundary
        to_bottom = bottom - h  # distance from pt_min to bottom boundary

        # calculate the location of dimension annotation
        if outside:
            dloc = to_top + dim_offset if dim_offset >= 0 else to_bottom + dim_offset
        else:
            dloc = dim_offset

        # dloc or -dloc is depends on if direction vector is towards +x
        if direction_vector[0] < -1e-8:
            dloc *= -1

        # print(pt_min)
        # print(to_top, to_bottom)
        # print(dloc)
        # use rotate dimension method
        return self.adddim(np.append(pt_min, 0), np.append(pt_max, 0), offset=dloc,
                           measure_dir=direction_vector, dimstyle=dimstyle)

    def seclib(self, file_name, sort=1, sec_name=None, spl_sub=100, update=True):
        """Select a group of regions and output their object ID, section properties and boundary information to .csv
        or .json file.

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
        """

        regions = self.select(objtype='region')

        if not regions:
            raise RuntimeError("No region has been selected.")

        if sort == 1:  # sort selected table from left to right, then upper to lower
            def sf(x):
                p = getattr(x, "Centroid")
                return [-p[1], p[0]]

            regions.sort(key=sf)
        elif sort == 2:  # sort selected table from upper to lower, then left to right
            def sf(x):
                p = getattr(x, "Centroid")
                return [p[0], -p[1]]

            regions.sort(key=sf)

        # load existing data or create a empty one for record
        if update:
            try:
                if file_name[-3:] == 'csv':
                    rec = pd.read_csv(file_name)
                elif file_name[-4:] == 'json':
                    rec = pd.read_json(file_name)
                else:
                    raise OSError("Unsupported export file type.")
            except (FileNotFoundError, ValueError):
                print(f"<{file_name}> is not existing. Creating file.")
                rec = pd.DataFrame([], index=['id', 'A', 'Ix', 'Iy', 'Ixy', 'Zx', 'Zy', 'Sx', 'Sy',
                                              'I1', 'I2', 'alpha', 'Z1', 'Z2', 'S1', 'S2', 'bnode', 'barc', 'belp'])
        else:
            rec = pd.DataFrame([], index=['id', 'A', 'Ix', 'Iy', 'Ixy', 'Zx', 'Zy', 'Sx', 'Sy',
                                          'I1', 'I2', 'alpha', 'Z1', 'Z2', 'S1', 'S2', 'bnode', 'barc', 'belp'])

        # define default section name
        if not sec_name or len(sec_name) != len(regions):
            sec_name = [f"Section_{n + 1:0>2}" for n in range(len(regions))]

        # record section information of each region
        for reg in regions:
            sec_info = self.getsecprop(reg)
            sec_info['bnode'], sec_info['barc'], sec_info['belp'] = self.getboundary(reg, spl_sub)
            sec_info['id'] = reg.ObjectID
            rec[sec_name.pop(0)] = pd.Series(sec_info)

        if file_name[-3:] == 'csv':
            rec.to_csv(file_name)
            print(f"Data of {len(regions)} Sections have been exported to <{file_name}>")
        elif file_name[-4:] == 'json':
            rec.to_json(file_name)
            print(f"Data of {len(regions)} Sections have been exported to <{file_name}>")
        else:
            raise OSError("Unsupported export file type.")

    def lbelem(self, spl_sub=10, dim_offset=-10, dimstyle=None, num_block="NUM.dwg", num_scale=1.0, file_name=None,
               update=True):
        """Mark out elements on selected regions and get elements geometry information.

        This is a quick tool for primary task producing structural data used by
        ``pyfacade.pymcad.Xmcd.addblock_lbcheck``

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
        """

        def clean_before_quit(clean_demarc=True):
            for bl in bls:
                try:
                    bl.Delete()
                except pywintypes.com_error:
                    continue
            try:
                elem.Delete()
            except pywintypes.com_error:
                pass

            if clean_demarc:
                for d in demarc:
                    try:
                        d.Delete()
                    except pywintypes.com_error:
                        continue

        # load existing data or create a empty one for record
        if update and file_name:
            try:
                with open('elems.json') as f:
                    rec = json.load(f)
            except (FileNotFoundError, ValueError):
                print(f"<{file_name}> is not existing. Creating from empty file.")
                rec = {}
        else:
            rec = {}

        while True:  # selecting section to mark
            sec = self.pick(objtype='region', prompt='Select a region of section:')
            if not sec:
                break  # final ending

            sec_id = sec.ObjectID  # object ID of entire section
            sec_cx, sec_cy = sec.Centroid  # centroid of selected section
            num = 1  # counting element
            elem_prop = {}  # recording dict

            while True:  # marking elements until canceled by user
                pt1 = self.get(prompt="specify the first corner of mark-out box, or mark by [Ployline]:", bits=1,
                               keyword="p")
                if not pt1:
                    break  # quit point 1

                if pt1 == 'p':  # drawing polyline box
                    pls = self.freedraw()
                    try:
                        box = self.makeregion(pls)[0]
                    except RuntimeError:
                        for pl in pls:
                            pl.Delete()
                        self.prompt("Fail to mark out element on section, please retry...\n")
                        continue

                else:  # draw rectangle box
                    pt2 = self.get(intype='corner', ref_pnt=pt1, prompt="specify the second corner of mark-out box:")
                    if not pt2:
                        break  # quit point 2
                    box = self.makeregion([self.addrect(pt1, pt2)])[0]

                bls = box.Explode()
                elem = sec.Copy()
                elem.Boolean(constants.acIntersection, box)
                if not elem.Area:
                    clean_before_quit(clean_demarc=False)
                    self.prompt("Fail to mark out element on section, please retry...\n")
                    continue

                # Validate element
                invalid_elem = False
                thks = []
                refpt = None
                slope = None
                demarc = []
                for bl in bls:
                    intpt = elem.IntersectWith(bl, constants.acExtendNone)
                    if intpt:  # when intersect exists
                        if len(intpt) == 6:
                            if not self.tolerate(intpt[:3], intpt[3:]):  # has two different intersect
                                n1, n2 = np.array(intpt[:3]), np.array(intpt[3:])
                                demarc.append(self.addline(n1, n2, color='red')[0])  # draw demarcating line
                                thks.append(np.linalg.norm(n2 - n1))  # record the intersect width
                                if refpt is None:  # define the reference point if not yet defined
                                    refpt = (n1 + n2) / 2
                                if not slope:  # define the slope of element if not yet defined
                                    slope = self.vangle(n2 - n1) - pi * 0.5
                        else:  # intersects more than two
                            invalid_elem = True
                            break

                if len(thks) != 1 and len(thks) != 2:  # no end or more than 2 ends
                    invalid_elem = True

                if invalid_elem:
                    clean_before_quit()
                    self.prompt("Invalid element definition, please retry...\n")
                    continue

                # mark out the valid element
                self.fillhatch([[elem]], scale=0.5, color='gray')  # fill hatch
                self.dimregion(elem, (cos(slope), sin(slope), 0), spl_sub=spl_sub, dim_offset=dim_offset,
                               outside=True,
                               dimstyle=dimstyle)  # mark dimension 'b'
                self.adddim(demarc[0].StartPoint, demarc[0].EndPoint, offset=dim_offset,
                            dimstyle=dimstyle)  # mark dimension 't'

                # read element information
                (x_min, y_min, z_min), (x_max, y_max, z_max) = elem.GetBoundingBox()
                bound_x = (x_min - sec_cx, x_max - sec_cx)  # element boundary along x
                bound_y = (y_min - sec_cy, y_max - sec_cy)  # element boundary along y
                elem_cx, elem_cy = elem.Centroid

                elem.Rotate(self.point(*refpt), -slope)  # rotate element to align with x-y axis
                min_c, max_c = elem.GetBoundingBox()
                length = max_c[0] - min_c[0]  # length of element
                Ae = elem.Area
                cpe = elem.Centroid
                Ix_, Iy_ = elem.MomentOfInertia
                Ie = Ix_ + Ae * ((cpe[1] - refpt[1]) ** 2 - (cpe[1]) ** 2)  # moment of inertia about longitudinal side
                elem.Rotate(self.point(*refpt), slope)  # rotate back

                clean_before_quit(clean_demarc=False)

                # add numbering mark on element
                if num_block:
                    nmpt = self.get(prompt="\nSpecify the location for number mark")
                    if nmpt:
                        lead_v = np.array([elem_cx, elem_cy, 0]) - np.array(
                            nmpt)  # lead vector point to centroid of element
                        atrr_length = np.linalg.norm(lead_v)
                        atrr_angle = self.vangle(lead_v)
                        self.insertblock(nmpt, block_name="NUM.dwg", scale=(num_scale, num_scale, num_scale),
                                         dynamic_prop={7: atrr_length, 8: atrr_angle}, attr={'N': num})

                # specify element type
                if len(thks) == 1:  # outstanding element
                    kw = "G U R"
                    prom = "Specify element type: [stress Gradient/Uniform compression/Reinforced uniform compression]"
                else:  # internal element
                    kw = "X Y U"
                    prom = "Specify element type: [stress gradient in X-aix/stress gradient in Y-aix/Uniform compression]"
                tp = self.kbinput(intype="keyword", prompt=prom, bits=0, keyword=kw, space=False, default='U')
                if not tp:
                    tp = "U"  # use conservative default value when input is cancelled by user

                # record element information
                elem_prop[str(num)] = {'length': length, 'thks': thks, 'rp': list(refpt), 'slope': slope,
                                       'bx': bound_x, 'by': bound_y, 'Ie': Ie, 'type': tp}
                num += 1

            rec[str(sec_id)] = elem_prop

        if file_name:  # save as json file
            with open(file_name, 'w') as f:
                json.dump(rec, f)
            print(f"Element information of {len(rec)} Sections have been exported to <{file_name}>")

        return rec

    def search(self, criteria=None):
        """Search objects from drawing according to specified criteria.

        :param criteria: one-argument function which receives drawing entities and return a boolean.
        :return: list of found objects.
        """

        srange = self.__dwg.ModelSpace  # search the entire drawing

        if not criteria:
            criteria = lambda x: True  # always True if no filtering criteria

        find = []
        for i in range(srange.Count):
            obj = self.__dwg.ObjectIdToObject(srange.Item(i).ObjectID)
            try:
                if criteria(obj):
                    find.append(obj)  # record object if fulfill the criteria
            except AttributeError:
                continue  # skip the object if it can't be checked by given criteria

        return find

    def replaceblock(self, block_name, replac_by, offset=(0., 0., 0.), new_scale=None, new_rotation=None,
                     new_dynamic_prop=None, new_attr=None, offset_in_scale=True, inherit=True):
        """Replace blocks with specified name by another block.

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
        """

        blks = self.search(lambda x: x.ObjectName == 'AcDbBlockReference' and x.EffectiveName == block_name)

        for b in blks:
            b_scale = (b.XScaleFactor, b.YScaleFactor, b.ZScaleFactor) if not new_scale else new_scale
            b_rotation = b.Rotation if new_rotation is None else new_rotation

            # ToDo: partially inherit
            if inherit and (new_dynamic_prop is None) and b.IsDynamicBlock:
                dp = b.GetDynamicBlockProperties()
                # inherit from original block only when properties are not read-only
                b_dynamic_prop = {p.PropertyName: p.Value for p in dp if not p.ReadOnly}
            else:
                b_dynamic_prop = new_dynamic_prop  # use new properties

            if inherit and (new_attr is None) and b.HasAttributes:
                b_attr = {a.TagString: a.TextString for a in b.GetAttributes()}  # inherit from original
            else:
                b_attr = new_attr  # use new attributes

            # scale offset
            b_offset = np.array(offset)*np.array(b_scale) if offset_in_scale else np.array(offset)

            # transform offset into local coordinates system
            b_offset = b_offset if b_rotation == 0 else self.rotatevec(b_offset, b_rotation)

            new_insert_pnt = np.array(b.InsertionPoint) + b_offset

            b.Delete()

            self.insertblock(insert_pnt=new_insert_pnt, block_name=replac_by, scale=b_scale,
                             rotation=b_rotation, dynamic_prop=b_dynamic_prop, attr=b_attr)

    @classmethod
    def multitry(cls, limit):
        """Decorator make function be called again when being rejected by application

        :param limit: int, attempting times before raising except.
        :return: decorated function.
        """
        def decf(func):
            @wraps(func)
            def wrapper(self, *args, i=limit, **kwargs):
                if i > 0:
                    try:
                        return func(self, *args, **kwargs)
                    except pywintypes.com_error as er:
                        if er.args[0] == -2147418111:  # Call was rejected by callee
                            time.sleep(0.2)
                            self.prompt("Problem occurs. RETRY...")
                            time.sleep(0.2)
                            return wrapper(self, i - 1)
                        else:
                            raise
                else:
                    return func(self)
            return wrapper
        return decf

    # endregion


# Subclass for frame analysis
class CADFrame(Acad):
    """Subclass of ``pyacad.Acad``, extends for interactively acquiring structural information of 2D frame.

    Acquired data is used to build up analysis model by Module ``pyfacade.pyeng``.

    :param file_name: str, file name (and path) to open with application. Activate the specified file
            if it is opened already. Otherwise, try to open the file from specified path. When successfully opened,
            prompts will be shown in the command line of AutoCAD, asking for selecting and specifying the basic frame.
    :param geoacc: int. number of decimal place to be kept when outputting the nodes' coordinates.
    """

    def __init__(self, file_name=None, geoacc=4):

        super().__init__(file_name)  # initialize the parent class
        self.__dwg = super()._doc

        _ls = super().select(objtype="AcDbLine", prompt="Select lines as frame members:")
        if not _ls:  # no valid selection
            raise TypeError("No line object has been selected.")
        _lid = [x.ObjectID for x in _ls]  # object ID of lines
        # end point of lines, round
        _endn = [(tuple(np.round(line.StartPoint, geoacc)), tuple(np.round(line.EndPoint, geoacc))) for line in _ls]
        self.__nodes = list(set(tuple(x) for be in _endn for x in be))  # record the node and remove duplicated.
        self.__nodes.sort()  # sort node

        # record boundaries
        nx, ny, _ = zip(*self.__nodes)
        self.__bx = (min(nx), max(nx))
        self.__by = (min(ny), max(ny))

        _beams = [[self.__nodes.index(i[0]), self.__nodes.index(i[1])] for i in _endn]  # beam set list

        # sort beam set, line objects and line id in same order
        _bs = list(zip(_beams, _ls, _lid))  # temporarily grouping
        _bs.sort(key=lambda x: x[0])  # sort according to beam set
        self.__beams, self.__bfd, self.__lid = zip(*_bs)

        # relative nodes coordinates for output.
        _origin = super().get(intype="point",
                              prompt="Specify the origin point. Default is the most left-bottom node of selected lines:",
                              default=self.__nodes[0])
        if not _origin:  # Null input
            raise KeyboardInterrupt("Process terminated by user.")
        self.__nd = [tuple(c[:2]) for c in np.round(np.array(self.__nodes) - _origin, geoacc)]

        self.__PMode = self.__dwg.GetVariable("PDMODE")  # record the current setting.

    @property
    def nodes(self):
        """List of nodes' coordinates. Read-only."""
        return self.__nd

    @property
    def beams(self):
        """List of beam set, indicating index pair of start and end nodes of each beam. Read-only."""
        return self.__beams

    def __cleanpt(self, points):  # cleaning procedure for internal use only.
        for p in points:
            p.Delete()
        self.__dwg.SetVariable("PDMODE", self.__PMode)

    def node_num(self, rng=None, color=None):
        """Show number of nodes on drawing.

        :param rng: list on int, indices of nodes require showing number. If not give, show numbers of all nodes.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :return: list of added text objects.
        """
        if not rng:
            rng = range(len(self.__nodes))  # show all the nodes
        text_height = round(0.02*max(self.__bx[1]-self.__bx[0], self.__by[1]-self.__by[0]))
        return [self.addtext(i, self.__nodes[i], height=text_height, color=color) for i in rng]

    def beam_num(self, rng=None, color=None):
        """Show number of beams on drawing.

        :param rng: list on int, indices of beams require showing number. If not give, show numbers of all beams.
        :param color: str as color name, or list of int [r,g,b] as RGB value. If not given, use current active color.
        :return: list of added text objects.
        """
        if not rng:
            rng = range(len(self.__bfd))  # show all the nodes

        text_height = round(0.02 * max(self.__bx[1] - self.__bx[0], self.__by[1] - self.__by[0]))
        return [self.addtext(i, (np.array(self.__bfd[i].StartPoint) + np.array(self.__bfd[i].EndPoint)) * 0.5,
                             height=text_height, color=color)
                for i in rng]

    @Acad.multitry(5)
    def set_restrain(self):
        """Specify restrain conditions on node from drawing.

        :return: dict. Definition of structural restrain in the form of
                {node_no:[res_condition_x, res_condition_y, res_condition_rotate], ...}, where:
                0=released and 1=restrained.
        """
        _pt = self.addnode(*self.__nodes)  # add node mark on drawing
        _ptid = [p.ObjectID for p in _pt]  # record point ID
        self.__dwg.SetVariable("PDMODE", 35)  # Show node in style 35

        # Select node to apply restrain
        respt = self.select(objtype='AcDbPoint', prompt="Select Point to apply restrain on:\n")
        if not respt:
            raise ValueError("No point has been selected.")

        # Create restrain dict
        restr = {}
        for p in respt:
            p.Highlight(True)
            nid = _ptid.index(p.ObjectID)
            num = self.node_num(rng=[nid], color='red')[0]
            prompt1 = f"Define restrain condition of for node {nid} "
            prompt2 = "in the form of Rx,Ry,Mz. <1,1,1>"
            prompt3 = "(1=restrained, 0=release): \n"
            prompt4 = "Invalid input, please define again "
            resin = self.kbinput(intype='string', prompt=prompt1 + prompt2 + prompt3, default='1,1,1')
            if not resin:  # input is canceled
                num.Delete()
                self.__cleanpt(_pt)
                raise KeyboardInterrupt("Process terminated by user.")

            while not re.match(r"^[01],[01],[01]$", resin):
                resin = self.kbinput(intype='string', prompt=prompt4 + prompt2 + prompt3, default='1,1,1')
                if not resin:  # input is canceled
                    self.__cleanpt(_pt)
                    raise KeyboardInterrupt("Process terminated by user.")

            restr[nid] = [int(x) for x in resin.split(',')]
            num.Delete()
            p.Highlight(False)

        self.__cleanpt(_pt)


        return restr

    @Acad.multitry(5)
    def set_release(self):
        """Specify end release of beams from drawing.

        :return: dict. Definition of release conditions of beams in the form of
                {beam_no:[(axial_condition_start, rotation_condition_start), (axial_condition_end, rotation_condition_end)],...},
                where: 0=fixed and 1=released.
        """

        # Select beam to apply end release on
        relsbm = self.select(objtype='AcDbLine', prompt="Select lines to apply end release on:\n")
        if not relsbm:
            raise ValueError("No line has been selected.")

        # Create release dict
        brels = {}
        for b in relsbm:  # for each selected beam
            b.Highlight(True)
            bid = self.__lid.index(b.ObjectID)
            num = self.beam_num(rng=[bid], color='green')[0]
            _pt = self.addnode(b.StartPoint, b.EndPoint)  # add node mark at beam ends
            _ptid = [p.ObjectID for p in _pt]  # record point ID
            self.__dwg.SetVariable("PDMODE", 35)  # Show node in style 35

            # Select node to apply release on
            relsbe = self.select(objtype='AcDbPoint', prompt="Select End to Apply Release:\n")
            if not relsbe:
                num.Delete()
                self.__cleanpt(_pt)
                b.Highlight(False)
                raise ValueError("No end point has been selected.")

            rels = [[0, 0], [0, 0]]
            for e in relsbe:
                e.Highlight(True)
                # Set translate release
                rels_t = self.kbinput(intype="keyword",
                                      prompt="Release axial translate at this end? [Yes/No] <Yes>. ",
                                      keyword="Yes No", default="Yes")
                if rels_t:
                    rels[_ptid.index(e.ObjectID)][0] = 1 if rels_t == "Yes" else 0
                else:
                    num.Delete()
                    self.__cleanpt(_pt)
                    b.Highlight(False)
                    raise KeyboardInterrupt("Process terminated by user.")
                # Check the availability
                while self.isbusy():
                    time.sleep(0.2)
                # Set rotation release
                rels_r = self.kbinput(intype="keyword",
                                      prompt="Release rotation at this end? [Yes/No] <Yes>  ",
                                      keyword="Yes No", default="Yes")
                if rels_r:
                    rels[_ptid.index(e.ObjectID)][1] = 1 if rels_r == "Yes" else 0
                else:
                    num.Delete()
                    self.__cleanpt(_pt)
                    b.Highlight(False)
                    raise KeyboardInterrupt("Process terminated by user.")
                e.Highlight(False)

            brels[bid] = [tuple(x) for x in rels]
            self.__cleanpt(_pt)
            num.Delete()
            b.Highlight(False)

        return brels

    @Acad.multitry(5)
    def set_udl(self):
        """Specify uniform distributed load on beams from drawing.

        :return: dict. Definition of applied *Uniformly Distributed Load* on model in the form of
                {beam_no:(axial_force, transverse_force),...}. Unit=N/mm.
        """
        # Select beam to apply load on
        loadbm = self.select(objtype='AcDbLine', prompt="Select lines to apply uniform distributed load on:\n")
        if not loadbm:
            raise ValueError("No line has been selected.")

        # Create dict
        Q = {}
        for b in loadbm:
            bid = self.__lid.index(b.ObjectID)
            lm = (np.array(b.StartPoint) + np.array(b.EndPoint)) * 0.5  # mid point
            lu = (lm + np.array(b.EndPoint)) * 0.5  # 3/4 point
            lv = self.rotatevec(lu - lm, pi * 0.5) + lm  # transverse point
            # Define axial force
            axis_mark = self.addleader(lu, lm, ltype="CONTINUOUS", color="red", arrow=0, headsize=0.05 * b.Length)
            _ual = self.kbinput(intype='real',
                                prompt=f"Define uniform longitudinal load on beam {bid}, Unit=N/mm. <0> ",
                                default=0.0)
            axis_mark.Delete()
            if _ual is None:
                raise KeyboardInterrupt("Process terminated by user.")
            # Check the availability
            while self.isbusy():
                time.sleep(0.2)
            # Define transverse force
            axis_mark = self.addleader(lv, lm, ltype="CONTINUOUS", color="red", arrow=0, headsize=0.05 * b.Length)
            _udl = self.kbinput(intype='real',
                                prompt=f"Define uniform transverse load on beam {bid}, Unit=N/mm. <0> ",
                                default=0.0)
            axis_mark.Delete()
            if _udl is None:
                raise KeyboardInterrupt("Process terminated by user.")
            Q[bid] = (_ual, _udl)

        return Q

    @Acad.multitry(5)
    def set_pointload(self):
        """Specify point load on nodes from drawing.

        :return: dict. Definition of applied *Concentrated Load* on model in the form of
                {node:[Fx, Fy, Mz]...}. Unit=N or N*mm.
        """
        _pt = self.addnode(*self.__nodes)  # add node mark on drawing
        _ptid = [p.ObjectID for p in _pt]  # record point ID
        self.__dwg.SetVariable("PDMODE", 35)  # Show node in style 35

        # Select node to apply node force on
        loadpt = self.select(objtype='AcDbPoint', prompt="Select Point to apply concentrated load on:\n")
        if not loadpt:
            self.__cleanpt(_pt)
            raise ValueError("No point has been selected.")

        # Create node force dict
        F = {}
        for p in loadpt:
            p.Highlight(True)
            nid = _ptid.index(p.ObjectID)
            prompt1 = f"Define point load on node <{nid}> "
            prompt2 = "in form of Fx,Fy,Mz "
            prompt3 = "Unit=N and N.mm. <0,0,0>\n"
            prompt4 = "Invalid input, please define again "
            resin = self.kbinput(intype='string', prompt=prompt1 + prompt2 + prompt3, default='0.0,0.0,0.0')
            if not resin:  # input is canceled
                self.__cleanpt(_pt)
                raise KeyboardInterrupt("Process terminated by user.")

            while not re.match(r"^[+-]?\d+(\.\d+)?,[+-]?\d+(\.\d+)?,[+-]?\d+(\.\d+)?$", resin):
                resin = self.kbinput(intype='string', prompt=prompt4 + prompt2 + prompt3, default='0.0,0.0,0.0')
                if not resin:  # input is canceled
                    self.__cleanpt(_pt)
                    raise KeyboardInterrupt("Process terminated by user.")

            F[nid] = [float(x) for x in resin.split(',')]
            p.Highlight(False)

        self.__cleanpt(_pt)
        return F

    # Internal for method set_E, set_A and set_I
    def __setprop(self, prop_type, lib=None):

        props = {"E": "Modulus of Elasticity",
                 "A": "Section Area",
                 "I": "Moment of Inertia"}
        units = {"E": ", Unit=N/mm^2",
                 "A": ", Unit=mm^2",
                 "I": ", Unit=mm^4"}

        num = self.beam_num(color='green')

        res = []
        if lib:  # when library is provided, require input a index as key in dict
            lib = {str(x): y for x, y in lib.items()}  # change key to str
            _default = list(lib.keys())[0]  # initialize default value = first key
            intype = "keyword"
            kw = ' '.join(list(lib.keys()))  # record dict key as keyword
            pt = "INDEX"
            unit = ""
            bits = 0
        else:  # when library is not provided, require input a real value
            _default = 1.0  # initialize default value
            intype = "real"
            kw = ""
            pt = "VALUE"
            unit = units[prop_type]
            bits = 6  # disallow 0 and negative value

        for i in range(len(self.__beams)):
            prompt1 = f"Assign {pt} of {props[prop_type]} for Beam {i}, or for all the [Rest]{unit} <{_default}>:"
            prompt2 = f"Assign {pt} of {props[prop_type]} for all the rest beams{unit} <{_default}>:"
            keyin = self.kbinput(intype=intype, prompt=prompt1, default=_default, keyword=kw + " Rest", bits=bits)
            if keyin is None:
                for n in num:
                    n.Delete()
                raise KeyboardInterrupt("Process terminated by user.")
            if type(keyin) == str:
                if keyin == 'Rest':  # keyword
                    _all = self.kbinput(intype=intype, prompt=prompt2, default=_default, keyword=kw, bits=bits)
                    for n in num:
                        n.Delete()
                    if _all is None:
                        raise KeyboardInterrupt("Process terminated by user.")
                    return res + [lib[_all] if lib else _all] * (len(self.__beams) - i)
                else:  # index
                    res.append(lib[keyin])
                    _default = keyin
            else:
                res.append(keyin)
                _default = keyin

        for n in num:
            n.Delete()
        return res

    @Acad.multitry(5)
    def set_E(self, lib=None):
        """Assign modulus of elasticity for each beams.

        :param lib: dict of modulus of elasticity, unit = N/mm :superscript:`2`. If provided, application will request
                    specifying key of dict for each beam. Otherwise, non-zero positive value will be requested.
        :return: list. Assigned *Modulus of Elasticity* of beams, unit = N/mm :superscript:`2`.
        """
        return self.__setprop("E", lib)

    @Acad.multitry(5)
    def set_A(self, lib=None):
        """Assign section area for each beams.

        :param lib: dict of section area, unit = mm :superscript:`2`. If provided, application will request specifying
                    key of dict for each beam. Otherwise, non-zero positive value will be requested.
        :return: list. Assigned *Section Area* of beams, unit = mm :superscript:`2`.
        """
        return self.__setprop("A", lib)

    @Acad.multitry(5)
    def set_I(self, lib=None):
        """Assign moment of inertia for each beams.

        :param lib: dict of moment of inertia, unit = mm :superscript:`4`. If provided, application will request
                    specifying key of dict for each beam. Otherwise, non-zero positive value will be requested.
        :return: list. Assigned *Moment of Inertia* of beams, unit = mm :superscript:`4`.
        """
        return self.__setprop("I", lib)

if __name__ == '__main__':
    testsec_file = 'C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\testangle.json'
    with open(testsec_file) as sf:
        testsec = json.load(sf)
    sec=testsec['Section_02']
    dv = (1, 2, 3)
    dvs = [(4, 5, 5), (-1, 0, 2), (7, 9, 8), (-10, 2, 0), (-10, 7, 0)]
    ds = Acad.boundalong(sec['bnode'], sec['barc'], sec['belp'], dv, *dvs)
    print(ds)

