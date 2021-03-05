# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:05:29 2020
ver beta_1.0
@author: qi.wang
"""

from lxml import etree as et
from pyfacade.transxml import xml_define, xml_eval, xml_ind, xml_stat, xml_ex, xml_prog, Xexpr
import re
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import win32com.client
import os

# Constant: file location
Abspath = os.path.dirname(__file__)

# Dictionary: mapping color name to Hex color code
Hexcolor = {'red': '#ff0000',
            'maroon': '#800000',
            'pink': '#ff8080',
            'violet': '#ff80ff',
            'magenta': '#ff00ff',
            'orange': '#ff8000',
            'yellow': '#ffff80',
            'lime': '#80ff80',
            'green': '#00ff00',
            'aqua': '#80ffff',
            'blue': '#0000ff',
            'gray': '#c0c0c0'
            }

# Dictionary: mapping comparing operate to ADATA code
co = {"<": "&lt;",
      "<=": "&lt;=",
      "=": "=",
      ">=": "&gt;=",
      ">": "&gt;"}

# Dictionary: fastener size, (df, As) or (df, d.min)
Fst_size = {'M5': (5, 14.2),
            'M6': (6, 20.1),
            'M8': (8, 36.6),
            'M10': (10, 58.0),
            'M12': (12, 84.3),
            'M16': (16, 157),
            'M20': (20, 245),
            'M22': (22, 303),
            'M24': (24, 353),
            'ST10': (4.8, 3.43),
            'ST12': (5.5, 3.99),
            'ST14': (6.3, 4.70)}

# Dictionary: fastener grade, (tensile, shear, bearing) as per alum code | as per steel code
Fst_grade = {'A2-50': ((210, 125, 420), (210, 145, 511)),
             'A2-70': ((450, 267, 900), (450, 311, 828)),
             'A2-80': ((583, 347, 1166), (560, 384, 1008)),
             'A4-50': ((210, 125, 420), (210, 145, 511)),
             'A4-70': ((450, 267, 900), (450, 311, 828)),
             'A4-80': ((583, 347, 1166), (560, 384, 1008)),
             'Gr4.6': ((200, 119, 400), (240, 160, 460)),
             'Gr8.8': ((533, 317, 1066), (560, 375, 1000))}

# Dictionary: material properties, (read from csv)
Material = pd.read_csv(Abspath+"/material.csv", index_col="signature")


# Class: Xmcd file maker
class Xmcd:
    """Operation interface parsing and modifying xmcd file.

    :param file_path: str, full path of target xmcd file.
    """
    def __init__(self, file_path):
        self.__tree = et.parse(file_path)  # parse file
        self.__sht = self.__tree.getroot()  # get the root
        self.__ns = self.__sht.nsmap  # get namespaces
        self.__regs = self.__tree.find("regions", self.__ns)  # get the main element <Regions>
        self.__regid = [0]  # initialize the list of region id
        self.__endline = 0  # initialize the end line location
        for reg in self.__regs:
            self.__regid.append(int(reg.get('region-id')))
            row = float(reg.get('align-y'))
            if row > self.__endline:
                self.__endline = row
        self.current = self.__endline  # current insert location, default at end of file

    @property
    def worksheet(self):
        """Root element of the xmcd file. Read-only."""
        return self.__sht

    @property
    def namespace(self):
        """Name space used by the xmcd file. Read-only."""
        return self.__ns

    @property
    def regions(self):
        """Element <regions>, the main part of the xmcd file. Read-only."""
        return self.__regs

    def _printtree(self, element, indent=0, levels=3, show_namespace=True):
        # internal function for printing tree structure
        if show_namespace:
            p = r"(.+)"
        else:
            p = r"{.*}(.+)"
        if indent <= levels:
            if len(element):
                print("  " * indent + "|_", re.match(p, element.tag).group(1))
                for child in element:
                    self._printtree(element=child, indent=indent + 1, levels=levels, show_namespace=show_namespace)
            else:
                print("  " * indent + "|_", re.match(p, element.tag).group(1))

    def show_fulltree(self, levels=3, show_namespace=True):
        """Show the entire structure of xmcd file.

        :param levels: int, max. levels to be shown, count from root as level 0
        :param show_namespace: bool, include name space as initial of element tag
        :return: None
        """
        if type(levels) is int and levels >= 0:
            print("\n")
            if show_namespace:
                print(self.__sht.tag)
            else:
                print(re.match(r"{.*}(.+)", self.__sht.tag).group(1))

            for element in self.__sht:
                print("=" * 20)
                self._printtree(element, indent=1, levels=levels, show_namespace=show_namespace)
        else:
            raise ValueError("levels must be non-negative integer")

    def show_region(self, levels=3, show_namespace=True):
        """Show the sub-structure of element <regions>.

        :param levels: int, max. levels to be shown, count from element <regions> as level 0
        :param show_namespace: bool, include name space as initial of element tag
        :return: None
        """
        if type(levels) is int and levels >= 0:
            print("\n")
            if show_namespace:
                print("<{}>".format(self.__regs.tag))
            else:
                print("<{}>".format(re.match(r"{.*}(.+)", self.__regs.tag).group(1)))

            for element in self.__regs:
                self._printtree(element, indent=1, levels=levels, show_namespace=show_namespace)
        else:
            raise ValueError("levels must be non-negative integer")

    @classmethod
    def read_region(cls, file_path):
        """Read all the <region> sub-elements from specified xmcd file to a list.

        :param file_path: str, full path of target xmcd file.
        :return: list of <region> sub-elements.
        """
        return [x[1] for x in et.iterparse(file_path, tag="{http://schemas.mathsoft.com/worksheet30}region")]

    @staticmethod
    def dfupdate(dataframe, repls, columns=None, remove=True):
        """A quick tool to replace string content in specified columns of data frame surrounded by Curly Braces ``{}``
        according to provided dict.

        :param dataframe: a pandas.DateFrame
        :param repls: dict, {'replace_mark_1': new_1, 'replace_mark_2': new_2,...}
        :param columns: list of str, column names to be update, *None* for all columns.
        :param remove: bool, remove the row which contains invalid replace mark.
        :return: tuple, (list of index of replaced row, list of index of unreplaced row)
        """
        cols = columns if columns else dataframe.columns  # check all columns as default
        replaced = []
        unreplaced = []
        for i, row in dataframe.iterrows():
            for col in cols:
                content = row[col]
                if type(content) == str:  # only apply to string content
                    mat = re.search(r'{(.+?)}', content)  # search first replace mark
                    while mat:
                        try:
                            content = content.replace(mat.group(), str(repls[mat.group(1)]))
                            dataframe.loc[i, col] = content # update the original data frame
                            replaced.append(i)
                        except KeyError:
                            unreplaced.append(i)
                            break
                        mat = re.search(r'{(.+?)}', content)  # search next replace mark

        if remove:  # remove rows with invalid replace mark
            for x in unreplaced:
                dataframe.drop(index=x, inplace=True)

        return replaced, unreplaced

    def copyfrom(self, file_path, row_offset=24):
        """Copy all <region> sub-elements from a specified file to the end of current file.

        :param file_path: str, full path of target xmcd file.
        :param row_offset: float, offset from the the last line of existing part to the first line of pasted part when
                presented in MathCAD Worksheet.
        :return: None
        """
        rs = self.read_region(file_path)
        fisrt_line = min([float(x.get('align-y')) for x in rs])
        start_row = self.current + row_offset  # initial location
        for r in rs:
            newid = max(self.__regid) + 1
            r.attrib['region-id'] = str(newid)  # update region id before insert
            row = start_row + float(r.get('align-y')) - fisrt_line
            r.attrib['align-y'] = str(row)  # update row location

            self.__regs.append(r)  # insert region

            self.__regid.append(newid)  # record the new region id
            self.current = row  # renew current insert location
            if row > self.__endline:  # renew end line location
                self.__endline = row

    def addtext(self, text, row, col=30, width=164, border=False, highlight=False, bgc="inherit", tag=""
                , lock_width=True, style="Normal", bold=False, italic=False, underline=False, color=None):
        """Insert a text region to current xmcd file.

        :param text: str, content to be added into xmcd file.
        :param id: int, id of new text region.
        :param row: float, target row location.
        :param col: float, target column location.
        :param width: float, width of text region, only valid when *lock_width* = True.
        :param border: bool, show border of text region.
        :param highlight: bool, highlight text region.
        :param bgc: str, color name or hex code in lowercase of background, only valid when *highlight* = True.
        :param tag: str, tag of region.
        :param lock_width: bool, fix the width of text region.
        :param style: str, name of pre-defined text style.
        :param bold: bool, use bold font style.
        :param italic: bool, use italic font style.
        :param underline: bool, text with underline.
        :param color: str, color name or hex code in lowercase for text.
        :return: A copy of inserted <region> element.
        """

        newid = max(self.__regid) + 1  # create a id for new region
        # translate color name into hex number
        if bgc != "inherit":
            try:
                bgc = Hexcolor[bgc]
            except KeyError:
                if not re.match(r"^#[0-9a-f]{6}$", bgc):  # if not a hex color code
                    bgc = 'inherit'  # set to default

        if color:
            try:
                color = Hexcolor[color]
            except KeyError:
                if not re.match(r"^#[0-9a-f]{6}$", color):  # if not a hex color code
                    color = None  # set to default

        # region attribute
        reg_att = {'region-id': str(newid),
                   'left': str(col),  # column location
                   'top': '0',
                   'width': str(width),
                   'height': '22',  # pre-fixed
                   'align-x': '0',
                   'align-y': str(row),  # row location
                   'show-border': str(border).lower(),
                   'show-highlight': str(highlight).lower(),
                   'is-protected': 'true',  # pre-fixed
                   'z-order': '0',  # pre-fixed
                   'background-color': bgc,
                   'tag': tag}
        # text attribute
        txt_att = {'use-page-width': "false",
                   'push-down': "false",
                   'lock-width': str(lock_width).lower()}
        # content attribute
        cont_att = {'style': style,
                    'margin-left': "inherit",
                    'margin-right': "inherit",
                    'text-indent': "inherit",
                    'text-align': "inherit",
                    'list-style-type': "inherit",
                    'tabs': "inherit"}

        new_region = et.SubElement(self.__regs, "region", attrib=reg_att)
        new_text = et.SubElement(new_region, "text", attrib=txt_att)
        insert = et.SubElement(new_text, "p", attrib=cont_att)

        if bold:
            insert = et.SubElement(insert, "b")
        if italic:
            insert = et.SubElement(insert, "i")
        if underline:
            insert = et.SubElement(insert, "u")
        if color:
            insert = et.SubElement(insert, "c", attrib={'val': color})

        insert.text = text  # write text into region
        self.__regid.append(newid)  # record the new region id
        self.current = row  # renew current insert location
        if row > self.__endline:  # renew end line location
            self.__endline = row

        return new_region

    def addmath(self, var_name, row, col=204, border=False, highlight=False, bgc="inherit", tag="",
                expression=None, evaluate=False, unit=None):
        """Insert a math region to current xmcd file.

        :param var_name: str, name of variable.
        :param id: int, id of new text region.
        :param row: float, target row location.
        :param col: float, target column location.
        :param border: bool, show border of math region.
        :param highlight: bool. highlight math region
        :param bgc: str, color name or hex code in lowercase of background, only valid when *highlight* = True.
        :param tag: str, tag of region.
        :param expression: str, expression for variable definition. When expression is not provided, insert a math
                region of evaluation (if *evaluate* = True) or a individual math region of variable only
                (if *evaluate* = False)
        :param evaluate: bool, in-line evaluate the variable.
        :param unit: str, unit use to overwrite default unit in evaluating result.
        :return: A copy of inserted <region> element.
        """

        newid = max(self.__regid) + 1  # create a id for new region
        # translate color name into hex number
        if bgc != "inherit":
            try:
                bgc = Hexcolor[bgc]
            except KeyError:
                if not re.match(r"^#[0-9a-f]{6}$", bgc):  # if not a hex color code
                    bgc = 'inherit'  # set to default

        # region attribute
        reg_att = {'region-id': str(newid),
                   'left': str(col),  # column location
                   'top': '0',
                   'width': '50',  # pre-fixed
                   'height': '22',  # pre-fixed
                   'align-x': '0',
                   'align-y': str(row),  # row location
                   'show-border': str(border).lower(),
                   'show-highlight': str(highlight).lower(),
                   'is-protected': 'true',  # pre-fixed
                   'z-order': '0',  # pre-fixed
                   'background-color': bgc,
                   'tag': tag}

        # create sub-element for variable definition or evaluation
        if expression is None:
            if evaluate:
                se = et.fromstring(xml_eval(var_name, unit))
            else:
                se = et.fromstring(xml_ind(var_name))
        else:
            se = et.fromstring(xml_define(var_name, expression, evaluate, unit))

        # append sub-element into created region
        new_region = et.SubElement(self.__regs, "region", attrib=reg_att)
        new_math = et.SubElement(new_region, "math", attrib={'optimize': 'false', 'disable-calc': 'false'})
        new_math.append(se)

        self.__regid.append(newid)  # record the new region id
        self.current = row  # renew current insert location
        if row > self.__endline:  # renew end line location
            self.__endline = row

        return new_region

    def addcompare(self, row, csign="<=", var_name1=None, var_name2=None, col=204, border=False, highlight=False,
                   bgc="inherit", tag="",
                   expression1=None, evaluate1=False, unit1=None, expression2=None, evaluate2=False, unit2=None):
        """Insert a special fusion region with 2 math sub-regions connected by a comparing symbol to current xmcd file.

        :param row: float, target row location.
        :param csign: str, comparing operation.
        :param var_name1: str, name of variable at left-hand.
        :param var_name2: str, name of variable at right-hand.
        :param col: float, target column location.
        :param border: bool, show border of text region.
        :param highlight: bool, highlight text region.
        :param bgc: str, color name or hex code in lowercase for background,  only valid when *highlight* = True.
        :param tag: str, tag of region.
        :param expression1: str, expression for left-hand variable definition.
        :param evaluate1: bool, in-line evaluate the variable at left-hand.
        :param unit1: str, unit use to overwrite default unit in evaluating result at left-hand.
        :param expression2: str, expression for right-hand variable definition.
        :param evaluate2: bool. in-line evaluate the variable at right-hand.
        :param unit2: unit use to overwrite default unit in evaluating result at right-hand.
        :return: A copy of inserted <region> element.
        """
        newid = max(self.__regid) + 1  # create a id for new region
        # translate color name into hex number
        if bgc != "inherit":
            try:
                bgc = Hexcolor[bgc]
            except KeyError:
                if not re.match(r"^#[0-9a-f]{6}$", bgc):  # if not a hex color code
                    bgc = 'inherit'  # set to default

        # region attribute for main text region and 2 sub math region
        reg_att = {'region-id': str(newid),
                   'left': str(col),  # column location
                   'top': '0',
                   'width': '50',  # pre-fixed
                   'height': '22',  # pre-fixed
                   'align-x': '0',
                   'align-y': str(row),  # row location
                   'show-border': str(border).lower(),
                   'show-highlight': str(highlight).lower(),
                   'is-protected': 'true',  # pre-fixed
                   'z-order': '0',  # pre-fixed
                   'background-color': bgc,
                   'tag': tag}
        # text attribute = all default
        txt_att = {'use-page-width': "false",
                   'push-down': "false",
                   'lock-width': "false"}
        # content attribute = all default
        cont_att = {'style': "Normal",
                    'margin-left': "inherit",
                    'margin-right': "inherit",
                    'text-indent': "inherit",
                    'text-align': "inherit",
                    'list-style-type': "inherit",
                    'tabs': "inherit"}

        new_region = et.SubElement(self.__regs, "region", attrib=reg_att)
        new_text = et.SubElement(new_region, "text", attrib=txt_att)
        insert = et.SubElement(new_text, "p", attrib=cont_att)
        self.__regid.append(newid)  # record the new region id

        if var_name1:  # when left-hand-side is defined
            # create math sub-element 1
            if expression1 is None:
                se1 = et.fromstring(xml_eval(var_name1, unit1))
            else:
                se1 = et.fromstring(xml_define(var_name1, expression1, evaluate1, unit1))
            # append math sub-element 1 into created region
            newid += 1
            reg_att['region-id'] = str(newid)
            math_region_1 = et.SubElement(insert, "region", attrib=reg_att)
            new_math_1 = et.SubElement(math_region_1, "math", attrib={'optimize': 'false', 'disable-calc': 'false'})
            new_math_1.append(se1)
            self.__regid.append(newid)  # record the new region id
            # add sign as tail of sub math region
            if var_name2:  # if right-hand-side is also define:
                math_region_1.tail = "  " + csign + "  "
            else:
                math_region_1.tail = "  " + csign
        else:
            # add sign as text of main region
            insert.text = csign + "  "

        if var_name2:  # when right-hand-side is defined
            # create math sub-element 2
            if expression1 is None:
                se2 = et.fromstring(xml_eval(var_name2, unit2))
            else:
                se2 = et.fromstring(xml_define(var_name2, expression2, evaluate2, unit2))
            # append math sub-element 2 into created region
            newid += 1
            reg_att['region-id'] = str(newid)
            math_region_2 = et.SubElement(insert, "region", attrib=reg_att)
            new_math_2 = et.SubElement(math_region_2, "math", attrib={'optimize': 'false', 'disable-calc': 'false'})
            new_math_2.append(se2)
            self.__regid.append(newid)  # record the new region id

        self.current = row  # renew current insert location
        if row > self.__endline:  # renew end line location
            self.__endline = row

        return new_region

    def addsolve(self, conditionset, unknown_guess, row, spacing=24, txt_col=30, math_col=204, math_border=False,
                 math_highlight=False, math_bgc="inherit", tag="", unit=None, txt_border=False, txt_highlight=False,
                 txt_bgc="inherit", txt_bold=False, txt_italic=False, txt_underline=False, txt_color=None):
        """Insert a *Solve Block* to current xmcd file.

        :param conditionset: nested list, conditions of solving in the form of
                            [['left-hand expression, 'symbol', right-hand expression], [...],...]
                            'symbol' can be ``'=='``, ``'<'``, ``'<='``, ``'>'``, ``'>='``, ``'!='``
        :param unknown_guess: dict, guesses of unknown variables. in the form of
                                {'variable_1: guess_1, 'variable_2: guess_2, ...}
        :param row: float, start row location.
        :param spacing: float, line spacing.
        :param txt_col: float, target column location of text.
        :param math_col: float, target column location of math expression.
        :param math_border: bool, show border of math region.
        :param math_highlight: bool, highlight math region.
        :param math_bgc: str., color name or hex code in lowercase for math background,
                        only valid when *highlight* = True.
        :param tag: str, tag of region.
        :param unit: str, unit use to overwrite default unit in evaluating result.
        :param txt_border: bool, show border of text region.
        :param txt_highlight:  bool, highlight text region
        :param txt_bgc: str, color name or hex code in lowercase for text background,
                        only valid  when *highlight* = True.
        :param txt_bold: bool, use bold font style.
        :param txt_italic: bool, use italic font style.
        :param txt_underline: bool, text with underline.
        :param txt_color: str. color name or hex code in lowercase for text.
        :return: None
        """

        var_names = [v for v in unknown_guess]  # name list of unknown variables

        # add initial keyword of solving block
        self.addtext("Equations:", row, col=txt_col, width=164, border=txt_border, highlight=txt_highlight, bgc=txt_bgc,
                     tag="", lock_width=True, style="Normal", bold=txt_bold, italic=txt_italic, underline=txt_underline,
                     color=txt_color)
        self.addmath('Given', row=row, col=math_col, border=math_border, highlight=math_highlight, bgc=math_bgc, tag=tag,
                     expression=None, evaluate=False, unit=None)

        # add equations
        for i in range(len(conditionset)):
            row += spacing
            t = tag+f"_eq{i}" if tag else ""  # name tag of each equation
            lefthand, ev, righthand = conditionset[i]
            self.addmath(f'lgcp({ev},{lefthand},{righthand})', row=row, col=math_col, border=math_border,
                         highlight=math_highlight, bgc=math_bgc, tag=t, expression=None, evaluate=False, unit=None)

        # add guess value
        self.addtext("Guess Value:", row+spacing, col=txt_col, width=164, border=txt_border, highlight=txt_highlight,
                     bgc=txt_bgc, tag="", lock_width=True, style="Normal", bold=txt_bold, italic=txt_italic,
                     underline=txt_underline, color=txt_color)
        for var in unknown_guess:
            row += spacing
            t = tag + f"_guess_{var}" if tag else ""  # name tag of each guess
            self.addmath(var_name=var, row=row, col=math_col, border=math_border, highlight=math_highlight,
                         bgc=math_bgc, tag=t, expression=unknown_guess[var], evaluate=False, unit=None)

        # add Find function
        row += spacing * (1+len(var_names)/4)  # additional spacing
        self.addtext("Solving:", row, col=txt_col, width=164, border=txt_border, highlight=txt_highlight,
                     bgc=txt_bgc, tag="", lock_width=True, style="Normal", bold=txt_bold, italic=txt_italic,
                     underline=txt_underline, color=txt_color)
        unknowns = ",".join(var_names)
        t = tag + f"_sol" if tag else ""  # name tag
        self.addmath('solv', row=row, col=math_col, border=math_border, highlight=math_highlight,
                     bgc=math_bgc, tag=t, expression=f'find({unknowns})', evaluate=True, unit=unit)

        # add definition of unknowns
        row += spacing*len(var_names)/4  # additional spacing
        self.addtext("Solutions:", row+spacing, col=txt_col, width=164, border=txt_border, highlight=txt_highlight,
                     bgc=txt_bgc, tag="", lock_width=True, style="Normal", bold=txt_bold, italic=txt_italic,
                     underline=txt_underline, color=txt_color)
        for i in range(len(var_names)):
            row += spacing
            t = tag + f"_res_{var_names[i]}" if tag else ""  # name tag of each variable
            self.addmath(var_name=var_names[i], row=row, col=math_col, border=math_border, highlight=math_highlight,
                         bgc=math_bgc, tag=t, expression=f'solv_{i}', evaluate=True, unit=unit)


    def _fromdata(self, data):
        """Insert region(s) to xmcd file according to provided pandas.Series or pandas.DataFrame"""
        if isinstance(data, pd.Series):  # one-line data
            if data.type > 0:  # if data is pure text
                self.addtext(text=data.main, row=data.row + self.current, col=data.col, border=data.border,
                             highlight=data.highlight, bgc=data.bgc, tag=data.tag, lock_width=data.lock_width,
                             style=data.style, bold=data.bold, italic=data.italic, underline=data.underline,
                             color=data.color)
            elif data.type == 0:  # if data is math
                self.addmath(var_name=data.main, row=data.row + self.current, col=data.col, border=data.border,
                             highlight=data.highlight, bgc=data.bgc, tag=data.tag, expression=data.expression,
                             evaluate=data.evaluate, unit=data.unit)

            elif data.type == -1:  # if data is a comparison
                var1, cs, var2 = data.main.split(";")
                exp1, exp2 = data.expression.split(";")
                u1, u2 = data.unit.split(";")
                self.addcompare(row=data.row + self.current, csign=cs, var_name1=var1, var_name2=var2, col=data.col,
                                border=data.border, highlight=data.highlight, bgc=data.bgc, tag=data.tag,
                                expression1=exp1 if exp1 else None, evaluate1=data.evaluate, unit1=u1,
                                expression2=exp2 if exp2 else None, evaluate2=data.evaluate2, unit2=u2)

        elif isinstance(data, pd.DataFrame):  # multi-line data
            for index, item in data.iterrows():
                if item.type > 0:  # if data is pure text
                    self.addtext(text=item.main, row=item.row + self.current, col=item.col, border=item.border,
                                 highlight=item.highlight, bgc=item.bgc, tag=item.tag, lock_width=item.lock_width,
                                 style=item.style, bold=item.bold, italic=item.italic, underline=item.underline,
                                 color=item.color)
                elif item.type == 0:  # if data is math
                    self.addmath(var_name=item.main, row=item.row + self.current, col=item.col, border=item.border,
                                 highlight=item.highlight, bgc=item.bgc, tag=item.tag, expression=item.expression,
                                 evaluate=item.evaluate, unit=item.unit)
                elif item.type == -1:  # if data is a comparison
                    var1, cs, var2 = item.main.split(";")
                    exp1, exp2 = item.expression.split(";")
                    u1, u2 = item.unit.split(";")
                    self.addcompare(row=item.row + self.current, csign=cs, var_name1=var1, var_name2=var2, col=item.col,
                                    border=item.border, highlight=item.highlight, bgc=item.bgc, tag=item.tag,
                                    expression1=exp1 if exp1 else None, evaluate1=item.evaluate, unit1=u1,
                                    expression2=exp2 if exp2 else None, evaluate2=item.evaluate2, unit2=u2)

    def write(self, file_path):
        """save current xmcd as specified file

        :param file_path: str, path and the file name.
        :return:
        """
        self.__tree.write(file_path, encoding='utf-8')  # write to target file
        print("file is created as <{}>".format(file_path))

    # region  ============================= TOOLBOX =======================================

    def make_contents(self, file_name, title_style, init_page=1, page_space=700, indent=0, prefix=""):
        """Create table of contents for current xmcd file and save it to specified file.

        :param file_name: str, path and name of target file to save the contents as.
        :param title_style: list of str, style names of titles to be included in the contents.
        :param init_page: int, start number of first page.
        :param page_space: float, height of one page.
        :param indent: int, numbers of 'Space' as indentations of sub-level contents. The level of a content is according
                        to the index of its style in `title_style` list.
        :param prefix: str, prefix of page numbers.
        :return: None
        """
        # Read titles and corresponding row numbers
        titles = []
        for rg in self.regions:

            if re.match(r"{.*}text", rg[0].tag) and rg[0][0].get('style') in title_style:
                titles.append((float(rg.get('top')),
                               title_style.index(rg[0][0].get('style')),
                               et.tostring(rg, method="text", encoding="UTF-8").strip().decode("UTF-8")))

            elif re.match(r"{.*}pageBreak", rg[0].tag):
                titles.append((float(rg.get('top')), 0, None))

        # Make content "<section title> <page number>"
        base_row = 0
        base_page = init_page
        content = []
        for r, n, text in titles:
            if text:
                content.append(
                    f"{''.join([' '] * indent * n)}{text}\t{prefix}{int((r - base_row) // page_space + base_page)}\n")
            else:
                # print(f"page break at {r} -- based on {base_row} as start of page {base_page}")
                base_page = int((r - base_row) // page_space + base_page) + 1

                base_row = r

        # Write content to file
        with open(file_name, 'w') as f:
            for line in content:
                f.write(line)
        print(f"Content has been created in file <{file_name}>.")

    def addblock_fst(self, fst_layout, loads, eccentricity, ply_mat, ply_thk, ply_edg=[0, 0], hole_type=['standard', 'standard'],
                     pryedge=[0, 0], loadsubs=None, fastener="bolt", name="M6", grade='A4-70', packing=0, grib=0,
                     alum_code=True, sec_num="1.1", sec_title=None, template=None):
        """Insert a group of regions for fastener checking to current xmcd file.

        :param fst_layout: 2d list of float, coordinates of fasteners in the form of [[u1, v1], [u2, v2],...], unit=mm.
        :param loads: list of float, external load in form as N.mm [Fu, Fv, Fn, Mu, Mv, Mn], unit=N, mm.
        :param eccentricity: list of float, distance from loading point to reference origin [du, dv, dn], unit=mm.
        :param ply_mat: list of str, material name of connected parts [part1, part2].
        :param ply_thk: list of float, thickness of connected parts, unit=mm.
        :param ply_edg: list of float, edge distance on connected parts, unit=mm.
        :param hole_type: list of str, hole type on connected parts.
        :param pryedge: list of float, nominal lever arm for prying force about u and v, unit =mm.
        :param loadsubs: list of float, external load for substitution if argument loads includes algebra.
        :param fastener: str, general name of fastener to be shown in calculations.
        :param name: str, fastener name/code.
        :param grade: str, grade code of fastener.
        :param packing: float, packing thickness, unit=mm.
        :param grib: float, grib length, unit=mm.
        :param alum_code: bool, verify fasteners according to *BS8118* as fastener on aluminum member when True,
                            or according to *BS5950* as fastener on steel member when False.
        :param sec_num: str, section number to be shown in title of inserted part.
        :param sec_title: str, section title of inserted part.
        :param template: str, file path of the template used to create the regions. If no template is specified, a
                        default one will be searched from the package folder.
        :return: None
        """
        # check the validity of fastener definition:
        m = re.match(r"^(M|ST)(\d+)([-_]CSK)?$", name)  # valid name: M6, M10-CSK, ST10, ST12_CSK
        if m:
            ftype = m.group(1)  # fastener type
            size = m.group(2)  # fastener size
            csk = m.group(3)  # flag of counter-sunk head
        else:
            raise ValueError("Invalid Fastener Name.")

        fst_full = "{size} {csk}{ftpye}".format(size=ftype + size if ftype == "M" else size + "#",
                                                csk="Countersunk " if csk else "",
                                                ftpye="Self-tapping " if ftype == "ST" else "")
        # dict: hole coefficient
        kbs={"standard": 1,
             "oversized": 0.7,
             "short slotted": 0.7,
             "long slotted": 0.5,
             "kidney shaped": 0.5}

        # make load and eccentricity to dict
        fs = {'F.u': loads[0], 'F.v': loads[1], 'F.n': loads[2],
              'M.u': loads[3], 'M.v': loads[4], 'M.n': loads[5]}
        ds = {'du': f"{eccentricity[0]}*mm", 'dv': f"{eccentricity[1]}*mm", 'dn': f"{eccentricity[2]}*mm"}

        # analyze fastener layouts
        fst_u, fst_v = (zip(*fst_layout))
        line_u = True if len(set(fst_v)) == 1 else False  # all inline along u
        line_v = True if len(set(fst_u)) == 1 else False  # all inline along v

        # region ===================<Internal Calculation>=============================
        _fs = np.array(fst_layout)
        _nf = len(_fs)
        _cp = _fs.sum(axis=0) / _nf
        _load = loadsubs if loadsubs else loads
        _df = Fst_size[ftype + size][0]  # fastener size
        _t = min(ply_thk)  # thickness of connected part
        # define parameter c.
        if _df / _t <= 10:
            ratio_exp = "d.f//t;<= 10;"
            para_c = 2
        elif 10 < _df / _t < 13:
            ratio_exp = "d.f//t;< 13;"
            para_c = 20 * _t / _df
        else:
            ratio_exp = "d.f//t;>= 13;"
            para_c = 1.5

        if any([type(x) == str for x in _load]):  # incl. algebra, use simplified conservative method
            simp_eval = True
            maxt, maxs, maxc = False, False, True
            _n = np.power(_fs - _cp, 2).sum(axis=1).argmax()  # the furthest fastener
            um, vm = _fs[_n]  # coordinates of most critical fastener
            print("Block is simplified due to lacking of load information. the stated case may not be the critical one")
            print("Please review the output and double check.")

        else:  # evaluate the utilization and most critical fastener
            simp_eval = False
            _Iv, _Iu = np.power((_fs - _cp), 2).sum(axis=0)
            _Ip = _Iu + _Iv

            # define force
            _F = np.array(_load[:3])
            _M = np.array(_load[3:])
            _d = np.array(eccentricity)
            _Mc = np.cross(_d - np.append(_cp, 0), _F) + _M
            _vu, _vv, _fn = _F
            _mu, _mv, _mn = _Mc

            # define fastener capacity
            _ls = (Fst_grade[grade][0] if alum_code else Fst_grade[grade][1])
            if ftype == 'M':
                _As = Fst_size[ftype + size][1]  # Stress Area
            else:
                _As = np.pi * (Fst_size[ftype + size][1]) ** 2 / 4
            _Pt = _As * _ls[0]  # Tensile strength
            _Pb = _df * 0.5 * _t * _ls[2] if csk else _df * _t * _ls[2]  # Bearing strength
            _bp = min(9 * _df / (8 * _df + 3 * packing), 1)
            _bg = min((8 * _df) / (3 * _df + grib), 1)
            _Ps = min(_bp, _bg) * _As * _ls[1]  # Shear strength

            # define local bearing capacity of connected part
            _Plb = []
            for i in range(len(ply_mat)):
                _pb = Material.loc[ply_mat[i], "pb"]  # material bearing strength
                if re.match(r"S.+", ply_mat[i]):  # steel or stainless steel
                    _kbs = kbs[hole_type[0]]  # define hole coeff.
                    if ply_edg[i]:  # edge distance is defined
                        _plb = min(_kbs * _pb * _df * _t, 0.5 * _kbs * _pb * ply_edg[i] * _t)
                    else:
                        _plb = _kbs * _pb * _df * _t
                else:  # aluminum
                    if ply_edg[i]:  # edge distance is defined
                        _plb = min(_pb * para_c * _df * _t / 1.2, _pb * ply_edg[i] * _t / 1.2)
                    else:
                        _plb = _pb * para_c * _df * _t / 1.2
                # print(_plb)
                _Plb.append(_plb)

            # calculate capacity utilization per bolt
            _ult_t = np.zeros(_nf)
            _ult_s = np.zeros(_nf)
            _ult_c = np.zeros(_nf)
            for i in range(_nf):
                _bu, _bv = _fs[i]
                _ft1 = _fn / _nf
                _ft2 = abs(_mu) / (5 / 6 * pryedge[0]) if line_u else _mu * (_bv - _cp[1]) / _Iu
                _ft3 = abs(_mv) / (5 / 6 * pryedge[1]) if line_v else -_mv * (_bu - _cp[0]) / _Iv
                _ftm = _ft1 + _ft2 + _ft3
                _vm = np.sqrt((_vu / _nf - _mn * (_bv - _cp[1]) / _Ip) ** 2 +
                              (_vv / _nf + _mn * (_bu - _cp[0]) / _Ip) ** 2)
                _ult_t[i] = _ftm  # record tension
                _ult_s[i] = _vm  # record shear
                if _ftm == 0 or _vm == 0:  # no combine check required
                    _ult_c[i] = -1  # indicator for N/A
                else:  # combined utilization
                    _ult_c[i] = (max(_ftm, 0) / _Pt) ** 2 + (_vm / _Ps) ** 2 if alum_code else (max(_ftm,
                                                                                                    0) / _Pt + _vm / _Ps) / 1.4

            # find the most critical fastener
            _tm, = np.where(_ult_t == _ult_t.max())  # id of fasteners taking max tension
            _sm, = np.where(_ult_s == _ult_s.max())  # id of fasteners taking max shear
            _cm, = np.where(_ult_c == _ult_c.max())  # id of fasteners taking max combined force

            if max(_ult_c) == -1:  # no combination case
                maxc = False
                if all(_ult_t == 0):  # no tension case
                    maxt, maxs = False, True
                    um_s, vm_s = _fs[_sm[0]]  # V
                elif all(_ult_s == 0):  # no shear case
                    maxt, maxs = True, False
                    um_t, vm_t = _fs[_tm[0]]  # Ft
                else:  # both tension and shear
                    maxt, maxs = True, True
                    um_t, vm_t = _fs[_tm[0]]  # Ft
                    um_s, vm_s = _fs[_sm[1]]  # V
            else:  # combined case + any additional max tension/shear case
                maxc = True
                if set(_tm) & set(_sm):  # max Ft and V at the same fastener
                    maxt, maxs = False, False
                    um, vm = _fs[(set(_tm) & set(_sm)).pop()]
                elif set(_tm) & set(_cm):  # max Ft and Combine at the same fastener
                    maxt, maxs = False, True
                    um_s, vm_s = _fs[_sm[0]]  # V
                    um, vm = _fs[(set(_tm) & set(_cm)).pop()]  # Combine
                elif set(_sm) & set(_cm):  # max V and Combine at the same fastener
                    maxt, maxs = True, False
                    um_t, vm_t = _fs[_tm[0]]  # Ft
                    um, vm = _fs[(set(_sm) & set(_cm)).pop()]  # Combine
                else:
                    maxt, maxs = True, True
                    um_t, vm_t = _fs[_tm[0]]  # Ft
                    um_s, vm_s = _fs[_sm[1]]  # V
                    um, vm = _fs[_cm[2]]  # Combine

            # evaluate results
            fta_pass = _ult_t.max() <= _Pt  # max tension
            sas_pass = _ult_s.max() <= _Ps  # max shear vs. fastener shear capacity
            sab_pass = _ult_s.max() <= _Pb  # max shear vs. fastener bearing
            ft_pass = _ult_t[_cm[0]] <= _Pt  # tension in max combine
            ss_pass = _ult_s[_cm[0]] <= _Ps  # shear in max combine vs. fastener shear capacity
            sb_pass = _ult_s[_cm[0]] <= _Pb  # shear in max combine vs. fastener bearing
            com_pass = _ult_c.max() <= 1  # max combine utilization
            lb_pass = [_ult_s.max() <= p for p in _Plb]

            # output for debug
            # print(np.array(list(zip(_ult_t, _ult_s, _ult_c))))
            # print(fta_pass, (sas_pass and sab_pass), ft_pass, (ss_pass and sb_pass), com_pass, lb_pass)

        # endregion ================================================

        if not sec_title:
            sec_title = f"{fastener} Connection Check"  # section title
        sec_level = len(sec_num.split('.'))  # section level of this block
        self.addtext(sec_num + ' ' + sec_title.title(), row=self.current + 24, width=300, col=18, lock_width=False,
                     style="Heading {}".format(sec_level))  # write section title

        if not template:
            template = Abspath + "\\block_fst.csv"  # use default template file

        df = pd.read_csv(template)  # read the preset block content

        # -----------modify and clean the data------------------
        df.tag.fillna("", inplace=True)  # replace nan in tag by Empty String
        df = df.where(df.notna(), None)  # replace nan in other columns by None

        # update {} in text
        df.loc[df.type == 2, 'main'] = df.loc[df.type == 2, 'main'].map(
            lambda x: x.format(sec=sec_num, fst_full=fst_full, grd=grade, fst=fastener.title(), subsec='{subsec}'))
        df.loc[df.type == 2, 'style'] = df.loc[df.type == 2, 'style'].map(lambda x: x.format(h=sec_level + 1))
        df.loc[df.type == 1, 'main'] = df.loc[df.type == 1, 'main'].map(
            lambda x: x.format(fst=fastener.lower(), n='{n}'))

        # assign bolt number
        df.loc[df.main == "n.f", 'expression'] = len(fst_layout)

        # assign fastener coordinates
        for i in range(len(fst_layout)):
            repeat = df.loc[df.remark == "fstc"].copy()  # make a copy for repeated part
            repeat.loc[repeat.main == 'u_{n}', 'expression'] = str(fst_layout[i][0]) + 'mm'
            repeat.loc[repeat.main == 'v_{n}', 'expression'] = str(fst_layout[i][1]) + 'mm'
            repeat.loc[:, 'main'] = repeat.loc[:, 'main'].map(lambda x: x.format(n=i + 1, fst="bolt"))  # fastener number
            if i == 0:
                subdf = repeat
            else:
                subdf = subdf.append(repeat)
        # insert into dataframe
        df = df.loc[:df.loc[df.remark == "break1"].index.item()].append(subdf).append(df.loc[df.loc[df.remark == "contin1"].index.item():])

        # define default expression of moment:
        Mfu = Xexpr("-F.v*dn+F.n*(dv-v.c)+M.u", alias="M.f.u")
        Mfv = Xexpr("F.u*dn-F.n*(du-u.c)+M.v", alias="M.f.v")
        Mfn = Xexpr("-F.u*(dv-v.c)+F.v*(du-u.c)+M.n", alias="M.f.n")

        # define default expression of maximum force on fastener
        if simp_eval:  # using conservative simplification
            ft1 = "N.f//n.f"
            ft2 = f"+|(M.f.u)//(5//6*{pryedge[0]}*mm)" if line_u else "+|(M.f.u*(v.m-v.c))//I.u"
            ft3 = f"+|(M.f.v)//(5//6*{pryedge[1]}*mm)" if line_v else "+|(M.f.v*(u.m-u.c))//I.v"
        else:
            ft1 = "N.f//n.f"
            ft2 = f"+|(M.f.u)//(5//6*{pryedge[0]}*mm)" if line_u else "+M.f.u*(v.m-v.c)//I.u"
            ft3 = f"+|(M.f.v)//(5//6*{pryedge[1]}*mm)" if line_v else "-M.f.v*(u.m-u.c)//I.v"
        Ftm = Xexpr(ft1+ft2+ft3, alias="F.t.m")
        Vum = Xexpr("|(V.f.u//n)", alias="V.u.m") if line_u \
            else Xexpr("|(V.f.u//n.f-M.f.n*(v.m-v.c)//I.p)", alias="V.u.m")
        Vvm = Xexpr("|(V.f.v//n.f)", alias="V.v.m") if line_v \
            else Xexpr("|(V.f.v//n.f+M.f.n*(u.m-u.c)//I.p)", alias="V.v.m")

        # assign loads
        ms = ['M.f.u', 'M.f.v', 'M.f.n']
        first_l = 'F.u'
        for f in fs:
            i = df.loc[df.main == f].index

            if fs[f]:  # value is not 0, define load
                if type(fs[f]) == str:
                    df.loc[i, ['expression', 'evaluate']] = [fs[f], True]
                else:
                    df.loc[i, 'expression'] = str(fs[f]) + ('N' if f in ['F.u', 'F.v', 'F.n'] else 'N*mm')

            else:  # value is 0, drop statement and modify the related formula
                if df.loc[i, 'main'].item() == first_l:  # if the dropped formula is the first line
                    df.loc[i + 1, 'row'] = 0  # modify the location of next line
                    first_l = df.loc[i + 1, 'main'].item()
                df.drop(i, inplace=True)  # delete statement

                if f in ['F.n', 'F.u', 'F.v']:  # delete description and statement for N.f/V.fu/V.fv
                    f_index = df.loc[df.expression == f].index
                    df.drop(f_index - 1, inplace=True)
                    df.drop(f_index, inplace=True)
                    if f == 'F.n':
                        Ftm.zero('N.f')
                    elif f == 'F.u':
                        Vum.zero('V.f.u')
                    elif f == 'F.v':
                        Vvm.zero('V.f.v')

                # set zero para in moment expression to simply the formula
                for m in [Mfu, Mfv, Mfn]:
                    m.zero(f)

        # assign eccentricity
        for m in [Mfu, Mfv, Mfn]:
            m.sub(ds, simp=True)

        # add moment expression to data if it is Not zero, or simplify related formula if it is zero
        if Mfu:
            Mfu.inject(df)
        else:
            Ftm.zero(Mfu.alias)
        if Mfv:
            Mfv.inject(df)
        else:
            Ftm.zero(Mfv.alias)
        if Mfn:
            Mfn.inject(df)
        else:
            Vum.zero(Mfn.alias)
            Vvm.zero(Mfn.alias)

        # get limit stress in string form
        ls = [f"{x}*MPa" for x in (Fst_grade[grade][0] if alum_code else Fst_grade[grade][1])]

        # assign reduction to shear strength if necessary
        if packing and (Vum or Vvm):
            df.loc[df.main == 't.pa', 'expression'] = str(packing) + 'mm'
            df.loc[df.main == '@b.p', 'expression'] = 'min((9*d.f)//(8*d.f+3*t.pa),1.0)'
        if grib and (Vum or Vvm):
            df.loc[df.main == 'T.g', 'expression'] = str(grib) + 'mm'
            df.loc[df.main == '@b.g', 'expression'] = 'min((8*d.f)//(3*d.f+T.g),1.0)'

        # assign fastener information
        df.loc[df.main == 'd.f', 'expression'] = f"{Fst_size[ftype + size][0]}*mm"  # norm. diameter

        if ftype == 'M':  # machine screw / bolt
            df.loc[df.main == 'A.s', 'expression'] = f"{Fst_size[ftype + size][1]}*mm^2"
        else:  # self-tapping screw
            df.loc[df.main == 'd.min', 'expression'] = f"{Fst_size[ftype + size][1]}*mm"
            df.loc[df.main == 'Stress area:', 'row'] = 36  # make line wider
            df.loc[df.main == 'A.s', ['expression', 'evaluate']] = ['@p*d.min^2//4', True]

        df.loc[df.main == 't', 'expression'] = str(min(ply_thk)) + 'mm'  # thickness

        # assign capacity only when corresponding force exist
        if Ftm:
            df.loc[df.main == 'P.t', 'expression'] = '2//3*A.s*' + ls[0] if csk else 'A.s*' + ls[0]  # Tensile strength
        if Vum or Vvm:
            df.loc[df.main == 'P.b', 'expression'] = 'd.f*0.5*t*' + ls[2] if csk else 'd.f*t*' + ls[
                2]  # Bearing strength
            # shear strength
            if packing and grib:
                df.loc[df.main == 'P.s', 'expression'] = 'min(@b.p*A.s*{ps},@b.g*A.s*{ps})'.format(ps=ls[1])
            elif packing:
                df.loc[df.main == 'P.s', 'expression'] = '@b.p*A.s*{ps}'.format(ps=ls[1])
            elif grib:
                df.loc[df.main == 'P.s', 'expression'] = '@b.g*A.s*{ps}'.format(ps=ls[1])
            else:
                df.loc[df.main == 'P.s', 'expression'] = 'A.s*{ps}'.format(ps=ls[1])

        if maxt:  # calculate max tension case additionally
            # coordinates of fastener with maximum tension
            df.loc[df.main == 'u.m.t', 'expression'] = str(um_t) + 'mm'
            df.loc[df.main == 'v.m.t', 'expression'] = str(vm_t) + 'mm'
            Ftm.sub({"u.m": "u.m.t", "v.m": "v.m.t"}, inplace=False).inject(df, alias="F.t.ma")
            if not simp_eval:  # activate evaluation conclusion
                if fta_pass:
                    df.loc[df.main == 'eval_fta', ['type', 'main']] = [100, "OK!"]
                else:
                    df.loc[df.main == 'eval_fta', ['type', 'main', 'color']] = [100, "Fail!", 'red']
                    df.loc[df.main == 'F.t.ma;<;P.t', ['main', 'bgc']] = ['F.t.ma;>;P.t', 'pink']
        else:
            df.drop(df.loc[df.remark == 'maxt'].index, inplace=True)

        if maxs:  # calculate max shear case additionally
            # coordinates of fastener with maximum shear
            df.loc[df.main == 'u.m.s', 'expression'] = str(um_s) + 'mm'
            df.loc[df.main == 'v.m.s', 'expression'] = str(vm_s) + 'mm'
            if Vum and Vvm:
                Vum.sub({"u.m": "u.m.s", "v.m": "v.m.s"}, inplace=False).inject(df, alias="V.u.ma")
                Vvm.sub({"u.m": "u.m.s", "v.m": "v.m.s"}, inplace=False).inject(df, alias="V.v.ma")
                df.loc[df.main == 'V.ma', 'expression'] = "\\(V.u.ma^2+V.v.ma^2)"
            elif Vum:
                Vum.sub({"u.m": "u.m.s", "v.m": "v.m.s"}, inplace=False).inject(df, alias="V.u.ma")
            elif Vvm:
                Vvm.sub({"u.m": "u.m.s", "v.m": "v.m.s"}, inplace=False).inject(df, alias="V.v.ma")
            if not simp_eval:  # activate evaluation conclusion
                if sas_pass and sab_pass:
                    df.loc[df.main == 'eval_sa', ['type', 'main']] = [100, "OK!"]
                else:
                    df.loc[df.main == 'eval_sa', ['type', 'main', 'color']] = [100, "Fail!", 'red']
                    if not sas_pass:
                        df.loc[df.main == 'V.ma;<;P.s', ['main', 'bgc']] = ['V.ma;>;P.s', 'pink']
                    if not sab_pass:
                        df.loc[df.main == 'V.ma;<;P.b', ['main', 'bgc']] = ['V.ma;>;P.b', 'pink']
        else:
            df.drop(df.loc[df.remark == 'maxs'].index, inplace=True)

        if maxc:  # calculate the max combine case
            # coordinates of fastener with maximum combined force
            df.loc[df.main == 'u.m', 'expression'] = str(um) + 'mm'
            df.loc[df.main == 'v.m', 'expression'] = str(vm) + 'mm'
            # assign strength checking to data if it is Not zero
            if Ftm:
                Ftm.inject(df)
                df.loc[df.main == 'F.t.m;<;P.t', 'type'] = -1  # activate the comparison
                if not simp_eval:  # activate the evaluation conclusion
                    if ft_pass:
                        df.loc[df.main == 'eval_ft', ['type', 'main']] = [100, "OK!"]
                    else:
                        df.loc[df.main == 'eval_ft', ['type', 'main', 'color']] = [100, "Fail!", 'red']
                        df.loc[df.main == 'F.t.m;<;P.t', ['main', 'bgc']] = ['F.t.m;>;P.t', 'pink']
            if Vum and Vvm:
                Vum.inject(df)
                Vvm.inject(df)
                df.loc[df.main == 'V.m', 'expression'] = "\\(V.u.m^2+V.v.m^2)"
            elif Vum:
                Vum.inject(df, alias="V.m")
            elif Vvm:
                Vvm.inject(df, alias="V.m")
            if Vum or Vvm:
                df.loc[df.main == 'V.m;<;P.s', 'type'] = -1  # activate the comparison
                df.loc[df.main == 'V.m;<;P.b', 'type'] = -1  # activate the comparison
                if not simp_eval:  # activate the evaluation conclusion
                    if ss_pass and sb_pass:
                        df.loc[df.main == 'eval_s', ['type', 'main']] = [100, "OK!"]
                    else:
                        df.loc[df.main == 'eval_s', ['type', 'main', 'color']] = [100, "Fail!", 'red']
                        if not ss_pass:
                            df.loc[df.main == 'V.m;<;P.s', ['main', 'bgc']] = ['V.m;>;P.s', 'pink']
                        if not sb_pass:
                            df.loc[df.main == 'V.m;<;P.b', ['main', 'bgc']] = ['V.m;>;P.b', 'pink']
            if Ftm and (Vum or Vvm):
                if alum_code:
                    df.loc[df.main == '@b', 'expression'] = "(F.t.m//P.t)^2+(V.m//P.s)^2"
                    limit = 1
                else:
                    df.loc[df.main == '@b', 'expression'] = "F.t.m//P.t+V.m//P.s"
                    limit = 1.4
                df.loc[df.main == '@b;<{limit};', 'type'] = -1  # activate the comparison
                df.loc[df.main == '@b;<{limit};', 'main'] = \
                    df.loc[df.main == '@b;<{limit};', 'main'].map(lambda x: x.format(limit=limit))
                if not simp_eval:  # activate the evaluation conclusion
                    if com_pass:
                        df.loc[df.main == 'eval_com', ['type', 'main']] = [100, "OK!"]
                    else:
                        df.loc[df.main == 'eval_com', ['type', 'main', 'color']] = [100, "Fail!", 'red']
                        df.loc[df.main == f'@b;<{limit};', ['main', 'bgc']] = [f'@b;>{limit};', 'pink']
        else:
            df.drop(df.loc[df.remark == 'maxc'].index, inplace=True)

        # local bearing check for each connected part
        if Vum or Vvm:  # when shear exists
            for i in range(len(ply_mat)):

                pbs = Material.loc[ply_mat[i], "pb"]

                if re.match(r"S.+", ply_mat[i]):  # steel or stainless steel
                    lbc = df.loc[df.remark == "lbc_s"].copy()  # make a copy for repeated part
                    lbc.loc[lbc.main == "k.bs", "expression"] = str(kbs[hole_type[0]])  # define hole coeff.
                    if ply_edg[i]:  # edge distance is defined
                        lbc.loc[lbc.main == "e", "expression"] = f"{ply_edg[i]}*mm"
                        lbc.loc[lbc.main == "P.lb", "expression"] = f"min(k.bs*{pbs}*MPa*d.f*t,0.5*k.bs*{pbs}*MPa*e*t)"
                    else:
                        lbc.loc[lbc.main == "P.lb", "expression"] = f"k.bs*{pbs}*MPa*d.f*t"

                else:  # aluminum
                    lbc = df.loc[df.remark == "lbc_a"].copy()  # make a copy for repeated part
                    lbc.loc[lbc.main == "d/t_ratio", "main"] = ratio_exp  # define d/t ratio
                    lbc.loc[lbc.main == "c", "expression"] = str(para_c)  # define parameter c.
                    if ply_edg[i]:  # edge distance is defined
                        lbc.loc[lbc.main == "e", "expression"] = f"{ply_edg[i]}*mm"
                        lbc.loc[
                            lbc.main == "P.lb", "expression"] = f"min(({pbs}*MPa*c*d.f*t)//@g.m.a,({pbs}*MPa*e*t)//@g.m.a)"
                    else:
                        lbc.loc[lbc.main == "P.lb", "expression"] = f"({pbs}*MPa*c*d.f*t)//@g.m.a"
                # update section numbering
                lbc.loc[lbc.type == 2, "main"] = lbc.loc[lbc.type == 2, "main"].map(lambda x: x.format(subsec=5 + i))
                # update evaluation
                if maxs:
                    lbc.loc[lbc.main == "P.lb;>;{V}", "main"] = lbc.loc[lbc.main == "P.lb;>;{V}", "main"].map(
                        lambda x: x.format(V="V.ma"))
                else:
                    lbc.loc[lbc.main == "P.lb;>;{V}", "main"] = lbc.loc[lbc.main == "P.lb;>;{V}", "main"].map(
                        lambda x: x.format(V="V.m"))
                # activate the evaluation conclusion
                if not simp_eval:
                    if lb_pass[i]:
                        lbc.loc[lbc.main == 'eval_lb', ['type', 'main']] = [100, "OK!"]
                    else:
                        lbc.loc[lbc.main == 'eval_lb', ['type', 'main', 'color']] = [100, "Fail!", 'red']
                        if maxs:
                            lbc.loc[lbc.main == "P.lb;>;V.ma", ["main", 'bgc']] = ["P.lb;<;V.ma", 'pink']
                        else:
                            lbc.loc[lbc.main == "P.lb;>;V.m", ["main", 'bgc']] = ["P.lb;<;V.m", 'pink']

                # record the repeated part
                if i == 0:
                    sec_lbc = lbc
                else:
                    sec_lbc = sec_lbc.append(lbc)
            df = df.loc[:df.loc[df.remark == "break2"].index.item()].append(sec_lbc)

        else:
            df = df.loc[:df.loc[df.remark == "break2"].index.item()]

        # final cleaning: drop all the statement with expression as 'unknown'
        df.drop(df.loc[df.expression == 'unknown'].index - 1, inplace=True)
        df.drop(df.loc[df.expression == 'unknown'].index, inplace=True)
        # drop useless properties statements
        if all(df.main != 'M.f.u') or line_u:
            df.drop([9, 10], inplace=True)  # delete Iu
        if all(df.main != 'M.f.v') or line_v:
            df.drop([11, 12], inplace=True)  # delete Iv
        if all(df.main != 'M.f.n'):
            df.drop([13, 14], inplace=True)  # delete Ip

        self._fromdata(df)  # write until break

    def addblock_lbcheck(self, section_elem, material='6063-T6', full_detail=False, avg_thk=True, sec_num="2.1.1",
                         sec_title="Local Buckling Check", template=None):
        """Insert a group of regions for local buckling checking of aluminum member.

        :param section_elem: dict, elements data of a section in the form of
                             {'1': {'length': .., 'thks': [..], 'rp': [..], 'slope': .., 'bx': [..], 'by': [..],
                             'Ie': .., 'type': ..},
                             '2':{...}
                             ...}
        :param material: str: material name.
        :param full_detail: bool, show full calculation detail.
        :param avg_thk: bool, get element thickness from average thickness of intersections when element is internal.
        :param sec_num: str, section number to be shown in title of inserted part.
        :param sec_title: str, section title of inserted part.
        :param template: str, file path of the template used to create the regions. If no template is specified, a
                        default one will be searched from the package folder.
        :return: None
        """

        sec_level = len(sec_num.split('.'))  # section level of this block
        self.addtext(sec_num + ' ' + sec_title.title(), row=self.current + 24, width=300, col=18, lock_width=False,
                     style="Heading {}".format(sec_level))  # write section title

        if not template:
            template = Abspath+"block_lbcheck.csv"  # use default template file

        df = pd.read_csv(template)  # read the preset block content
        df.tag.fillna("", inplace=True)  # replace nan in tag by Empty String
        df = df.where(df.notna(), None)  # replace nan in other columns by None

        rep = {}  # dict of replacement mapping

        for num, lb_data in section_elem.items():

            t = np.mean(lb_data['thks']) if avg_thk else min(lb_data['thks'])  # element thickness

            if lb_data['type'] == 'X' or lb_data['type'] == 'Y':  # internal element under stress gradient

                cal_part = "int_grad"  # related part name
                internal = True

                # define y.o and y.c
                ys = lb_data['bx'] if lb_data['type'] == 'X' else lb_data['by']
                yc = np.abs(ys).max()
                yo = np.abs(ys).min()*np.sign(ys[0]*ys[1])
                g = 0.7+0.3*(yo/yc) if yo/yc > -1 else 0.8/(1-yo/yc)
                beta = g*lb_data['length']/t

                rep.update({'exp_yo': f"{round(yo, 3)}*mm",
                            'exp_yc': f"{round(yc, 3)}*mm",
                            'exp_t': f"{round(t, 3)}*mm"})
                if full_detail:
                    rep['exp_g'] = xml_prog(["pgif(lgand(lgcp(<,y.o//y.c,1),lgcp(>,y.o//y.c,-1)),0.70+0.30*(y.o//y.c))",
                                            "pgif(lgcp(<=,y.o//y.c,-1),0.80//(1-y.o//y.c))"])
                    rep['row_1'] = 36
                    rep['row_2'] = 96
                else:
                    rep['exp_g'] = '0.70+0.30*(y.o//y.c)' if yo/yc > -1 else '0.8//(1-yo//yc)'
                    rep['row_1'] = 36
                    rep['row_2'] = 36

            elif lb_data['type'] == 'U':  # internal or outstanding element under uniform compression

                cal_part = "uniform"  # related part name
                b = lb_data['length']  # length of element
                beta = b / t

                if len(lb_data['thks']) == 2:  # internal element
                    internal = True
                    rep['ele_type'] = 'Internal'

                elif len(lb_data['thks']) == 1:  # outstanding element
                    internal = False
                    rep['ele_type'] = 'Outstanding'

                rep.update({'exp_b': f"{round(b, 3)}*mm",
                            'exp_t': f"{round(t, 3)}*mm"})

            elif lb_data['type'] == 'G':  # outstanding element under stress gradient

                cal_part = "out_grad"  # related part name
                internal = False
                d = lb_data['length']  # length of element
                beta = d / t

                rep.update({'exp_d': f"{round(d, 3)}*mm",
                            'exp_t': f"{round(t, 3)}*mm"})

            elif lb_data['type'] == 'R':  # reinforced outstanding element under stress gradient

                cal_part = "out_rein"  # related part name
                internal = False
                b = lb_data['length']  # length of element
                Ie = lb_data['Ie']
                # solving for 'c'
                def func(x):
                    return t * x ** 3 / 12 + x * t * ((x + t) / 2) ** 2 + b * t ** 3 / 12 - Ie
                c = fsolve(func, 10)[0]
                h = (1+0.1*(c/t-1)**2)**(-0.5)
                beta = h*b/t

                rep.update({'exp_b': f"{round(b, 3)}*mm",
                            'exp_t': f"{round(t, 3)}*mm",
                            'exp_Ie': f"{round(Ie, 3)}*mm^4"})

                if full_detail:
                    rep['exp_h'] = xml_prog(["pgif(lgcp(>=,c//t,1),(1+0.1*(c//t-1)^2)^(-0.5))",
                                            "pgelse(1)"])
                    rep['a_0'] = 0
                    rep['a_1'] = 1
                    rep['b_0'] = -255  # hind {b_0} line
                    rep['row_1'] = 36
                    rep['row_2'] = 48
                else:
                    rep['exp_h'] = '(1+0.1*(c//t-1)^2)^(-0.5)' if c/t >= 1 else '1'
                    rep['exp_c'] = f"{round(c, 3)}*mm"
                    rep['a_0'] = -255  # hind {a_0} line
                    rep['a_1'] = -255  # # hind {a_1} line
                    rep['b_0'] = 0
                    rep['row_1'] = 36
                    rep['row_2'] = 30

            cals = df.loc[df.remark == cal_part].copy()  # read corresponding calculation part
            rep['num'] = f"Element {num}"  # element number

            # evaluation part
            eps = np.sqrt(250 / Material.loc[material, 'py'])
            slender = False  # slender flag
            limit_0 = 18 if internal else 6
            limit_1 =22 if internal else 7
            if beta <= limit_0*eps:
                rep['lb_class'] = 'Fully Compact'
                rep['coff'] = limit_0
                rep['sign'] = '<='
            elif beta <= limit_1*eps:
                rep['lb_class'] = 'Semi_compact'
                rep['coff'] = limit_1
                rep['sign'] = '<='
            else:
                rep['lb_class'] = 'Slender'
                rep['coff'] = limit_1
                rep['sign'] = '>'
                slender = True

            Xmcd.dfupdate(cals, repls=rep, columns=['main', 'type', 'row', 'expression'])  # update
            cals.type = cals.type.astype(int)  # change columns to correct type
            cals.row = cals.row.astype(float)  # change columns to correct type
            self._fromdata(cals)  # write into xml

            if slender:  # reduce element thickness
                reds = df.loc[df.remark == 'reduce'].copy()  # read corresponding calculation part
                if internal:
                    rep['cur'] = 'C'
                    rep['exp_kL'] = '32//x-220//x^2'
                    rep['row_3'] = 36
                else:
                    rep['cur'] = 'A'
                    if full_detail:
                        rep['exp_kL'] = xml_prog(["pgif(lgand(lgcp(>,x,7),lgcp(<=,x,12.1)),11//x-28//x^2)",
                                                  "pgif(lgcp(>,x,12.1),105//x^2)"])
                        rep['row_3'] = 72
                    else:
                        rep['exp_kL'] = '11//x-28//x^2' if beta/eps <= 12.1 else '105//x^2'
                        rep['row_3'] = 36
                Xmcd.dfupdate(reds, repls=rep, columns=['main', 'row', 'expression'])  # update
                reds.row = reds.row.astype(float)  # change columns to correct type
                reds.type = reds.type.astype(int)  # change columns to correct type
                self._fromdata(reds)  # write into xml

    # endregion


# Class: MathCAD API
class Mathcad():
    """MathCAD Automation API

    :param visible: bool, show MathCAD Application UI after launching.
    """
    def __init__(self, visible=True):
        self.__app = win32com.client.Dispatch("MathCAD.Application")
        self.__app.Visible = visible
        print(f"MathCAD ver. {self.__app.version} has been launched successfully from <{self.__app.fullname}>")

    @property
    def visible(self):
        """bool, visibility of application."""
        return self.__app.Visible

    @visible.setter
    def visible(self, isvisible):
        self.__app.Visible = isvisible

    @property
    def filepath(self):
        """str, default file path."""
        return self.__app.DefaultFilePath

    @filepath.setter
    def filepath(self, path):
        self.__app.DefaultFilePath = path

    def worksheet(self, file_name=None):
        """Request a worksheet object with specified file name.

        :param file_name: str, name (and path) of requested xmcd file. When specified, the target worksheet will be
                        firstly searched from files currently opened by application. If no valid worksheet is found,
                        the function then try to load requested file according to specified name and path. Otherwise,
                        if *file_name* is left None as default, the current active worksheet will be returned.
        :return: ``pymcad.Worksheet`` object.
        """

        # make a worksheet instance
        if file_name:
            for sht in self.__app.worksheets:  # find if requested file is opened
                if sht.Worksheet.Name == file_name or sht.Worksheet.FullName == file_name:
                    w, = sht.Worksheet.Windows
                    w.Activate()  # bring the requested one to top
                    return Worksheet(sht.Worksheet)

            try:
                ws = self.__app.WorkSheets.Open(file_name)  # load requested file
            except:
                print(f"Can't open {file_name}")
                raise
            else:
                print(f"MathCAD file <{ws.Name}> has been loaded successfully")
                return Worksheet(ws)
        else:
            return Worksheet(self.__app.ActiveWorksheet)  # return active worksheet as default

    def sheetslist(self, fullname=False):
        """Get a name list of worksheet opened by application.

        :param fullname: bool, show full path of the worksheet.
        :return: list of str.
        """
        if fullname:
            return [sht.Worksheet.FullName for sht in self.__app.worksheets]
        else:
            return [sht.Worksheet.Name for sht in self.__app.worksheets]

    def closeall(self, quit=False, save=True, mute=True):
        """Close all the worksheets.

        :param quit: bool, quit the application after closing worksheets.
        :param save: bool, save the worksheets before closing.
        :param mute: bool, save automatically without asking for user confirmation.
        :return: None.
        """
        if save:
            saveopt = 0 if mute else 1
        else:
            saveopt = 2
        if quit:
            self.__app.Quit(saveopt)
            print("MathCAD application is closed.")
        else:
            self.__app.CloseAll(saveopt)


# Class: MathCAD Worksheet Object
class Worksheet():
    """Interface with MathCAD worksheet.

    Create a instance by calling ``worksheet`` method of ``pymcad.Mathcad``.
    """
    def __init__(self, mcsheet):
        self.__sht = mcsheet
        self.__window, = mcsheet.Windows

    @property
    def name(self):
        """Name of the worksheet. Read-only."""
        return self.__sht.Name

    @property
    def path(self):
        """Path of the worksheet file. Read-only."""
        return self.__sht.Path

    @property
    def fullname(self):
        """The fully-qualified path to the worksheet. Read-only."""
        return self.__sht.FullName

    @property
    def windowstate(self):
        """State of the worksheet window.

        | 0- The window is maximized.
        | 1- The window is minimized.
        | 2- The window is resizable.
        """
        return self.__window.WindowState

    @windowstate.setter
    def windowstate(self, state_enum):
        self.__window.WindowState = state_enum

    def region(self, tag):
        """Request a region object in the worksheet by its tag name.

        :param tag: str, tag name of requested region.
        :return: ``pymcad.Region`` object.
        """

        for reg in self.__sht.Regions:
            if reg.Tag == tag:
                return Region(reg, self.__window)
        print(f"Region with tag '{tag}' not found.")

    def activate(self):
        """Activate the worksheet and bring it to the top of the application UI.

        :return: None.
        """
        self.__window.Activate()

    def save(self, full_file_name=None):
        """Save the worksheet.

        :param full_file_name: str, path and full name to save the file as. If it is not provided, the file will be
                                saved in-place.
        :return: None.
        """
        if full_file_name:
            self.__sht.SaveAs(full_file_name, 20)
            print(f"File has been saved as <{full_file_name}>")

        else:  # no file name specified
            self.__sht.Save()
            print("File has been saved.")

    def getvalue(self, var_name):
        """Read the **last** value of a variable in worksheet.

        :param var_name: str, name of variable.
        :return: | str if the value of variable is a string.
                 | float if the value of variable is a real number.
                 | complex if the value of variable is a complex.
                 | numpy.ndarray if the value of variable is a matrix.
        """
        val = self.__sht.GetValue(var_name)
        if val.Type == 'Numeric':
            if val.Imag:  # value is a complex
                return complex(val.Real, val.Imag)
            else:
                return val.Real
        elif val.Type == "String":
            return val.Value
        elif val.Type == "Matrix":
            m = np.zeros((val.rows, val.cols))
            for j in range(val.rows):
                for k in range(val.cols):
                    m[j, k] = val.GetElement(j, k).Real
            return m

    def setvalue(self, var_name, value):
        """Set Initial value of a variable.

        .. warning:: value set by this way could **NOT** be saved.

        :param var_name: str, name of variable.
        :param value: float, string, or array-like.
        :return: None.
        """
        self.__sht.SetValue(var_name, value)

    def recalculate(self):
        """Re-calculate the worksheet.

        :return: None.
        """
        self.__sht.Recalculate()

    def scrollto(self, coordinate_x, coordinate_y):
        """Scroll the window of worksheet to specified location

        :param coordinate_x: float, X-coordinate to scroll the window to.
        :param coordinate_y: float, y-coordinate to scroll the window to.
        :return: None.
        """
        self.__window.ScrollTo(int(coordinate_x), int(coordinate_y))

    def printout(self):
        """Print the work sheet by default print setting.

        :return: None.
        """
        print(f"Printing <{self.name}>...")
        self.__sht.PrintAll()

    def close(self, save=True, mute=True):
        """Close the worksheet.

        :param save: bool, save the worksheet before closing.
        :param mute: bool, save automatically without asking for user confirmation.
        :return: None.
        """
        if save:
            saveopt = 0 if mute else 1
        else:
            saveopt = 2
        msg=f"File <{self.name}> is closed."
        self.__sht.Close(saveopt)
        print(msg)


# Class: Region Object
class Region():
    """Interface with region in MathCAD worksheet.

    Create a instance by calling ``region`` method of ``pymcad.Worksheet``.'
    """
    def __init__(self, mcregion, wnd):
        self.__reg = mcregion
        self.__wnd = wnd

    @property
    def x(self):
        """X-coordinate of the region. Read-only."""
        return self.__reg.X

    @property
    def y(self):
        """Y-coordinate of the region. Read-only."""
        return self.__reg.Y

    @property
    def tag(self):
        """tag name of the region. Read-only."""
        return self.__reg.Tag

    @property
    def type(self):
        """Type of the region. Read-only.

        | 0 - Text region.
        | 1 - Math region.
        | 2 - Bitmap region.
        | 3 - Metafile region.
        | 4 - OLE object region.
        """
        return self.__reg.Type

    @property
    def xml(self):
        """XML data of the region. Only applicable to *Math Region*"""
        if self.type == 1:  # only math type region has mathinterface
            return self.__reg.MathInterface.XML

    @xml.setter
    def xml(self, xml_content):
        if self.type == 1:
            self.__reg.MathInterface.XML = xml_content
        else:
            raise TypeError("Unsupported region typ.")

    @property
    def errmsg(self):
        """Error message of the region. Read-only."""
        if self.type == 1:  # only math type region has mathinterface
            return self.__reg.MathInterface.ErrorMsg

    def locate(self):
        """Locate window to the region.

        :return: None
        """
        self.__wnd.Activate()
        self.__wnd.ScrollToRegion(self.__reg)

