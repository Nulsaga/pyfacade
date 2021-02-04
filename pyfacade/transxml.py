# -*- coding: utf-8 -*-
"""
Late updated on 16/11/2020
-Add new operator supported and customized function definition
-Update docstrings

@author: Qi.Wang
"""

import re
import numpy as np

# region Complied regex pattern for reading expression
# ()
pars = re.compile(r"(?P<oper1>\//|[\^#*])?(?P<par>(?P<init>[!|\\]|[A-Za-z]+)?\((?P<exp>[^()]+)\))(?P<oper2>\//)?",
                  re.DOTALL)
# A^B or A#B
pow_rot = re.compile(r"(?P<var1>&[^&]+?&|[^+\-\*\/\^#]+)(?P<oper>\^|#)(?P<var2>&[^&]+?&|[^+\-\*\/\^#]+)", re.DOTALL)
# AxB or A/B
mult_div = re.compile(r"(?P<var1>&[^&]+?&|(^-)?[^+\-\*\/]+)(?P<oper>\*{1,2}|\//)(?P<var2>&[^&]+?&|[^+\-\*\/]+)",
                      re.DOTALL)
# A+B or A-B
add_min = re.compile(r"(?P<var1>&[^&]+?&|^-?[^+\-]+)(?P<oper>\+|\-)(?P<var2>&[^&]+?&|[^+\-]+)", re.DOTALL)

# () or init()
pars_z = re.compile(r"(?P<init>[!|\\]|[A-Za-z]+)?\((?P<exp>[^()]+)\)", re.DOTALL)
# 0^A, 0#A, A^0, A#0
powrot_z = re.compile(
    r"((?P<z1>(?<![\w_\.\^])0)|(?P<var1>&\d+&|[^+\-\*\/\^#]+))(?P<oper>\^|#)(?(z1)(?P<var2>&\d+&|[^+\-\*\/\^#]+)|(?P<z2>0))",
    re.DOTALL)
# 0xA, 0//A,  A*0, A//0
multdiv_z = re.compile(
    r"((?P<z1>(?<![\w_\.\^])0)|(?P<var1>&\d+&|(^-)?[^+\-\*\/\^#]+))(?P<oper>\*{1,2}|\//)(?(z1)(?P<var2>&\d+&|[^+\-\*\/\^#]+)|(?P<z2>0))",
    re.DOTALL)
# +0, -0
adder_z = re.compile(r"((?P<s>^-?)|[\+|-])0(?(s)\+|)?", re.DOTALL)
# endregion


# region -Making XML block-
# Dictionary: mapping english letter to greek character
greek = {
    'A': u'\u0391',
    'B': u'\u0392',
    'G': u'\u0393',
    'D': u'\u0394',
    'E': u'\u0395',
    'Z': u'\u0396',
    'H': u'\u0397',
    'Q': u'\u0398',
    'I': u'\u0399',
    'K': u'\u039A',
    'L': u'\u039B',
    'M': u'\u039C',
    'N': u'\u039D',
    'X': u'\u039E',
    'O': u'\u039F',
    'P': u'\u03A0',
    'R': u'\u03A1',
    'S': u'\u03A3',
    'T': u'\u03A4',
    'U': u'\u03A5',
    'F': u'\u03A6',
    'C': u'\u03A7',
    'Y': u'\u03A8',
    'W': u'\u03A9',
    'a': u'\u03B1',
    'b': u'\u03B2',
    'g': u'\u03B3',
    'd': u'\u03B4',
    'e': u'\u03B5',
    'z': u'\u03B6',
    'h': u'\u03B7',
    'q': u'\u03B8',
    'i': u'\u03B9',
    'k': u'\u03BA',
    'l': u'\u03BB',
    'm': u'\u03BC',
    'n': u'\u03BD',
    'x': u'\u03BE',
    'o': u'\u03BF',
    'p': u'\u03C0',
    'r': u'\u03C1',
    's': u'\u03C3',
    't': u'\u03C4',
    'u': u'\u03C5',
    'f': u'\u03C6',
    'c': u'\u03C7',
    'y': u'\u03C8',
    'w': u'\u03C9',
}

# Dictionary: mapping operation keywords
Oper = {
    '+': 'plus',
    '-': 'minus',
    '*': 'mult',
    '**': 'mult style="x"',
    '//': 'div',
    '^': 'pow',
    '#': 'nthRoot',
    '!': 'factorial',
    '\\': 'sqrt',
    '|': 'absval',
    '==': 'equal',
    '<': 'lessThan',
    '<=': 'lessOrEqual',
    '>': 'greaterThan',
    '>=': 'greaterOrEqual',
    '!=': 'notEqual'}


# Function: change escaped letter by @ to greek character
def repl_grk(formula_str):
    return re.sub(r'@(?P<letter>[a-zA-Z])', lambda x: greek[x.group('letter')], formula_str)


# Function: add variable statement line / block into xml
def xml_stat(var):
    if re.match(r'^&.+&$', var, re.DOTALL):  # a simple xml <apply> bloc
        return var.strip("&")

    elif re.match(r'^-&.+&$', var, re.DOTALL):  # a xml <apply> block with Negative Sign
        m = ['<ml:apply>', '<ml:neg/>', var.strip("-&"), '</ml:apply>']
        return '\n'.join(m)

    if re.match(r'^[+-]?\d+\.?\d*$', var):  # a number
        return '<ml:real>{0}</ml:real>'.format(var)

    if len(var.split('_')) == 1:  # variable with no indexer

        if re.match(r'^[+-]?[a-zA-Z@]+\d*(\.[\w@]+)*$', var):  # a variable name
            if re.match(r'^-', var):  # var with initial negative sign
                m = ['<ml:apply>', '<ml:neg/>', '<ml:real>', repl_grk(var.strip('-')), '</ml:real>', '</ml:apply>']
                return '\n'.join(m)
            else:
                return '<ml:id>{0}</ml:id>'.format(repl_grk(var))

        elif re.match(r'(^[+-]?\d+\.?\d*)([a-zA-Z@]+\d*(\.[\w@]+)*)$', var):  # number x variable
            sg = re.match(r'(^[+-]?\d+\.?\d*)([a-zA-Z@]+\d*(\.[\w@]+)*)$', var)
            m = ['<ml:apply>', '<ml:mult style="auto-select"/>',
                 '<ml:real>{0}</ml:real>'.format(sg.group(1)),
                 '<ml:id>{0}</ml:id>'.format(repl_grk(sg.group(2))),
                 '</ml:apply>']
            return '\n'.join(m)

    elif len(var.split('_')) == 2:  # 1-d indexer found
        if re.match(r'^-', var):  # var with initial negative sign
            m = ['<ml:apply>', '<ml:neg/>', '<ml:apply>', '<ml:indexer/>']
        else:
            m = ['<ml:apply>', '<ml:indexer/>']

        for v in var.strip('-').split('_'):
            m.append(xml_stat(v))
        m.append('</ml:apply>')

        if re.match(r'^-', var):
            m.append('</ml:apply>')

        return '\n'.join(m)

    elif len(var.split('_')) == 3:  # 2-d indexer found
        v, d1, d2 = var.strip('-').split('_')
        if re.match(r'^-', var):  # var with initial negative sign
            m = ['<ml:apply>', '<ml:neg/>', '<ml:apply>', '<ml:indexer/>']
        else:
            m = ['<ml:apply>', '<ml:indexer/>']

        m.extend([xml_stat(v),
                  '<ml:sequence>',
                  xml_stat(d1),
                  xml_stat(d2),
                  '</ml:sequence>',
                  '</ml:apply>'])

        if re.match(r'^-', var):
            m.append('</ml:apply>')

        return '\n'.join(m)

    else:
        raise ValueError("Invalid or unsupported expression: {0}".format(var))


# Function: create regular <apply> block for 2-parameter operation
def xml_apply(operation, var1, var2):
    m = ['<ml:apply>']

    m.append('<ml:{0}/>'.format(Oper[operation]))

    m.append(xml_stat(var1))

    m.append(xml_stat(var2))

    m.append('</ml:apply>')

    return '&' + '\n'.join(m) + '&'  # add start and end identity '&'


# Function: solve parentheses block
def xml_par(init, expression, oper1, oper2):
    if init:  # special functional parentheses with init

        if init in ['\\', '|']:  # single variable operation
            m = ['<ml:apply>']
            m.append('<ml:{0}/>'.format(Oper[init]))
            m.append(xml_stat(xml_ex(expression)))
            m.append('</ml:apply>')

        elif init == '!':  # single variable operation needs explict parentheses
            m = ['<ml:apply>', '<ml:factorial/>', '<ml:parens>']
            m.append(xml_stat(xml_ex(expression)))
            m.extend(['</ml:parens>', '</ml:apply>'])

        # try to match predefined special operation, otherwise keep 'init()' as regular function
        else:
            try:
                return getattr(soper, init)(*expression.split(','))
            except AttributeError:
                m = ['<ml:apply>']
                m.append(xml_stat(init))
                pars = expression.split(",")
                if len(pars) == 1:  # only 1 parameter
                    m.append(xml_stat(xml_ex(expression)))
                else:
                    m.append('<ml:sequence>')
                    for p in pars:
                        m.append(xml_stat(xml_ex(p)))
                    m.append('</ml:sequence>')
                m.append('</ml:apply>')

    else:  # regular functional parentheses

        if oper1 in ["^", "#", "//"] or (oper2 and oper1 != "*"):  # implict parentheses
            return xml_ex(expression)

        else:  # explict parentheses
            m = ['<ml:parens>']
            m.append(xml_stat(xml_ex(expression)))
            m.append('</ml:parens>')

    return '&' + '\n'.join(m) + '&'  # add start and end identity '&'


# Function: read regular math expression
def xml_ex(math_expression):

    if re.match(r'^&[^&]+&$', math_expression, re.DOTALL):  # check if expression is a pre-parsed xml block
        return math_expression

    match = pars.search(math_expression)

    while match:
        p = xml_par(match.group('init'), match.group('exp'), match.group('oper1'), match.group('oper2'))
        math_expression = math_expression.replace(match.group('par'), p)
        if re.match(r'^&[^&]+&$', math_expression, re.DOTALL):  # check if all translation is complete
            return math_expression
        match = pars.search(math_expression)

    for pa in [pow_rot, mult_div, add_min]:

        match = pa.search(math_expression)

        while match:
            p = xml_apply(match.group('oper'), match.group('var1'), match.group('var2'))
            math_expression = math_expression.replace(match.group(0), p)
            if re.match(r'^&[^&]+&$', math_expression, re.DOTALL):  # check if all translation is complete
                return math_expression
            match = pa.search(math_expression)

    return '&' + xml_stat(math_expression) + '&'


# Function: create a definition xml block      
def xml_define(var_name, expression, ev=False, unit=None):
    """Translate a variable definition expression to MathCAD XML.

    :param var_name: str, name of variable.
    :param expression: str, math expression to be assigned to variable.
    :param ev: bool, evaluate the result of expression in line.
    :param unit: str, unit to be used when present the result, only valid when *ev* = True.
    :return: str, formatted MathCAD XML.
    """

    m = ['''<ml:define xmlns:ml="http://schemas.mathsoft.com/math30">''',
         xml_stat(var_name)]

    if ev:
        m.append('<ml:eval>')

    m.append(xml_stat(xml_ex(str(expression))))

    if ev:
        if unit:
            m.append('<ml:unitOverride>')
            m.append(xml_stat(xml_ex(unit)))
            m.append('</ml:unitOverride>')
        m.append('</ml:eval>')

    m.append('</ml:define>')

    return '\n'.join(m)


# Function: create a evaluation xml block
def xml_eval(var_expression, unit=None):
    """Translate and evaluate a math expression to MathCAD XML.

    :param var_expression: str, math expression to be evaluated
    :param unit: str, unit to be used when presenting the evaluated result.
    :return: str, formatted MathCAD XML.
    """

    m = ['''<ml:eval xmlns:ml="http://schemas.mathsoft.com/math30">''',
         xml_stat(xml_ex(var_expression))]

    if unit:
        m.append('<ml:unitOverride>')
        m.append(xml_stat(xml_ex(unit)))
        m.append('</ml:unitOverride>')

    m.append('</ml:eval>')

    return '\n'.join(m)


# Function: make a xml block include individual expressed
def xml_ind(content):
    """Translate a individual expression or variable to MathCAD XML.

    :param content: str, math expression or variable name.
    :return: str, formatted MathCAD XML.
    """
    ns = ''' xmlns:ml="http://schemas.mathsoft.com/math30">'''  # name space as initial head
    sub_statement = xml_stat(xml_ex(content))
    return sub_statement.replace('>', ns, 1)  # insert initial head to xml string


# Function: create a function definition xml block
def xml_func(func_name, vars, expression):
    """Translate a function definition expression to MathCAD XML.

    :param func_name: str, name of function.
    :param vars: str or list of str, arguments of the function.
    :param expression: str, expression of the function.
    :return: str, formatted MathCAD XML.
    """

    m = ['''<ml:define xmlns:ml="http://schemas.mathsoft.com/math30">''',
         '<ml:function>',
         xml_stat(func_name),
         '<ml:boundVars>']

    if type(vars) == str:  # one var
        m.append(xml_stat(vars))
    else:
        for var in vars:
            m.append(xml_stat(var))

    m.extend(['</ml:boundVars>',
              '</ml:function>',
              xml_stat(xml_ex(str(expression))),
              '</ml:define>'])

    return '\n'.join(m)


# Class: Predefined special operations recognized by xml_par as 'init'
class soper():
    # Predefined special operation: sigma
    @staticmethod
    def sigma(var_name, main_expression, bounds1, bounds2):
        m = ['<ml:apply>', '<ml:summation/>',
             '<ml:lambda>',
             '<ml:boundVars>', xml_stat(var_name), '</ml:boundVars>',
             xml_stat(xml_ex(str(main_expression))),
             '</ml:lambda>', '<ml:bounds>',
             xml_stat(xml_ex(str(bounds1))), xml_stat(xml_ex(str(bounds2))),
             '</ml:bounds>', '</ml:apply>']

        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined special operation: iterated product
    @staticmethod
    def iterp(var_name, main_expression, bounds1, bounds2):
        m = ['<ml:apply>', '<ml:product/>',
             '<ml:lambda>',
             '<ml:boundVars>', xml_stat(var_name), '</ml:boundVars>',
             xml_stat(xml_ex(str(main_expression))),
             '</ml:lambda>', '<ml:bounds>',
             xml_stat(xml_ex(str(bounds1))), xml_stat(xml_ex(str(bounds2))),
             '</ml:bounds>', '</ml:apply>']

        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined special operation: definite integral
    @staticmethod
    def integral(var_name, main_expression, bounds1, bounds2):
        m = ['<ml:apply>', '<ml:integral auto-algorithm="true" algorithm="adaptive"/>',
             '<ml:lambda>',
             '<ml:boundVars>', xml_stat(var_name), '</ml:boundVars>',
             xml_stat(xml_ex(str(main_expression))),
             '</ml:lambda>', '<ml:bounds>',
             xml_stat(xml_ex(str(bounds1))), xml_stat(xml_ex(str(bounds2))),
             '</ml:bounds>', '</ml:apply>']

        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined special operation: matrix
    @staticmethod
    def matrix(rows, cols, *vars):
        m = [f'''<ml:matrix rows="{rows}" cols="{cols}">''']
        for v in vars:
            m.append(xml_stat(xml_ex(v)))
        for i in range(int(rows)*int(cols)-len(vars)):  # fill undefined items with 0
            m.append('<ml:real>0</ml:real>')
        m.append('</ml:matrix>')

        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined special operation: determinant
    @staticmethod
    def determ(matrix):
        m = ['<ml:apply>',
             '<ml:determinant/>',
             xml_stat(xml_ex(matrix)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined special operation: transpose
    @staticmethod
    def transpose(matrix):
        m = ['<ml:apply>',
             '<ml:transpose/>',
             xml_stat(xml_ex(matrix)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined special operation: cross product
    @staticmethod
    def crossp(matrix_1, matrix_2):
        m = ['<ml:apply>',
             '<ml:crossProduct/>',
             xml_stat(xml_ex(matrix_1)),
             xml_stat(xml_ex(matrix_2)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: local variable definition
    @staticmethod
    def locd(var, expression):
        m = ['<ml:localDefine>',
             xml_stat(xml_ex(var)),
             xml_stat(xml_ex(expression)),
             '</ml:localDefine>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: logical operation AND
    @staticmethod
    def lgand(var1, var2):
        m = ['<ml:apply>', '<ml:and/>',
             xml_stat(xml_ex(var1)),
             xml_stat(xml_ex(var2)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: logical operation OR
    @staticmethod
    def lgor(var1, var2):
        m = ['<ml:apply>', '<ml:or/>',
             xml_stat(xml_ex(var1)),
             xml_stat(xml_ex(var2)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: logical operation NOT
    @staticmethod
    def lgnot(var):
        m = ['<ml:apply>', '<ml:not/>',
             xml_stat(xml_ex(var)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: logical compare, =, >, >=, <, <=, !=
    @staticmethod
    def lgcp(operation, var1, var2):
        m = ['<ml:apply>', f'<ml:{Oper[operation]}/>',
             xml_stat(xml_ex(var1)),
             xml_stat(xml_ex(var2)),
             '</ml:apply>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: IF statement
    @staticmethod
    def pgif(condition, action):
        if action == 'pgbreak':
            act = '<ml:break/>'
        elif action == 'pgcontinue':
            act = '<ml:continue/>'
        else:
            act = xml_stat(xml_ex(action))
        m = ['<ml:ifThen>',
             xml_stat(xml_ex(condition)),
             act,
             '</ml:ifThen>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: ELSE statement
    @staticmethod
    def pgelse(action):
        if action == 'pgbreak':
            act = '<ml:break/>'
        elif action == 'pgcontinue':
            act = '<ml:continue/>'
        else:
            act = xml_stat(xml_ex(action))
        m = ['<ml:otherwise>',
             act,
             '</ml:otherwise>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: FOR loop
    @staticmethod
    def pgfor(loop_var, start, step1, end, action):
        if action == 'pgbreak':
            act = '<ml:break/>'
        elif action == 'pgcontinue':
            act = '<ml:continue/>'
        else:
            act = xml_stat(xml_ex(action))

        m = ['<ml:for>',
             xml_stat(xml_ex(loop_var)),
             '<ml:range>',
             '<ml:sequence>',
             xml_stat(xml_ex(start)),
             xml_stat(xml_ex(step1)),
             '</ml:sequence>',
             xml_stat(xml_ex(end)),
             '</ml:range>',
             act,
             '</ml:for>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: WHILE loop
    @staticmethod
    def pgwhile(condition, action):
        if action == 'pgbreak':
            act = '<ml:break/>'
        elif action == 'pgcontinue':
            act = '<ml:continue/>'
        else:
            act = xml_stat(xml_ex(action))
        m = ['<ml:while>',
             xml_stat(xml_ex(condition)),
             act,
             '</ml:while>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: RETURN
    @staticmethod
    def pgreturn(var):
        m = ['<ml:return>',
             xml_stat(xml_ex(var)),
             '</ml:return>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

    # Predefined programing operation: Error Handle
    @staticmethod
    def pgtry(statment, action_on_error):
        if action_on_error == 'pgbreak':
            act = '<ml:break/>'
        elif action_on_error == 'pgcontinue':
            act = '<ml:continue/>'
        else:
            act = xml_stat(xml_ex(action_on_error))
        m = ['<ml:tryCatch>',
             xml_stat(xml_ex(statment)),
             act,
             '</ml:tryCatch>']
        return '&' + '\n'.join(m) + '&'  # add start and end identity '&'

# endregion


# region =================== TOOLBOX ==========================

# Function: Substitution variables in string expression
def xml_sub(math_expression, subdict, simp=True):
    """Substitute variables in provided expression according to specified dict.

    :param math_expression: str, original expression.
    :param subdict: dict, a mapping dict for substitution in the form of {var1: value1, var2: value2...}
    :param simp: bool, simplify the expression by removing any item equals to 0 after substitution.
    :return: str, new expression with specified variables substituted
    """

    # internal function for simplification
    def simplify(sub_exp):
        for pa in [powrot_z, multdiv_z]:
            mat = pa.search(sub_exp)
            while mat:
                if mat.group('oper') == "#" and mat.group('z1'):  # invalid case 0#A
                    raise ValueError("Invalid zero operation: 0th root of a variable")
                if mat.group('oper') == "//" and mat.group('z2'):  # invalid case A//0
                    raise ValueError("Invalid zero operation: zero divider")
                sub_exp = sub_exp.replace(mat.group(0), "0")  # replace the matched part by 0
                mat = pa.search(sub_exp)
        # delete 0 adder
        sub_exp = adder_z.sub("", sub_exp)
        return "0" if sub_exp == "" else sub_exp

    # replace according to input dict
    ex = math_expression
    for p in subdict:
        ex = re.sub(f"(?<![\w_\.]){p}(?![\w_\.])", str(subdict[p]), ex)

    # return the new expression directly if no need cleaning zero items
    if not simp:
        return ex

    # solve the parens part
    rec = {}
    count = 0
    match = pars_z.search(ex)
    while match:
        inpar = simplify(match.group('exp'))  # simplyfy the content in parens
        if (inpar == "0" or inpar == "-0") and (match.group('init') in ['\\', '|'] or not match.group('init')):
            # case (0), \\(0), |(0)
            ex = ex.replace(match.group(0), "0")  # replace the whole part as 0
        else:
            ex = ex.replace(match.group(0), f"&{count}&")  # represent the whole part part by '&n&'
            if match.group('init'):  # case init(...)
                rec[count] = match.group(0).replace(match.group('exp'), inpar)  # recode whole part
            else:
                if re.search(r"[\+\-\*\//\^\#]", inpar):  # case (A oper B...)
                    rec[count] = f"({inpar})"  # recode inner part with parens
                else:  # case (A)
                    rec[count] = inpar  # recode inner part
            count += 1
        match = pars_z.search(ex)

    # simplify the latest expression without parens
    ex = simplify(ex)

    # replace back the represented part
    for i in range(count, 0, -1):
        ex = ex.replace(f"&{i - 1}&", rec[i - 1])

    return ex


# Function: translate a numpy array into a xml statement block
def to_matrix(input_array):
    """Produce a statement block of matrix according to input array.

    :param input_array: array_like as matrix data.
    :return: str, pre-formatted XML paragraph.
    """
    arr = np.asarray(input_array)
    content_str = str(list(arr.flatten(order='F'))).strip('[]').replace(' ', '')
    rows = arr.shape[0]
    try:
        columns = arr.shape[1]
    except IndexError:
        columns = 1
    return xml_ex(f'matrix({rows},{columns},{content_str})')  # output as '& + expression + &'


# Function: create a programing xml statement block
def xml_prog(statements):
    """Produce a statement block of in-line programing according to input string.

    :param statements: list of str, programing statements. each str represents a line of statement in programing.
    :return: str, pre-formatted XML paragraph.
    """

    m = ['<ml:program>']

    for s in statements:
        m.append(xml_stat(xml_ex(s)))

    m.append('</ml:program>')

    return '&' + '\n'.join(m) + '&'  # add start and end identity '&'


# Class: Special formatted string of expression for xml parser
class Xexpr():
    """Formatted string of math expression.

    :param expression: str, original math expression. Invalid blank in string will be removed automatically.
    :param alias: alias of the expression.
    """
    def __init__(self, expression, alias=None):
        self.__ex = expression.replace(" ", "")  # delete blank in expression
        self.alias = str(alias)

    def __bool__(self):
        """Return **False** when instance represents a empty expression ('') or '0'."""
        if (not self.__ex) or (self.__ex == "0"):
            return False
        else:
            return True

    @property
    def ex(self):
        """Math expression string."""
        return self.__ex

    @ex.setter
    def ex(self, new_expr):
        self.__ex = new_expr.replace(" ", "")

    def sub(self, subdict, inplace=True, simp=False):
        """Substitute variables in expression according to specified dict.

        :param subdict: dict, a mapping dict for substitution in the form of {var1: value1, var2: value2...}
        :param inplace: bool, overwrite the expression.
        :param simp: bool, simplify the expression by removing any item equals to 0 after substitution.
        :return: A new ``Xexpr`` object with variables substituted if *inplace* = False. Else, return None.
        """
        if inplace:
            self.__ex = xml_sub(self.__ex, subdict=subdict, simp=simp)
        else:
            return Xexpr(xml_sub(self.__ex, subdict=subdict, simp=simp), alias=self.alias)

    def zero(self, vars, inplace=True, simp=True):
        """Set the specified variables in expression to Zero.

        :param vars: str or list of str, variable name to be set to 0.
        :param inplace: bool, overwrite the  expression.
        :param simp: bool, simplify the expression by removing any item equals to 0 after substitution.
        :return: A new ``Xexpr`` Object with the variables set to 0 if *inplace* = False. Else, return None.
        """
        if type(vars) == str:  # one var name
            vdict = {vars: 0}
        else:
            vdict = {v: 0 for v in vars}
        if inplace:
            self.__ex = xml_sub(self.__ex, subdict=vdict, simp=simp)
        else:
            return Xexpr(xml_sub(self.__ex, subdict=vdict, simp=simp), alias=self.alias)

    def inject(self, dataframe, alias_col="main", expr_col="expression", alias=None):
        """Insert the math expression to specified data frame by matching the alias.

        :param dataframe: pandas.DataFrame, target data frame to write expression in.
        :param alias_col: str, column in data frame where to match the alias.
        :param expr_col: str, target column in data frame to insert expression in
        :param alias: str, alias to be matched.
        :return: None
        """
        if not alias:
            if self.alias:
                alias = self.alias
            else:
                raise ValueError("Alias is not defined")
        dataframe.loc[dataframe[alias_col] == alias, expr_col] = self.__ex


# endregion


