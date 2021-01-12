# -*- coding: utf-8 -*-
"""
Pyeng for structural analysis, Based on CoreMod-Beta 1.1

(Last Update on 12/11/2020): add function of translation releasing assignment, update deflection-related function into
3 dimensions. Update the docstring

ver beta_1.2
@author: qi.wang
"""

import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv
from os import getcwd


# region <Math Functions>
# Function: matrix transform beam matrix from global coordinate system to local
def mtrans(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, s, 0, 0, 0, 0],
                     [-s, c, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, c, s, 0],
                     [0, 0, 0, -s, c, 0],
                     [0, 0, 0, 0, 0, 1],
                     ])


# Function: matrix transform vectors from local coordinate system to global
def vtrans(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, ],
                     [s, c, ],
                     ])


# Function: construct matrix of stiffness of a beam element
def stiffb2(E, A, I, L, start_release=(0, 0), end_release=(0, 0)):
    """ matrix of stiffness of beam element"""
    ka = 0 if (start_release[0] or end_release[0]) else E * A / L  # no axial stiffness if translate released at any end

    if (not start_release[1]) and (not end_release[1]):
        k_ii = np.array([[ka, 0, 0], [0, 12 * E * I / L ** 3, 6 * E * I / L ** 2],
                         [0, 6 * E * I / L ** 2, 4 * E * I / L]])
        k_ij = np.array([[-ka, 0, 0], [0, -12 * E * I / L ** 3, 6 * E * I / L ** 2],
                         [0, -6 * E * I / L ** 2, 2 * E * I / L]])
        k_ji = np.array([[-ka, 0, 0], [0, -12 * E * I / L ** 3, -6 * E * I / L ** 2],
                         [0, 6 * E * I / L ** 2, 2 * E * I / L]])
        k_jj = np.array([[ka, 0, 0], [0, 12 * E * I / L ** 3, -6 * E * I / L ** 2],
                         [0, -6 * E * I / L ** 2, 4 * E * I / L]])

    elif (start_release[1]) and (not end_release[1]):
        k_ii = np.array([[ka, 0, 0], [0, 3 * E * I / L ** 3, 0], [0, 0, 0]])
        k_ij = np.array([[-ka, 0, 0], [0, -3 * E * I / L ** 3, 3 * E * I / L ** 2], [0, 0, 0]])
        k_ji = np.array([[-ka, 0, 0], [0, -3 * E * I / L ** 3, 0], [0, 3 * E * I / L ** 2, 0]])
        k_jj = np.array([[ka, 0, 0], [0, 3 * E * I / L ** 3, -3 * E * I / L ** 2],
                         [0, -3 * E * I / L ** 2, 3 * E * I / L]])

    elif (not start_release[1]) and (end_release[1]):
        k_ii = np.array([[ka, 0, 0], [0, 3 * E * I / L ** 3, 3 * E * I / L ** 2],
                         [0, 3 * E * I / L ** 2, 3 * E * I / L]])
        k_ij = np.array([[-ka, 0, 0], [0, -3 * E * I / L ** 3, 0], [0, -3 * E * I / L ** 2, 0]])
        k_ji = np.array([[-ka, 0, 0], [0, -3 * E * I / L ** 3, -3 * E * I / L ** 2], [0, 0, 0]])
        k_jj = np.array([[ka, 0, 0], [0, 3 * E * I / L ** 3, 0], [0, 0, 0]])

    else:
        k_ii = np.array([[ka, 0, 0], [0, 0, 0], [0, 0, 0]])
        k_ij = np.array([[-ka, 0, 0], [0, 0, 0], [0, 0, 0]])
        k_ji = np.array([[-ka, 0, 0], [0, 0, 0], [0, 0, 0]])
        k_jj = np.array([[ka, 0, 0], [0, 0, 0], [0, 0, 0]])

    return np.block([[k_ii, k_ij], [k_ji, k_jj]])


# Function: translate uniformly distributed load to equivalent node load
def equivload(udl, ual, L, start_release=(0, 0), end_release=(0, 0)):
    """ change uniformly distributed load to equivalent node load"""
    if (not start_release[0]) and (not end_release[0]):
        N_start = ual * L / 2
        N_end = ual * L / 2
    elif (start_release[0]) and (not end_release[0]):
        N_start = 0
        N_end = ual * L
    elif (not start_release[0]) and (end_release[0]):
        N_start = ual * L
        N_end = 0
    else:
        raise ValueError("Unstable System. Axial restrains are released at both end")

    if (not start_release[1]) and (not end_release[1]):
        return np.array([N_start, udl * L / 2, udl * L * L / 12, N_end, udl * L / 2, -udl * L * L / 12])

    elif (start_release[1]) and (not end_release[1]):
        return np.array([N_start, 3 * udl * L / 8, 0, N_end, 5 * udl * L / 8, -udl * L * L / 8])

    elif (not start_release[1]) and (end_release[1]):
        return np.array([N_start, 5 * udl * L / 8, udl * L * L / 8, N_end, 3 * udl * L / 8, 0])

    else:
        return np.array([N_start, udl * L / 2, 0, N_end, udl * L / 2, 0])


# Function: shape function of a beam element
def shapeb2(x, L, Axial=False):
    """ shape function of beam element"""
    N_iu = 1 - x / L
    N_iv = 1 - 3 * x ** 2 / L ** 2 + 2 * x ** 3 / L ** 3
    N_ia = x - 2 * x ** 2 / L + x ** 3 / L ** 2
    N_ju = x / L
    N_jv = 3 * x ** 2 / L ** 2 - 2 * x ** 3 / L ** 3
    N_ja = -x ** 2 / L + x ** 3 / L ** 2
    if Axial:
        return np.array([[N_iu, 0, 0, N_ju, 0, 0],
                         [0, N_iv, N_ia, 0, N_jv, N_ja]],
                        dtype=object)
    else:
        return np.array([0, N_iv, N_ia, 0, N_jv, N_ja], dtype=object)
# endregion


# Class: 2D Beam Element
class Beam2():
    """Finite element represents a two-dimensional beam in structural model.

    :param start_node: int, index of start node of specified beam in node_list.
    :param end_node: int, index of end node of specified beam in node_list.
    :param section_area: float, cross section area of beam, unit=mm :superscript:`2`.
    :param moment_of_inertia: float, moment of inertia of beam section, unit=mm :superscript:`4`.
    :param modulus_of_elasticity: float, modulus of elasticity of beam material, unit=N/mm :superscript:`2`.
    :param node_list: nested list as [[x0,y0],[x1,y1],...], list of coordinates of all nodes in structural model.
            Unit=mm.
    """

    def __init__(self, start_node, end_node, section_area, moment_of_inertia, modulus_of_elasticity, node_list):
        self.nid = (start_node, end_node)
        self.start = node_list[start_node]
        self.end = node_list[end_node]
        self.Ia = moment_of_inertia
        self.A = section_area
        self.E = modulus_of_elasticity
        # Below set some default values
        self.start_rels = (0, 0)  # (translation, rotation)
        self.end_rels = (0, 0)
        self.udl = 0
        self.ual = 0
        self.__r = np.zeros(6)
        self.__D = None
        self.__edf = None

    def eval(self):
        """Establish necessary structural parameters for finite element analyze.

        .. note:: Run this method once after initialization or any changing on initial parameters."""
        self.__L = math.sqrt((self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2)
        self.__ang = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.__MT = mtrans(self.angle)
        self.__Mk = stiffb2(self.E, self.A, self.Ia, self.__L, self.start_rels, self.end_rels)
        self.__MK = np.transpose(self.__MT) @ self.__Mk @ self.__MT
        self.__r = equivload(self.udl, self.ual, self.__L, self.start_rels, self.end_rels)

    @property
    def length(self):
        """Length of beam. unit=mm."""
        return self.__L

    @property
    def angle(self):
        """Angle between beam axial direction and +x direction, unit=rad."""
        return self.__ang

    @property
    def T(self):
        """The translate matrix from global coordinates to beam local coordinates."""
        return self.__MT

    @property
    def K(self):
        """The standard stiffness matrix of beam element"""
        return self.__MK

    @property
    def nodef(self):
        """Equivalent applied force at node, in the form of [N_start, V_start, M_start, N_end, V_end, M_end],
        unit=N or N*mm
        """
        return self.__r

    # ------------------- for SOLVER BLOCK ----------------------------

    def fill_GK(self, Global_Matrix):
        """ Insert element stiffness into *Global Stiffness Matrix* of the model.

        :param Global_Matrix: numpy.ndarray, target matrix.
        :return: None
        """
        MK = self.__MK
        i, j = self.nid
        Global_Matrix[3 * i:3 * i + 3, 3 * i:3 * i + 3] += MK[0:3, 0:3]
        Global_Matrix[3 * i:3 * i + 3, 3 * j:3 * j + 3] += MK[0:3, 3:6]
        Global_Matrix[3 * j:3 * j + 3, 3 * i:3 * i + 3] += MK[3:6, 0:3]
        Global_Matrix[3 * j:3 * j + 3, 3 * j:3 * j + 3] += MK[3:6, 3:6]

    def fill_GR(self, Global_Matrix):
        """Insert equivalent node into *Global Node Force Matrix* of the model.

        :param Global_Matrix: numpy.ndarray, target matrix.
        :return: None
        """
        R = np.transpose(self.__MT) @ self.__r
        i, j = self.nid
        Global_Matrix[3 * i:3 * i + 3] += R[0:3]
        Global_Matrix[3 * j:3 * j + 3] += R[3:6]

    # ------------------ for RESULTS OUTPUT -----------------------------

    def solve(self, Global_D):
        """Import solved results from global model to beam element.

        :param Global_D: numpy.ndarray, target matrix
        :return: None
        """
        i, j = self.nid
        self.__D = np.append(Global_D[3 * i:3 * i + 3], Global_D[3 * j:3 * j + 3])
        self.__edf = self.__MT @ (self.__MK @ self.__D - np.transpose(self.__MT) @ self.__r)

    def endforce(self):
        """Output solved end force of beam element in the form of [[N_start, V_start, M_start, N_end, V_end, M_end]
        unit=N or N*mm"""
        return self.__edf

    def enddisp(self):
        """Output local displacement at both end of beam element in the form of
        [axial_start, translation_start, rotation_start, axial_end, translation_end, rotation_end], unit=mm or rad"""

        ld = self.__MT @ self.__D
        L = self.__L
        E = self.E
        I = self.Ia
        A = self.A

        if self.end_rels[1]:  # rotation release at end node
            d1 = np.delete(ld, 5, 0)
            k21 = np.array([0, 6 * E * I / L ** 2, 2 * E * I / L, 0, -6 * E * I / L ** 2])
            m2_udl = self.udl * L ** 2 / 12
            ld[5] = (self.__edf[5] - m2_udl - k21.dot(d1)) / (4 * E * I / L)

        if self.start_rels[1]:  # rotation release at start node
            d2 = np.delete(ld, 2, 0)
            k12 = np.array([0, 6 * E * I / L ** 2, 0, -6 * E * I / L ** 2, 2 * E * I / L])
            m1_udl = -self.udl * L ** 2 / 12
            ld[2] = (self.__edf[2] - m1_udl - k12.dot(d2)) / (4 * E * I / L)

        if self.end_rels[0]:  # translation release at end node
            ld[3] = (self.__edf[3]+self.ual*L/2-ld[0]*(-E*A/L))/(E*A/L)

        if self.start_rels[0]:  # translation release at start node
            ld[0] = (self.__edf[0]+self.ual*L/2-ld[3]*(-E*A/L))/(E*A/L)

        return ld

    def get_N(self, samp, *samps):
        """Calculate axial force at specified position on beam element.

        :param samp: float, relative position along the element length, from 0 to 1.
        :param samps:  float, other relative positions along the element length, from 0 to 1.
        :return: float or list. Unit=N, Positive value is tension.
        """
        sN = self.__edf[0]
        if not samps:
            return -(sN + self.ual * samp * self.__L)
        else:
            return [-(sN + self.ual * x * self.__L) for x in ([samp] + list(samps))]

    def extre_N(self):
        """Find the extreme value of axial force on the element.

        :return: tuple ((position_occurs_min, min_value), (position_occurs_max, max_value)), unit of axial force=N.
        """
        res_min = minimize(self.get_N, 0, bounds=((0, 1),))
        res_max = minimize(lambda x: -self.get_N(x), 0, bounds=((0, 1),))
        return (res_min.x[0], self.get_N(res_min.x[0])), (res_max.x[0], self.get_N(res_max.x[0]))

    def plot_N(self, sn=5, scale=1, deci=4):
        """Show axial force curve of the element.

        :param sn: int. Number of slice when plotting the curve. More slice provides more smooth appearance of curve.
        :param scale: float, zoom-out scales when plotting the curve.
        :param deci: int, number of decimal places kept when plotting values on curve.
        :return: tuple (min_axial_force, max_axial_force), unit=N.
        """
        # Draw Aixal force Curvature
        a = np.linspace(0, 1, sn)
        u = list(np.linspace(0, self.__L, sn))
        v = list(map(lambda x: x / scale, self.get_N(*a)))
        uv_list = zip([0] + u + [self.__L], [0] + v + [0])
        xy_list = np.array(list(map(lambda x: vtrans(self.__ang) @ x, uv_list)))
        x, y = list(zip(*[c + np.array(self.start) for c in xy_list]))
        plt.plot(x, y, color='coral')
        # Markup the maximum value
        exN = self.extre_N()
        u_max = exN[1][0] * self.__L
        v_max = exN[1][1] / scale
        uv_max = [u_max, v_max]
        xy_max = vtrans(self.__ang) @ uv_max + np.array(self.start)
        plt.text(*xy_max, str(round(v_max * scale, deci)))
        # Markup the min value
        u_min = exN[0][0] * self.__L
        v_min = exN[0][1] / scale
        uv_min = [u_min, v_min]
        xy_min = vtrans(self.__ang) @ uv_min + np.array(self.start)
        plt.text(*xy_min, str(round(v_min * scale, deci)))
        # return extreme value
        return exN[0][1], exN[1][1]

    def get_V(self, samp, *samps):
        """Calculate shear force at specified position on beam element.

        :param samp: float, relative position along the element length, from 0 to 1.
        :param samps:  float, other relative positions along the element length, from 0 to 1.
        :return: float or list. Unit=N.
        """
        sV = self.__edf[1]
        if not samps:
            return sV + self.udl * samp * self.__L
        else:
            return [sV + self.udl * x * self.__L for x in ([samp] + list(samps))]

    def extre_V(self):
        """Find the extreme value of shear force on the element.

        :return: tuple ((position_occurs_min, min_value), (position_occurs_max, max_value)), unit of shear force=N.
        """
        res_min = minimize(self.get_V, 0, bounds=((0, 1),))
        res_max = minimize(lambda x: -self.get_V(x), 0, bounds=((0, 1),))
        return (res_min.x[0], self.get_V(res_min.x[0])), (res_max.x[0], self.get_V(res_max.x[0]))

    def plot_V(self, sn=5, scale=1, deci=4):
        """Show shear force curve of the element.

        :param sn: int. Number of slice when plotting the curve. More slice provides more smooth appearance of curve.
        :param scale: float, zoom-out scales when plotting the curve.
        :param deci: int, number of decimal places kept when plotting values on curve.
        :return: tuple (min_shear_force, max_shear_force), unit=N.
        """
        # Draw Shear Curvature
        a = np.linspace(0, 1, sn)
        u = list(np.linspace(0, self.__L, sn))
        v = list(map(lambda x: x / scale, self.get_V(*a)))
        uv_list = zip([0] + u + [self.__L], [0] + v + [0])
        xy_list = np.array(list(map(lambda x: vtrans(self.__ang) @ x, uv_list)))
        x, y = list(zip(*[c + np.array(self.start) for c in xy_list]))
        plt.plot(x, y, 'c')
        # Markup the maximum value
        exV = self.extre_V()
        u_max = exV[1][0] * self.__L
        v_max = exV[1][1] / scale
        uv_max = [u_max, v_max]
        xy_max = vtrans(self.__ang) @ uv_max + np.array(self.start)
        plt.text(*xy_max, str(round(v_max * scale, deci)))
        # Markup the min value
        u_min = exV[0][0] * self.__L
        v_min = exV[0][1] / scale
        uv_min = [u_min, v_min]
        xy_min = vtrans(self.__ang) @ uv_min + np.array(self.start)
        plt.text(*xy_min, str(round(v_min * scale, deci)))
        # return extreme value
        return exV[0][1], exV[1][1]

    def get_M(self, samp, *samps):
        """Calculate bending moment at specified position on beam element.

        :param samp: float, relative position along the element length, from 0 to 1.
        :param samps:  float, other relative positions along the element length, from 0 to 1.
        :return: float or list. Unit=N*mm.
        """
        sV = self.__edf[1]
        sM = self.__edf[2]
        if not samps:
            return sM - sV * samp * self.__L - 0.5 * self.udl * (samp * self.__L) ** 2
        else:
            return [sM - sV * x * self.__L - 0.5 * self.udl * (x * self.__L) ** 2 for x in ([samp] + list(samps))]

    def extre_M(self):
        """Find the extreme value of bending moment on the element.

        :return: tuple ((position_occurs_min, min_value), (position_occurs_max, max_value)), unit of moment = N*mm.
        """
        res_min = minimize(self.get_M, 0.5, bounds=((0, 1),))
        res_max = minimize(lambda x: -self.get_M(x), 0.5, bounds=((0, 1),))
        return (res_min.x[0], self.get_M(res_min.x[0])), (res_max.x[0], self.get_M(res_max.x[0]))

    def plot_M(self, sn=10, scale=1, deci=4):
        """Show moment curve of the element.

        :param sn: int. Number of slice when plotting the curve. More slice provides more smooth appearance of curve.
        :param scale: float, zoom-out scales when plotting the curve.
        :param deci: int, number of decimal places kept when plotting values on curve.
        :return: tuple (min_bending_moment, max_bending_moment), unit=N*mm.
        """
        # Draw Moment Curvature
        a = np.linspace(0, 1, sn)
        u = list(np.linspace(0, self.__L, sn))
        v = list(map(lambda x: x / scale, self.get_M(*a)))
        uv_list = zip([0] + u + [self.__L], [0] + v + [0])
        xy_list = np.array(list(map(lambda x: vtrans(self.__ang) @ x, uv_list)))
        x, y = list(zip(*[c + np.array(self.start) for c in xy_list]))
        plt.plot(x, y, 'g')
        # Markup the maximum value
        exM = self.extre_M()
        u_max = exM[1][0] * self.__L
        v_max = exM[1][1] / scale
        uv_max = [u_max, v_max]
        xy_max = vtrans(self.__ang) @ uv_max + np.array(self.start)
        plt.text(*xy_max, str(round(v_max * scale, deci)))
        # Markup the min value
        u_min = exM[0][0] * self.__L
        v_min = exM[0][1] / scale
        uv_min = [u_min, v_min]
        xy_min = vtrans(self.__ang) @ uv_min + np.array(self.start)
        plt.text(*xy_min, str(round(v_min * scale, deci)))
        # return extreme value
        return exM[0][1], exM[1][1]

    def get_Defy(self, samp, *samps):
        """Calculate transverse deflection at specified position on beam element.

        :param samp: float, relative position along the element length, from 0 to 1.
        :param samps:  float, other relative positions along the element length, from 0 to 1.
        :return: float or list. Unit=mm.
        """
        if not samps:
            return (shapeb2(samp * self.__L, self.__L).dot(self.enddisp())
                    + self.udl * ((samp * self.__L) ** 2) * (self.__L - samp * self.__L) ** 2 / (
                                24 * self.E * self.Ia))
        else:
            return [shapeb2(x * self.__L, self.__L).dot(self.enddisp())
                    + self.udl * ((x * self.__L) ** 2) * (self.__L - x * self.__L) ** 2 / (
                                24 * self.E * self.Ia)
                    for x in ([samp] + list(samps))]

    def get_Defx(self, samp, *samps):
        """Calculate longitudinal displacement at specified position on beam element.

        :param samp: float, relative position along the element length, from 0 to 1.
        :param samps:  float, other relative positions along the element length, from 0 to 1.
        :return: float or list. Unit=mm.
        """
        if not samps:
            return (shapeb2(samp * self.__L, self.__L, Axial=True).dot(self.enddisp())[0]
                    + self.ual * (samp * self.__L)*(self.__L-samp * self.__L) / (2 * self.A * self.E))
        else:
            return [shapeb2(x * self.__L, self.__L, Axial=True).dot(self.enddisp())[0]
                    + self.ual * (x * self.__L)*(self.__L-x * self.__L) / (2 * self.A * self.E)
                    for x in ([samp] + list(samps))]

    def get_Defxy(self, samp, *samps, resultant=True):
        """ Calculate resultant displacement at specified position on beam element.
        return longitudinal deflection and transverse deflection respectively when resultant=False

        :param samp: float, relative position along the element length, from 0 to 1.
        :param samps: float, other relative positions along the element length, from 0 to 1.
        :param resultant: bool, return resultant deflection
        :return: float or list of float if *resultant* =True, ``numpy.ndarray`` [def_x, def_y] or list of
                ``numpy.ndarray`` if *resultant* =False. Unit=mm
        """
        if not samps:
            dxy = (shapeb2(samp * self.__L, self.__L, Axial=True).dot(self.enddisp())
                   + np.array([self.ual * (samp * self.__L)*(self.__L-samp * self.__L) / (2 * self.A * self.E),
                               self.udl * ((samp * self.__L) ** 2) * (self.__L - samp * self.__L) ** 2 / (
                               24 * self.E * self.Ia)]))
            return np.linalg.norm(dxy) if resultant else dxy

        else:
            dxys = [shapeb2(x * self.__L, self.__L, Axial=True).dot(self.enddisp())
                    + np.array([self.ual * (x * self.__L)*(self.__L-x * self.__L) / (2 * self.A * self.E),
                               self.udl * ((x * self.__L) ** 2) * (self.__L - x * self.__L) ** 2 / (
                               24 * self.E * self.Ia)])
                    for x in ([samp] + list(samps))]
            return [np.linalg.norm(d) for d in dxys] if resultant else dxys

    def extre_Defy(self):
        """Find the extreme value of transverse deflection on the element.

        :return: tuple ((position_occurs_min, min_value), (position_occurs_max, max_value)), unit of deflection = mm.
        """
        res_min = minimize(self.get_Defy, 0.5, bounds=((0, 1),))
        res_max = minimize(lambda x: -self.get_Defy(x), 0.5, bounds=((0, 1),))
        return (res_min.x[0], self.get_Defy(res_min.x[0])), (res_max.x[0], self.get_Defy(res_max.x[0]))

    def extre_Defx(self):
        """Find the extreme value of longitudinal displacement of the element.

        :return: tuple ((position_occurs_min, min_value), (position_occurs_max, max_value)), unit of displacement = mm.
        """
        start_def = self.get_Defx(0)  # extreme longitudinal deflection occurs at either start or end node
        end_def = self.get_Defx(1)
        if start_def <= end_def:  # output (min., max.)
            return (0, start_def), (1, end_def)
        else:
            return (1, end_def), (0, start_def)

    def extre_Defxy(self):
        """Find the extreme value of resultant displacement on the element.

        :return: tuple ((position_occurs_min, min_value), (position_occurs_max, max_value)), unit of displacement = mm.
        """
        res_min = minimize(self.get_Defxy, 0.5, bounds=((0, 1),))
        res_max = minimize(lambda x: -self.get_Defxy(x), 0.5, bounds=((0, 1),))
        return (res_min.x[0], self.get_Defxy(res_min.x[0])), (res_max.x[0], self.get_Defxy(res_max.x[0]))

    def plot_Def(self, sn=10, scale=1, deci=4, def_type="transverse"):
        """Show deflection curve of the element.

        :param sn: int, Number of slice when plotting the curve. More slice provides more smooth appearance of curve.
        :param scale: float, zoom-out scales when plotting the curve.
        :param deci: int, number of decimal places kept when plotting values on curve.
        :param def_type: int or string, type of deflection to be shown, acceptable indicator:

                + *0, 'xy', 'resultant'* - resultant deflection.
                + *1, 'x', 'axial'* - longitudinal deformation.
                + *2, 'y', 'transverse'* - transverse deflection.

        :return: tuple (min_deflection, max_deflection), unit=mm.
        """
        # Draw Deflection Curvature
        a = np.linspace(0, 1, sn)  # interpolate points
        uv0 = np.zeros((sn, 2))
        uv0[:, 0] = np.linspace(0, self.__L, sn)  # original coordinates of interpolate points
        uv_list = np.array(self.get_Defxy(*a, resultant=False)) / scale + uv0  # displaced coordinates
        xy_list = np.array(list(map(lambda x: vtrans(self.__ang) @ x, uv_list)))  # convert to global XY
        x, y = list(zip(*[c + np.array(self.start) for c in xy_list]))
        plt.plot(x, y, '--m')
        # Get extreme value
        if def_type in [2, "y", "transverse"]:
            exDef = self.extre_Defy()
        elif def_type in [1, "x", "axial"]:
            exDef = self.extre_Defx()
        elif def_type in [0, "xy", "resultant"]:
            exDef = self.extre_Defxy()
        else:
            raise ValueError("Unsupported deflection type.")
        # Markup the max value
        u_max = exDef[1][0] * self.__L  # location on element, tp, simplified, use the original coordinate
        v_max = exDef[1][1] / scale
        uv_max = [u_max, v_max]
        xy_max = vtrans(self.__ang) @ uv_max + np.array(self.start)
        plt.text(*xy_max, str(round(v_max * scale, deci)))
        # Markup the min value
        u_min = exDef[0][0] * self.__L   # location on element, tp, simplified, use the original coordinate
        v_min = exDef[0][1] / scale
        uv_min = [u_min, v_min]
        xy_min = vtrans(self.__ang) @ uv_min + np.array(self.start)
        plt.text(*xy_min, str(round(v_min * scale, deci)))
        # return extreme value
        return exDef[0][1], exDef[1][1]


# Class: Solver for 2D Beam Structure
class Bsolver():
    """ FEM solver for beam model

        :param b2objs: list of ``Beam2`` objects involved in the model.
        :param node_coord: nested list as [[x0,y0],[x1,y1],...], list of coordinates of all nodes in structural model.
                Unit=mm.
        :param restrain_dict: dict. Definition of structural restrain in the form of
                {node_no:[res_condition_x, res_condition_y, res_condition_rotate], ...}, where:
                0=released and 1=restrained.
        :param beam_release: dict. Definition of release conditions of beams in the form of
                {beam_no:[(axial_condition_start, rotation_condition_start), (axial_condition_end, rotation_condition_end)],...},
                where: 0=fixed and 1=released.
        :param distributed_load: dict. Definition of applied *Uniformly Distributed Load* on model in the form of
                {beam_no:(axial_force, transverse_force),...}. Unit=N/mm.
        :param node_forces: dict. Definition of applied *Concentrated Load* on model in the form of
                {node:[Fx, Fy, Mz]...}. Unit=N or N*mm.
    """

    def __init__(self, b2objs, node_coord, restrain_dict, beam_release={}, distributed_load={}, node_forces={}):
        self.elements = b2objs
        self.nodes = node_coord
        self.restrains = restrain_dict
        self.pointload = node_forces
        self.__rels = beam_release

        for i in beam_release:  # assign to beam elements
            self.elements[i].start_rels, self.elements[i].end_rels = beam_release[i]
        for i in distributed_load:  # assign to beams
            self.elements[i].ual, self.elements[i].udl = distributed_load[i]

    def solve(self):
        """Solve the model.

        :return: Matrix of global node displacement.

        .. warning::

            Once being called successfully, solved results (such as member force, deflection, etc.) will be
            automatically assigned to related beam objects. Any existing result in those objects will be hence
            overridden.
        """
        # Initialize Structural Properties of Beams
        for beam in self.elements:
            beam.eval()

        # record 0-strain items, based on restrain condition
        rl = []
        for n in self.restrains:
            rl += [3 * n + i for i in range(3) if self.restrains[n][i]]
        rl.sort()

        # Setup Global Matrix of Stiffness and External Force  
        self.__GKM = np.zeros((3 * len(self.nodes), 3 * len(self.nodes)))
        self.__GRM = np.zeros(3 * len(self.nodes))
        for i in self.pointload:
            self.__GRM[3 * i:3 * i + 3] += self.pointload[i]
        for beam in self.elements:
            beam.fill_GK(self.__GKM)
            beam.fill_GR(self.__GRM)

        # Delete restrained items of which strain = 0
        GKM_d = np.delete(np.delete(self.__GKM, rl, 0), rl, 1)
        GRM_d = np.delete(self.__GRM, rl, 0)

        # Solve the linear equations
        RES = np.linalg.solve(GKM_d, GRM_d)

        # Get list of Global Node Displacement
        self.__GDM = RES
        for i in rl:
            self.__GDM = np.insert(self.__GDM, i, 0, 0)

        # Establish solved results on beam elements
        for beam in self.elements:
            beam.solve(self.__GDM)

        return self.__GDM

    def get_react(self):
        """Output matrix of node reactions in the form of [[Rx_0, Ry_0, M_0], [Rx_1, Ry_1, M_1], ...]. Unit = N or N*mm.

        """
        r = np.array(self.__GKM @ self.__GDM - self.__GRM)
        return r.reshape((int(len(r) / 3), 3))

    def get_endf(self):
        """Output overall matrix of beam end forces. Unit = N or N*mm."""
        bef = [beam.endforce() for beam in self.elements]
        return bef

    def get_enddisp(self):
        """ Output overall matrix of beam end displacements. Unit = mm or rad."""
        bes = [beam.enddisp() for beam in self.elements]
        return bes

    def get_summary(self, datatype, envelop=False):
        """Output extreme value of specified data.

        :param datatype: int or str. indicator of required data type.

                ===  ==========================
                 0    'Axial Force'
                 1    'Shear'
                 2    'Moment'
                 3    'Deflection'
                 4    'Axial Displacement'
                 5    'Resultant Displacement'
                ===  ==========================

        :param envelop: bool, output the extreme value of entire model. Extreme value for each beam element is listed
                when *envelop* = False
        :return: tuple (min_value, max_value) if *envelop* = True. or list of tuple
                [(min_value_beam0, max_value_beam0), (min_value_beam1, max_value_beam1), ...]
        """

        typelist = ("Axial Force", "Shear", "Moment", "Deflection", "Axial Displacement", "Resultant Displacement")
        if datatype == 0 or datatype == typelist[0]:
            exl = [beam.extre_N() for beam in self.elements]
        elif datatype == 1 or datatype == typelist[1]:
            exl = [beam.extre_V() for beam in self.elements]
        elif datatype == 2 or datatype == typelist[2]:
            exl = [beam.extre_M() for beam in self.elements]
        elif datatype == 3 or datatype == typelist[3]:
            exl = [beam.extre_Defy() for beam in self.elements]
        elif datatype == 4 or datatype == typelist[4]:
            exl = [beam.extre_Defx() for beam in self.elements]
        elif datatype == 5 or datatype == typelist[5]:
            exl = [beam.extre_Defxy() for beam in self.elements]

        else:
            raise ValueError("Unsupported Data Tpye")
            quit()

        if envelop:
            allv = [x[1] for ele in exl for x in ele]
            return (min(allv), max(allv))
        else:
            return [(x[0][1], x[1][1]) for x in exl]

    def __plot_mdl(self, show_node=True):
        """ Transfer model information to plot function"""
        for beam in self.elements:
            x, y = list(zip(*[beam.start, beam.end]))
            plt.plot(x, y, 'b')

        for p in self.restrains:
            if self.restrains[p][2]:
                plt.plot(*self.nodes[p], 'sr')
            else:
                plt.plot(*self.nodes[p], '^r')
        if show_node:
            for nd in self.nodes:
                plt.plot(*nd, '.k')
        for m in self.__rels:
            if self.__rels[m][0] == (1, 0):  # translation released
                plt.plot(*self.elements[m].start, '+y')
            elif self.__rels[m][0] == (0, 1):  # rotation released
                plt.plot(*self.elements[m].start, 'oy')
            elif self.__rels[m][0] == (1, 1):  # both released
                plt.plot(*self.elements[m].start, 'xy')

            if self.__rels[m][1] == (1, 0):
                plt.plot(*self.elements[m].end, '+y')
            elif self.__rels[m][1] == (0, 1):
                plt.plot(*self.elements[m].end, 'oy')
            elif self.__rels[m][1] == (1, 1):
                plt.plot(*self.elements[m].end, 'xy')

        xlist, ylist = list(zip(*self.nodes))
        return (min(xlist), min(ylist), max(xlist), max(ylist))

    def show_mdl(self, node_num=True, beam_num=True, size=(8, 6), save=False, path=None):
        """Plot model figure.

        :param node_num: bool, show node number on figure.
        :param beam_num: bool, show beam number on figure.
        :param size: tuple (size_x, size_y), size of the plotted figure.
        :param save: bool, save model figure.
        :param path: str, path to save the model figure. Current working directory is used if *path* = None (default).
        :return: None
        """
        if not path:
            path = getcwd()

        plt.figure(figsize=size)
        plt.title('Model')
        left, bottom, right, top = self.__plot_mdl()
        b = max((right - left), (top - bottom))

        if node_num:
            for i in range(len(self.nodes)):
                plt.text(self.nodes[i][0] + b / 100, self.nodes[i][1] + b / 100, str(i),
                         fontsize='large', color='maroon')

        if beam_num:
            for i in range(len(self.elements)):
                x = (self.elements[i].start[0] + self.elements[i].end[0]) / 2 + b / 100
                y = (self.elements[i].start[1] + self.elements[i].end[1]) / 2 + b / 100
                plt.text(x, y, str(i), fontsize='large', color='indigo')
        plt.axis('equal')
        if save:
            plt.savefig(path + "\\model.jpg")
        else:
            plt.show()

    def show_diag(self, ptype, Size=(8, 6), ShowEnve=True, DivN=5, Scale=1, Accr=4, save=False, path=None):
        """Plot the diagram showing specified results.

        :param ptype: str, type of required results, is one of *'Axial Force'*, *'Shear'*, *'Moment'*, *'Deflection'*,
                      *'Axial Displacement'* and *'Resultant Displacement'*
        :param Size: tuple (size_x, size_y), size of the plotted figure.
        :param ShowEnve: bool, show min/max value in figure.
        :param DivN: int, Number of slice when plotting the curve.
        :param Scale: float, zoom-out scales when plotting the curve.
        :param Accr: int, number of decimal places kept when marking values on diagram.
        :param save: bool, save diagram figure.
        :param path: str, path to save the diagram figure. Current working directory is used if *path* = None (default).
        :return: None
        """
        if not path:
            path = getcwd()

        plt.figure(figsize=Size)
        env = []

        if ptype == 'Axial Force':
            unit = 'N'
            for beam in self.elements:
                env += beam.plot_N(sn=DivN, scale=Scale, deci=Accr)
        elif ptype == 'Shear':
            unit = 'N'
            for beam in self.elements:
                env += beam.plot_V(sn=DivN, scale=Scale, deci=Accr)
        elif ptype == 'Moment':
            unit = 'N.mm'
            for beam in self.elements:
                env += beam.plot_M(sn=DivN, scale=Scale, deci=Accr)
        elif ptype == 'Deflection':
            unit = 'mm'
            for beam in self.elements:
                env += beam.plot_Def(sn=DivN, scale=Scale, deci=Accr, def_type=2)
        elif ptype == 'Axial Displacement':
            unit = 'mm'
            for beam in self.elements:
                env += beam.plot_Def(sn=DivN, scale=Scale, deci=Accr, def_type=1)
        elif ptype == 'Resultant Displacement':
            unit = 'mm'
            for beam in self.elements:
                env += beam.plot_Def(sn=DivN, scale=Scale, deci=Accr, def_type=0)
        else:
            raise ValueError("Unsupported Diagram Type")

        plt.axis('equal')
        self.__plot_mdl(show_node=False)
        if ShowEnve:
            txt = ' Max. {n} = {0}{u} / Min. {n} = {1}{u}'.format(round(max(env), Accr), round(min(env), Accr), n=ptype,
                                                                  u=unit)
        else:
            txt = ''
        plt.title('<{0} Diagram> \n {1}'.format(ptype, txt))
        plt.axis('off')

        if save:
            plt.savefig(path + "\\" + ptype + ".jpg")
        else:
            plt.show()

    def out_csv(self, datatype=0, path=None):
        """ Output results to CSV file.


        :param datatype: int, indicator of required data type to be output.

                + 1 - output reaction data.
                + 2 - output member force and deflection data.
                + 0 - output all data.

        :param path: str, path to save the CSV file. Current working directory is used if *path* = None (default).
        :return: None
        """
        # Node Reaction
        if not path:
            path = getcwd()

        if datatype == 1 or datatype == 0:
            headers = ["Node No.", "Rx (N)", "Ry (N)", "Mz (Nmm)"]
            reactions = self.get_react()
            with open(path + '\\Reaction.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                for i in self.restrains.keys():
                    f_csv.writerow([i]+list(reactions[i]))
            print(f"Reaction load have been saved to {path}")

        # Beam Force and Deflection
        if datatype == 2 or datatype == 0:
            headers = ["Fu1 (N)", "Fv1 (N)", "M1 (N.mm)", "Fu2 (N)", "Fv2 (N)", "M2 (N.mm)",
                       "Fu Min (N)", "Fu Max (N)", "Fv Min (N)", "Fv Max (N)",
                       "M Min (N.mm)", "M Max (N.mm)",  "Def_u Min (mm)", "Def_u Max (mm)",
                       "Def_v Min (mm)", "Def_v Max (mm)"]

            endforces = self.get_endf()
            axial = self.get_summary(0)
            shear = self.get_summary(1)
            moment = self.get_summary(2)
            deflection_v = self.get_summary(3)
            deflection_u = self.get_summary(4)

            with open(path + '\\Member.csv', 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                for i in range(len(axial)):
                    f_csv.writerow(list(endforces[i]) + list(axial[i])
                                   + list(shear[i]) + list(moment[i])
                                   + list(deflection_u[i]) + list(deflection_v[i]))
            print(f"Member Force and Deflection have been saved to {path}")

# SAMPLE
if __name__ == '__main__':

    import json
    import pandas as pd

    # Read Section Properties from json file into dict
    # section_lib = "D:\\Coding File\\PyCharm\\pyfacade\\working_file\\section_lib_mullion.json"
    section_lib = "C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\section_lib_mullion.json"
    with open(section_lib) as f:
        sec_lib = json.load(f)

    # Read Material Properties from csv file into DataFrame
    mat_lib = pd.read_csv("material.csv", index_col="signature")

    # Step 1. Define Coordinate of Node in format: [[x1,y1],[x2,y2]...]
    nodes = [[6750, 0], [6250, 0], [5000, 0], [3750, 0], [3000, 0], [0, 0]]
    nodes2 = [[0, 6750], [0, 6250], [0, 5000], [0, 3750], [0, 3000], [0,0]]

    # Step 2. Define Beam Set in format: [[node1, node2]...]
    beams = [[i, i + 1] for i in range(len(nodes) - 1)]
    beams2 = [[i, i + 1] for i in range(len(nodes2) - 1)]

    # Step 3. Define modulus of elasticity of beam
    E1 = mat_lib.loc['6063-T6', 'E']
    E2 = 70000

    # Step 4. Define section area and moment of inertia of beam
    S1_A = sec_lib['Male_Mullion']['A']
    S1_Ia = sec_lib['Male_Mullion']['Ix']
    S2_A = sec_lib['Female_Mullion']['A']
    S2_Ia = sec_lib['Female_Mullion']['Ix']
    S3_A = 30*60
    S3_Ia = 30*60**3/12

    # Step 5. Create list of beam object
    b = [Beam2(*x, S1_A, S1_Ia, E1, nodes) for x in beams]
    b2 = [Beam2(*x, S3_A, S3_Ia, E2, nodes2) for x in beams2]

    # Step 6. Define restrain in format: {node:[x, y, rotate]...}, 1=restrained
    restr = {1: [1, 1, 0], 4: [1, 1, 0], 5: [0, 1, 0]}
    restr2 = {1: [1, 1, 0], 4: [1, 1, 0], 5: [1, 0, 0]}

    # Step 7. Define end release of each beam in format: {beam:[start, end]...},  1=released
    brels = {2: [(0, 0), (1, 1)]}
    brels2 = {2: [(0, 0), (1, 1)]}
    # brels2 = {3: [(1, 1), (0, 0)]}

    # Step 8. Apply load on beams in format: {beam:(ual, udl)...}
    Q = {i: (0, 2) for i in range(len(beams))}
    Q2 = {i: (0, 2) for i in range(len(beams2))}

    # Step 9. Apply global node force in format: {node:[Fx, Fy, Mz]...}
    F = {}

    # Step 10. Setup solver instance
    model1 = Bsolver(b, nodes, restr, brels, Q, F)
    model2 = Bsolver(b2, nodes2, restr2, brels2, Q2, F)

    # Step 11. Run solve
    model2.solve()

    # Step 12. Output Results
    # model2.show_mdl()
    # model2.show_diag('Axial Force', DivN=5, Scale=500, Accr=4)
    model2.show_diag('Moment', DivN=20, Scale=2000, Accr=4)
    # print(model2.get_summary("Axial Force", envelop=False))
    # print(model2.get_summary("Deflection", envelop=False))
    # print(model2.get_summary("Axial Displacement", envelop=False))
    # print(model2.get_summary("Resultant Displacement", envelop=False))
    # model2.show_diag('Deflection', DivN=20, Scale=0.1, Accr=3)
    # model2.show_diag('Axial Displacement', DivN=20, Scale=0.1, Accr=3)
    # model2.show_diag('Resultant Displacement', DivN=20, Scale=0.1, Accr=3)
    # model2.out_csv(path="C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\")



