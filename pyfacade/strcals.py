# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:07:44 2020

Function Toolbox for Quick Structural Calculation

@author: qi.wang
"""

import os
import json
import time
import numpy as np
from numpy import sin, cos, arctan, pi
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize
from collections import namedtuple
from pyfacade.pyacad import Acad, CADFrame
from pyfacade.pyeng import Beam2, Bsolver


# Constant: file location
Abspath = os.path.dirname(__file__)

# Dictionary: material properties, (read from csv)
Material = pd.read_csv(Abspath+"\\material.csv", index_col="signature")

# Named tuple for results of verification output
Transom_summary = namedtuple("Transom_summary", "max_stress, max_deflection_wl, max_deflection_dl")
Transom_output = namedtuple("Transom_output", "shear, stress, deflection_wl, deflection_dl")
Mullion_summary = namedtuple("Mullion_summary", "max_stress, max_deflection")
Mullion_model = namedtuple("Mullion_model", "N, V, Mn, Mp, D")
Mullion_verify = namedtuple("Mullion_verify", "axial, shear, moment_n, moment_p, stress, deflection")
Mullion_critical = namedtuple("Mullion_critical", "N, V, S, D")
Frame_model = namedtuple("Frame_model", "nodes, beams, A, I, E, restrain, release, udl, pl")


def eqslds(I1, I2, alpha, E):
    """Calculate equivalent slenderness of a section.

    :param I1: float, moment of inertia about major axis of the section.
    :param I2: float, moment of inertia about major axis of the section.
    :param alpha: float, angle between major aixs and x-axis, in radians.
    :param E: float, modulus of elasticity of the section material.
    :return: tuple of float, (k_yx, k_yy, k_xx, k_xy), where *k_ij* represents equivalent slenderness about i-axis under
             force in j-direction.
    """
    k_yx = (I1 * cos(alpha) ** 2 + I2 * sin(alpha) ** 2) / (
            I1 * I2 * E)  # slenderness about y-axis under force in x-axis
    k_yy = (I1 - I2) * sin(2 * alpha) / (2 * I1 * I2 * E)  # slenderness about y-axis under force in y-axis
    k_xx = k_yy  # slenderness about x-axis under force in x-axis
    k_xy = (I1 * sin(alpha) ** 2 + I2 * cos(alpha) ** 2) / (
            I1 * I2 * E)  # slenderness about x-axis under force in y-axis
    return k_yx, k_yy, k_xx, k_xy


def biabend(section_prop, Mx, My, full_output=False):
    """Calculate maximum normal stress on section due to biaxial bending.

    :param section_prop: sub-dict for a section extracted from the dict created by ``pyacad.Acad.seclib``.
    :param Mx: float, bending moment about x-axis, unit = N*mm.
    :param My: float, bending moment about y-axis, unit = N*mm.
    :param full_output: bool, output all the calculation parameter.
    :return: tuple, in the form of (s_min, s_max, theta, Mn, In, dn)

            | s_min - float, minimum stress on the section, unit = N/mm :superscript:`2`
            | s_max - float, maximum stress on the section, unit = N/mm :superscript:`2`
            | theta - float, angle between neutral axis and x-axis, in radians.
            | Mn - float, resultant effective moment about neutral axis, unit = N*mm
            | In - float, moment of inertia of section about neutral axis, unit = mm :superscript:`4`.
            | dn - tuple of float (d_min, d_max), minimum and maximum relative distance from the most critical points to
                    neutral axis.

            The last 4 items are excluded when *full_output* = False.
    """

    # Avoid divided by zero:
    if Mx == 0 and My == 0:
        return 0, 0

    Ix = section_prop['Ix']
    Iy = section_prop['Iy']
    Ixy = section_prop['Ixy']

    # Angle between neutral axis and x-axis
    theta = arctan((My * Ix + Mx * Ixy) / (Mx * Iy + My * Ixy))

    # Resolved effective moment about neutral axis
    Mn = Mx * cos(theta) + My * sin(theta)

    # Find the elastic section modulus about neutral axis
    In = (Ix + Iy) / 2 + (Ix - Iy) / 2 * cos(2 * theta) - Ixy * sin(2 * theta)
    dn = Acad.boundalong(section_prop['bnode'], section_prop['barc'], Acad.rotatevec([1, 0, 0], theta + pi / 2))

    if full_output:
        return Mn * dn[0] / In, Mn * dn[1] / In, theta, Mn, In, dn
    else:
        s1 = Mn * dn[0] / In
        s2 = Mn * dn[1] / In
        if s1 > 0:
            s1, s2 = s2, s1  # keep s1 compressive stress
        return s1, s2


def combsec(sec_lib, sec_mat, combinations):
    """Group sections, calculate matrix of load sharing and equivalent slenderness according to combination cases

    :param sec_lib: nested dict stats section properties in the form of
                    {sec_name_1: {'I1':.., 'I2':..,'alpha':..}, sec_name_2:...}
    :param sec_mat: dict, material of involved sections, {sec_name_1: material_1, sec_name_2: material_2, ...}
    :param combinations: nested list, relation of section combinations, [[sec_name_1, sec_name_2, comb_indicator],...]
                         The *comb_indicator* is one of below:

                         | 'x', 'X' or 1 - combine in x-direction
                         | 'y', 'Y' or 2 - combine in y-direction
                         | 'xy', 'XY' or 0 - combine in both directions

    :return: tuple, (section group in x-direction, section group in y-direction, matrix of force sharing,
             matrix of moment sharing, matrix of equivalent slenderness)
    """

    # Function: sort section pair into corresponding group according to the combination relations
    def sec_grouping(s1, s2, grp, comb):
        if comb:
            for g in grp:
                if (s1 in g) and (s2 not in g):
                    g.append(s2)
                    break
                elif (s1 not in g) and (s2 in g):
                    g.append(s1)
                    break
                elif (s1 in g) and (s2 in g):
                    break
            else:
                grp.append([s1, s2])  # add se1 and sec2 as a new group if no existing is found
        else:
            for s in [s1, s2]:
                for g in grp:
                    if s in g:  # find if sec1 or sec2 is already in certain group
                        break
                else:
                    grp.append([s])  # add sec1 or sec2 as separated group

    secs = list(sec_mat.keys())  # section order follows the input dict of section material
    ns = len(secs)  # total number of section components

    # equivalent slenderness of sections
    k = {sec: eqslds(sec_lib[sec]['I1'], sec_lib[sec]['I2'], sec_lib[sec]['alpha'], Material.loc[sec_mat[sec], 'E'])
         for sec in secs}

    # Create matrix of linear equations
    empty_EF = True
    group_x = []  # initialize: force groups in x-axis  [[group1],[group2]...]
    group_y = []  # initialize: force groups in y-axis  [[group1],[group2]...]
    for cb in combinations:
        sec1, sec2, d = cb
        id1 = secs.index(sec1)
        id2 = secs.index(sec2)
        if d in ["x", "X", 1]:  # combined along x-direction

            # equation of deformation
            eqs_v = np.zeros(ns * 2)  # initialize the 1xN matrix, N= 2*section numbers
            eqs_m = np.zeros(ns * 2)  # initialize the 1xN matrix, N= 2*section numbers
            eqs_v[id1 * 2:id1 * 2 + 2] = k[sec1][0], k[sec1][1]  # [..., k_s1_yx, k_s1_yy,...]
            eqs_v[id2 * 2:id2 * 2 + 2] = -k[sec2][0], -k[sec2][1]  # [..., -k_s2_yx, -k_s2_yy,...]
            eqs_m[id1 * 2:id1 * 2 + 2] = k[sec1][0], -k[sec1][1]  # [..., k_s1_yx, -k_s1_yy,...]
            eqs_m[id2 * 2:id2 * 2 + 2] = -k[sec2][0], k[sec2][1]  # [..., -k_s2_yx, k_s2_yy,...]
            # # grouping force in x-direction:
            sec_grouping(sec1, sec2, group_x, comb=True)
            # # grouping force in y-direction:
            sec_grouping(sec1, sec2, group_y, comb=False)

        elif d in ["y", "Y", 2]:  # combined along y-direction

            # equation of deformation
            eqs_v = np.zeros(ns * 2)  # initialize the 1xN matrix, N= 2*section numbers
            eqs_m = np.zeros(ns * 2)  # initialize the 1xN matrix, N= 2*section numbers
            eqs_v[id1 * 2:id1 * 2 + 2] = k[sec1][2], k[sec1][3]  # [..., k_s1_xx, k_s1_xy,...]
            eqs_v[id2 * 2:id2 * 2 + 2] = -k[sec2][2], -k[sec2][3]  # [..., -k_s2_xx, -k_s2_xy,...]
            eqs_m[id1 * 2:id1 * 2 + 2] = k[sec1][2], -k[sec1][3]  # [..., k_s1_xx, -k_s1_xy,...]
            eqs_m[id2 * 2:id2 * 2 + 2] = -k[sec2][2], k[sec2][3]  # [..., -k_s2_xx, k_s2_xy,...]
            # # grouping force in x-direction:
            sec_grouping(sec1, sec2, group_x, comb=False)
            # # grouping force in y-direction:
            sec_grouping(sec1, sec2, group_y, comb=True)

        elif d in ["xy", "XY", 0]:  # combined along both x-direction and y-direction
            # equation of deformation
            eqs_v = np.zeros((2, ns * 2))  # initialize the 2xN matrix, N= 2*section numbers
            eqs_m = np.zeros((2, ns * 2))  # initialize the 2xN matrix, N= 2*section numbers
            eqs_v[0, id1 * 2:id1 * 2 + 2] = k[sec1][0], k[sec1][1]  # [..., k_s1_yx, k_s1_yy,...]
            eqs_v[0, id2 * 2:id2 * 2 + 2] = -k[sec2][0], -k[sec2][1]  # [..., -k_s2_yx, -k_s2_yy,...]
            eqs_v[1, id1 * 2:id1 * 2 + 2] = k[sec1][2], k[sec1][3]  # [..., k_s1_xx, k_s1_xy,...]
            eqs_v[1, id2 * 2:id2 * 2 + 2] = -k[sec2][2], -k[sec2][3]  # [..., -k_s2_xx, -k_s2_xy,...]
            eqs_m[0, id1 * 2:id1 * 2 + 2] = k[sec1][0], -k[sec1][1]  # [..., k_s1_yx, -k_s1_yy,...]
            eqs_m[0, id2 * 2:id2 * 2 + 2] = -k[sec2][0], k[sec2][1]  # [..., -k_s2_yx, k_s2_yy,...]
            eqs_m[1, id1 * 2:id1 * 2 + 2] = k[sec1][2], -k[sec1][3]  # [..., k_s1_xx, -k_s1_xy,...]
            eqs_m[1, id2 * 2:id2 * 2 + 2] = -k[sec2][2], k[sec2][3]  # [..., -k_s2_xx, k_s2_xy,...]
            # grouping force in x-direction:
            sec_grouping(sec1, sec2, group_x, comb=True)
            # grouping force in y-direction:
            sec_grouping(sec1, sec2, group_y, comb=True)

        else:
            raise ValueError("Unrecognized combination indicator.")

        # build up equations about deformation
        if empty_EF:
            EFV = eqs_v
            EFM = eqs_m
            empty_EF = False
        else:
            EFV = np.vstack([EFV, eqs_v])
            EFM = np.vstack([EFM, eqs_m])

    # Add equations about force balance
    for g in group_x:
        eqs = np.zeros((1, ns * 2))
        for sec in g:
            eqs[0, secs.index(sec) * 2] = 1
        EFV = np.vstack([EFV, eqs])
        EFM = np.vstack([EFM, eqs])
    for g in group_y:
        eqs = np.zeros((1, ns * 2))
        for sec in g:
            eqs[0, secs.index(sec) * 2 + 1] = 1
        EFV = np.vstack([EFV, eqs])
        EFM = np.vstack([EFM, eqs])

    # Solve linear equations, get a list of load sharing factors in order of load groups
    group_len = len(group_x) + len(group_y)  # total length of group_x and group_y
    rs_d = np.zeros((ns * 2 - group_len, group_len))
    rs_f = np.eye(group_len)
    eta_v = solve(EFV, np.vstack([rs_d, rs_f]))  # for force sharing
    eta_m = solve(EFM, np.vstack([rs_d, rs_f]))  # for moment sharing

    # Global matrix of equivalent slenderness
    es = np.zeros((ns * 2, ns * 2))  # initialize the NxN matrix, N= 2*section numbers
    for n in range(ns):
        es[n, n * 2] = k[secs[n]][0]  # k_yx
        es[n, n * 2 + 1] = k[secs[n]][1]  # k_yy
        es[n + ns, n * 2] = k[secs[n]][2]  # k_xx
        es[n + ns, n * 2 + 1] = k[secs[n]][3]  # k_xy

    return group_x, group_y, eta_v, eta_m, es


def check_transoms(section_lib, section_mat, section_comb, span, h1, h2, load_app, wl, dl1, dl2=0.0, wlf=0.0, imp=0.0,
                   imq=0.0, feature=0.0, wlf_flip=True, four_side1=True, four_side2=True, ds1=0.0, ds2=0.0, wlc=0.5,
                   summary=False):
    """Verification of transom of combined sections.

    :param section_lib: str or dict, section library stating section properties and boundary information. In form of
                        str as path and name of json file created by ``pyacad.Acad.seclib``, or a nested dict in the
                        same form.
    :param section_mat: dict, material of involved sections, {sec_name_1: material_1, sec_name_2: material_2, ...}
    :param section_comb: nested list, relation of section combinations, [[sec_name_1, sec_name_2, comb_indicator],...].
                        The *comb_indicator* is one of below:

                        | 'x', 'X' or 1 - combine in x-direction.
                        | 'y', 'Y' or 2 - combine in y-direction.
                        | 'xy', 'XY' or 0 - combine in both directions.

    :param span: float, span of transom, unit = mm.
    :param h1: float, height of upper panel, unit = mm.
    :param h2: float, height of lower panel, unit = mm.
    :param load_app: list of str, name of sections take loads [for_wl1, for_wl2, for_wlf, for_imp, for_dl1, for_dl2].

                     | for_wl1 - name of section taking wind load from upper panel.
                     | for_wl2 - name of section taking wind load from lower panel.
                     | for_wlf - name of section taking wind load from horizontal feature, if any.
                     | for_imp - name of section taking vertical imposed load, if any.
                     | for_dl1 - name of section taking dead load from upper panel, if any.
                     | for_dl2 - name of section taking dead load from lower panel, if any.

                    the last four items can be ``None`` when corresponding load does not exist.

    :param wl: list of float, design wind load on panel, [pressure, suction], unit = kPa, positive as pressure,
            negative as suction.
    :param dl1: float, design weight of upper panel, unit = N.
    :param dl2: float, design weight of lower panel, unit = N.
    :param wlf: float, design wind load on feature, unit = kPa, positive as uplifting, negative as downward.
    :param imp: float, design imposed point load, unit = N, positive as uplifting, negative as downward.
    :param imq: float, design imposed linear load, unit = N/mm, positive as uplifting, negative as downward.
    :param feature: float, windward breadth of horizontal feature, unit = mm.
    :param wlf_flip: bool, also check the case that wind load direction on feature is flipped.
    :param four_side1: bool, load path of upper panel is considered as 4-side-supported.
    :param four_side2: bool, load path of lower panel is considered as 4-side-supported.
    :param ds1: float, distance from ends to apply location of upper panel weight. unit = mm.
                Apply panel weight as udl if *ds1* = 0.
    :param ds2: float, distance from ends to apply location of lower panel weight. unit = mm.
                Apply panel weight as udl if *ds1* = 0.
    :param wlc: float, combination factor of wind load when combine with imposed load.
    :param summary: bool, output the summary of verification only.
    :return: namedtuple ``Transom_summary`` if *summary* = True. Otherwise return namedtuple ``Transom_output``.
    """

    # define load factor for alum. member
    fd_a = 1.2  # adverse dl
    fd_ab = 0.8  # beneficial dl
    fw_a = 1.4  # wl
    fI_a = 1.33  # im

    # define load factor for steel member
    fd_s = 1.4  # adverse dl
    fd_sb = 1.0  # beneficial dl
    fw_s = 1.4  # wl
    fI_s = 1.6  # im

    # initialize section properties
    if type(section_lib) == str and section_lib[-4:] == 'json':  # read from specified jason file
        with open(section_lib) as f:
            sec_lib = json.load(f)
    elif type(section_lib) == dict:  # read directly as a sorted dictionary
        sec_lib = section_lib
    else:
        raise ValueError("Unsupported Type of Section Library")

    # Record section component name
    secs = list(section_mat.keys())  # section order follows the input dict of section material
    ns = len(secs)  # total number of section components

    # calculate combination-related data
    group_x, group_y, eta_v, eta_m, es = combsec(sec_lib, section_mat, section_comb)
    group_len = len(group_x) + len(group_y)

    def locsec(section, grp):
        for g in grp:
            if section in g:
                return grp.index(g)
        else:
            raise ValueError(f"Can't find <{section}> in <{grp}>.")

    # Calculate member force & deflection due to wind pressure on panel
    wp, ws = wl
    if four_side1:
        # moment due to wp on upper panel
        M_wp1 = wp * span ** 3 / 24000 if h1 >= span else wp * h1 * span ** 2 * (3 - (h1 / span) ** 2) / 48000
        # shear due to wp on upper panel
        V_wp1 = wp * span ** 2 / 8000 if h1 >= span else wp * h1 * (span - h1 / 2) / 4000
        # deflection coefficient due to wp on upper panel
        d_wp1 = wp * span ** 5 / 240000 if h1 >= span else wp * h1 * span ** 4 * (25 - 40 * (h1 * 0.5 / span) ** 2 +
                                                                                  16 * (h1 * 0.5 / span) ** 4) / 3840000
    else:
        M_wp1 = wp * h1 * span ** 2 / 16000  # moment due to wp on upper panel, udl case
        V_wp1 = wp * h1 * span / 4000  # shear due to wp on upper panel, udl case
        d_wp1 = 5 * wp * h1 * span ** 4 / 768000  # deflection coefficient due to wp on upper panel, udl case
    if four_side2:
        # moment due to wp on lower panel
        M_wp2 = wp * span ** 3 / 24000 if h2 >= span else wp * h2 * span ** 2 * (3 - (h2 / span) ** 2) / 48000
        # shear due to wp on lower panel
        V_wp2 = wp * span ** 2 / 8000 if h2 >= span else wp * h2 * (span - h2 / 2) / 4000
        # deflection coefficient due to wp on lower panel
        d_wp2 = wp * span ** 5 / 240000 if h2 >= span else wp * h2 * span ** 4 * (25 - 40 * (h2 * 0.5 / span) ** 2 +
                                                                                  16 * (h2 * 0.5 / span) ** 4) / 3840000
    else:
        M_wp2 = wp * h2 * span ** 2 / 16000  # moment due to wp on lower panel, udl case
        V_wp2 = wp * h2 * span / 4000  # shear due to wp on lower panel, udl case
        d_wp2 = 5 * wp * h2 * span ** 4 / 768000  # deflection coefficient due to wp on lower panel, udl case

    # Calculate member force & deflection coefficient due to wind suction on panel
    M_ws1 = M_wp1 * ws / wp
    V_ws1 = V_wp1 * ws / wp
    d_ws1 = d_wp1 * ws / wp
    M_ws2 = M_wp2 * ws / wp
    V_ws2 = V_wp2 * ws / wp
    d_ws2 = d_wp2 * ws / wp

    # Calculate member force & deflection coefficient due to wind load on feature
    M_wf = -wlf * feature * span ** 2 / 8000  # negative moment if load is positive (uplifting)
    V_wf = wlf * feature * span / 2000
    d_wf = 5 * wlf * feature * span ** 4 / 384000

    # Calculate member force & deflection coefficient due to dead load
    V_sw = [-sec_lib[s]['A'] * Material.loc[section_mat[s], 'dens'] * 9.807e-9 * span / 2 for s in
            secs]  # shear due to self-weight
    M_sw = [sec_lib[s]['A'] * Material.loc[section_mat[s], 'dens'] * 9.807e-9 * span ** 2 / 8 for s in
            secs]  # moment due to self-weight
    d_sw = [-5 * sec_lib[s]['A'] * Material.loc[section_mat[s], 'dens'] * 9.807e-9 * span ** 4 / 384 for s in
            secs]  # deflection coefficient due to self-weight
    V_pd1 = -abs(dl1) / 2  # shear, always negative
    if ds1:  # pair of concentrated load
        M_pd1 = abs(dl1) / 2 * ds1  # moment, always positive
        d_pd1 = -abs(dl1) * ds1 * (3 * span ** 2 - 4 * ds1 ** 2) / 48  # deflection coefficient, always negative
    else:  # udl
        M_pd1 = abs(dl1) * span / 8
        d_pd1 = -abs(dl1) * span ** 3 * 5 / 384
    V_pd2 = -abs(dl2) / 2  # shear, always negative
    if ds2:  # pair of concentrated load
        M_pd2 = abs(dl2) / 2 * ds2  # moment, always positive
        d_pd2 = -abs(dl2) * ds2 * (3 * span ** 2 - 4 * ds2 ** 2) / 48  # deflection coefficient, always negative
    else:  # udl
        M_pd2 = abs(dl2) * span / 8
        d_pd2 = -abs(dl2) * span ** 3 * 5 / 384

    # Calculate member force due to imposed load
    M_imp = -imp * span / 4  # negative moment if load is positive (uplifting)
    V_imp = imp / 2
    M_imq = -imq * span ** 2 / 8  # negative moment if load is positive (uplifting)
    V_imq = imq * span / 2
    M_im = M_imp if abs(M_imp) > abs(M_imq) else M_imq  # take the larger one
    V_im = V_imp if abs(V_imp) > abs(V_imq) else V_imq  # take the larger one

    # Total matrix of shear force, bending moment and deflection coefficient
    V_all = np.zeros((group_len, ns + 9))
    M_all = np.zeros((group_len, ns + 9))
    d_all = np.zeros((group_len, ns + 8))
    V_all[locsec(load_app[0], group_x), 0] = V_wp1
    V_all[locsec(load_app[1], group_x), 1] = V_wp2
    V_all[locsec(load_app[0], group_x), 2] = V_ws1
    V_all[locsec(load_app[1], group_x), 3] = V_ws2
    M_all[locsec(load_app[0], group_x), 0] = M_wp1
    M_all[locsec(load_app[1], group_x), 1] = M_wp2
    M_all[locsec(load_app[0], group_x), 2] = M_ws1
    M_all[locsec(load_app[1], group_x), 3] = M_ws2
    d_all[locsec(load_app[0], group_x), 0] = d_wp1
    d_all[locsec(load_app[1], group_x), 1] = d_wp2
    d_all[locsec(load_app[0], group_x), 2] = d_ws1
    d_all[locsec(load_app[1], group_x), 3] = d_ws2
    if wlf > 0:  # uplifting wind on feature
        V_all[locsec(load_app[2], group_y) + len(group_x), 4] = V_wf
        M_all[locsec(load_app[2], group_y) + len(group_x), 4] = M_wf
        d_all[locsec(load_app[2], group_y) + len(group_x), 4] = d_wf
        if wlf_flip:  # consider the flipped case too
            V_all[locsec(load_app[2], group_y) + len(group_x), 5] = -V_wf
            M_all[locsec(load_app[2], group_y) + len(group_x), 5] = -M_wf
            d_all[locsec(load_app[2], group_y) + len(group_x), 5] = -d_wf
    elif wlf < 0:  # downward wind on feature
        V_all[locsec(load_app[2], group_y) + len(group_x), 5] = V_wf
        M_all[locsec(load_app[2], group_y) + len(group_x), 5] = M_wf
        d_all[locsec(load_app[2], group_y) + len(group_x), 5] = d_wf
        if wlf_flip:  # consider the flipped case too
            V_all[locsec(load_app[2], group_y) + len(group_x), 4] = -V_wf
            M_all[locsec(load_app[2], group_y) + len(group_x), 4] = -M_wf
            d_all[locsec(load_app[2], group_y) + len(group_x), 4] = -d_wf
    if imp or imq:
        V_all[locsec(load_app[3], group_y) + len(group_x), 6] = V_im
        M_all[locsec(load_app[3], group_y) + len(group_x), 6] = M_im
    if dl1:
        V_all[locsec(load_app[4], group_y) + len(group_x), 7] = V_pd1
        M_all[locsec(load_app[4], group_y) + len(group_x), 7] = M_pd1
        d_all[locsec(load_app[4], group_y) + len(group_x), 6] = d_pd1
    if dl2:
        V_all[locsec(load_app[5], group_y) + len(group_x), 8] = V_pd2
        M_all[locsec(load_app[5], group_y) + len(group_x), 8] = M_pd2
        d_all[locsec(load_app[5], group_y) + len(group_x), 7] = d_pd2
    for x in range(ns):
        V_all[locsec(secs[x], group_y) + len(group_x), 9 + x] = V_sw[x]
        M_all[locsec(secs[x], group_y) + len(group_x), 9 + x] = M_sw[x]
        d_all[locsec(secs[x], group_y) + len(group_x), 8 + x] = d_sw[x]

    # Load on sections due to each load case
    V_shared = eta_v @ V_all
    M_shared = eta_m @ M_all

    # Matrix of load combination factor
    lc_wim_a = np.array([[fw_a, fw_a, 0, 0, wlc * fw_a, 0],
                         [fw_a, fw_a, 0, 0, wlc * fw_a, 0],
                         [0, 0, fw_a, fw_a, 0, wlc * fw_a],
                         [0, 0, fw_a, fw_a, 0, wlc * fw_a],
                         [0, fw_a, 0, fw_a, 0, 0],
                         [fw_a, 0, fw_a, 0, wlc * fw_a, wlc * fw_a],
                         [0, 0, 0, 0, fI_a, fI_a]])
    lc_d_a = np.array([fd_a, fd_ab, fd_a, fd_ab, fd_a, fd_a] * (ns + 2)).reshape(((ns + 2), 6))
    lcf_a = np.vstack([lc_wim_a, lc_d_a])

    if not (imp or imq):
        lcf_a = np.delete(lcf_a, [4, 5], axis=1)
    if not wlf:
        lcf_a = np.delete(lcf_a, [1, 3], axis=1)

    # Factored combined load shared by each section
    VC_a = V_shared @ lcf_a
    MC_a = M_shared @ lcf_a

    # Factored combined load for steel section
    steel_sec = [secs.index(sec) for sec in section_mat if section_mat[sec][0] == 'S']  # record id of steel section
    if steel_sec:  # steel section exists
        lc_wim_s = np.array([[fw_s, fw_s, 0, 0, wlc * fw_s, 0],
                             [fw_s, fw_s, 0, 0, wlc * fw_s, 0],
                             [0, 0, fw_s, fw_s, 0, wlc * fw_s],
                             [0, 0, fw_s, fw_s, 0, wlc * fw_s],
                             [0, fw_s, 0, fw_s, 0, 0],
                             [fw_s, 0, fw_s, 0, wlc * fw_s, wlc * fw_s],
                             [0, 0, 0, 0, fI_s, fI_s]])
        lc_d_s = np.array([fd_s, fd_sb, fd_s, fd_sb, fd_s, fd_s] * (ns + 2)).reshape(((ns + 2), 6))
        lcf_s = np.vstack([lc_wim_s, lc_d_s])
        if not (imp or imq):
            lcf_s = np.delete(lcf_s, [4, 5], axis=1)
        if not wlf:
            lcf_s = np.delete(lcf_s, [1, 3], axis=1)
        VC_s = V_shared @ lcf_s
        MC_s = M_shared @ lcf_s

    # Result: Shear force & bending stress on sections
    shear = {}
    stress = {}
    for i in range(ns):
        sec_v = VC_s[i * 2:i * 2 + 2, :] if i in steel_sec else VC_a[i * 2:i * 2 + 2, :]  # extract related shear
        shear[secs[i]] = [(sec_v[0, j], sec_v[1, j]) for j in range(sec_v.shape[1])]  # sort as per load combination
        sec_m = MC_s[i * 2:i * 2 + 2, :] if i in steel_sec else MC_a[i * 2:i * 2 + 2, :]  # extract related moment
        stress[secs[i]] = [biabend(sec_lib[secs[i]], sec_m[1, j], sec_m[0, j]) for j in range(sec_m.shape[1])]

    # Deflection of sections due to each load case
    defs = es @ eta_v @ d_all

    # Matrix of deflection combination factor: Wind Load
    dc_w = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0]])
    dc_d = np.zeros((ns + 2, 4))
    dcf_wind = np.vstack([dc_w, dc_d])
    if not wlf:
        dcf_wind = np.delete(dcf_wind, [1, 3], axis=1)

    # Matrix of deflection combination factor: Dead Load
    dcf_dead = np.array([0, 0, 0, 0, 0, 0] + [1] * (ns + 2))

    # Combined Deflection of each section
    def_wind = defs @ dcf_wind  # wind load cases
    def_dead = defs @ dcf_dead  # dead load cases

    # Result: Deflection of Sections
    deflection_w = {}
    deflection_d = {}
    for i in range(ns):
        deflection_w[secs[i]] = [(def_wind[i, j], def_wind[ns + i, j]) for j in range(def_wind.shape[1])]
        deflection_d[secs[i]] = (def_dead[i], def_dead[ns + i])

    if summary:  # return summary
        max_stress = [0, 0]  # [max. stress on alum member, max stress on steel member]
        for s in secs:
            stress_list = [abs(x) for case in stress[s] for x in case]
            max_s = max(stress_list)
            index = 1 if secs.index(s) in steel_sec else 0
            if max_stress[index] < max_s:
                max_stress[index] = max_s
        max_deflection_w = [abs(def_wind[0:ns, :].flatten()).max(), abs(def_wind[ns:, :].flatten()).max()]
        max_deflection_d = [abs(def_dead[0:ns]).max(), abs(def_dead[ns:]).max()]
        return Transom_summary(max_stress, max_deflection_w, max_deflection_d)

    else:  # return detail results
        return Transom_output(shear, stress, deflection_w, deflection_d)


def check_mullions(section_lib, section_mat, section_comb, height, b1, b2, load_app, wl, panel_dens1, panel_dens2,
                   support1, support2=None, support2_lateral=False, wlvf=0.0, wlhf=0.0, bvf=0.0, bhf=0.0, hf_gap=0.0,
                   hf_loc=None, wlvf_flip=True, wlhf_flip=True, summary=False):
    """Verification of mullion of consistently combined sections.

    :param section_lib: str or dict, section library stating section properties and boundary information. In form of
                        str as path and name of json file created by ``pyacad.Acad.seclib``, or a nested dict in the
                        same form.
    :param section_mat: dict, material of involved sections, {sec_name_1: material_1, sec_name_2: material_2, ...}
    :param section_comb: list, relationship of section combinations, [[sec_name_1, sec_name_2, comb_indicator],...].
                        The *comb_indicator* is one of below:

                        | 'x', 'X' or 1 - combine in x-direction
                        | 'y', 'Y' or 2 - combine in y-direction
                        | 'xy', 'XY' or 0 - combine in both directions

    :param height: list of float, height of mullions from lower to upper, unit = mm
    :param b1: float, width of panel at left side of mullion, unit = mm
    :param b2: float, width of panel at right side of mullion, unit = mm
    :param load_app: list of str, name of sections take loads
                    [for_wl1, for_wl2, for_dl1, for_dl2, for_wlvf, for_wlhf1, for_wlhf2]

                     | for_wl1 - name of section taking wind load from left panel.
                     | for_wl2 - name of section taking wind load from right panel.
                     | for_dl1 - name of section taking dead load from left panel.
                     | for_dl2 - name of section taking dead load from right panel.
                     | for_wlvf - name of section taking wind load from vertical feature, if any.
                     | for_wlhf1 - name of section taking wind load from horizontal feature at left side, if any.
                     | for_wlhf2 - name of section taking wind load from horizontal feature at right side, if any.

                    the last three items can be ``None`` when corresponding load does not exist

    :param wl: list of float, design wind load on panel, [pressure, suction], unit = kPa, positive as pressure,
               negative as suction.
    :param panel_dens1: float, design area density of panel at left side of mullion, unit = kN/m :superscript:`2`.
    :param panel_dens2: float, design area density of panel at right side of mullion, unit = kN/m :superscript:`2`.
    :param support1: list of float, distance from top end of mullion to primary support, unit = mm.
    :param support2: list of float, distance from primary support to secondary support, unit = mm.
                     No secondary support in that span if corresponding item in list is 0.
    :param support2_lateral: bool, secondary support also provides lateral restrain.
    :param wlvf: float, design wind load on vertical feature, unit = kPa, positive as rightward, negative as leftward.
    :param wlhf: float, design wind load on horizontal feature, unit = kPa, positive as uplifting, negative as downward.
    :param bvf: float, windward breadth of vertical feature, unit = mm.
    :param bhf: float, windward breadth of horizontal feature, unit = mm.
    :param hf_gap: float, gap between horizontal feature and centroid of connected section, unit = mm.
    :param hf_loc: list, distance of horizontal features to top end of mullion at each span, unit = mm.
                   No horizontal feature in that span if corresponding item in list is 0.
    :param wlvf_flip: bool, also check the case that wind load direction on vertical feature is flipped.
    :param wlhf_flip: bool, also check the case that wind load direction on horizontal feature is flipped.
    :param summary: bool, output the summary of verification only.
    :return: if *summary* = True, return a namedtuple ``Mullion_summary``. otherwise return 3 namedtuple,
             ``Mullion_model``, ``Mullion_verify`` and ``Mullion_critical``.
    """

    # Define load factor for alum. member
    fd_a = 1.2  # adverse dl
    fd_ab = 0.8  # beneficial dl
    fw_a = 1.4  # wl

    # Define load factor for steel member
    fd_s = 1.4  # adverse dl
    fd_sb = 1.0  # beneficial dl
    fw_s = 1.4  # wl

    # Initialize section properties
    if type(section_lib) == str and section_lib[-4:] == 'json':  # read from specified jason file
        with open(section_lib) as f:
            sec_lib = json.load(f)
    elif type(section_lib) == dict:  # read directly as a sorted dictionary
        sec_lib = section_lib
    else:
        raise ValueError("Unsupported Type of Section Library")

    # Record section component name
    secs = list(section_mat.keys())  # section order follows the input dict of section material
    ns = len(secs)  # total number of section components

    # Calculate combination-related data
    group_x, group_y, eta_v, eta_m, es = combsec(sec_lib, section_mat, section_comb)
    group_len = len(group_x) + len(group_y)

    def locsec(section, grp):
        for g in grp:
            if section in g:
                return grp.index(g)
        else:
            raise ValueError(f"Can't find <{section}> in <{grp}>.")

    # region <Calculation Model of Mullion>  -->> model_wl, model_dl, model_hf, model_vf

    # Define Coordinate of Node in format: [[x1,y1],[x2,y2]...]
    spans = len(height)  # number of total span in model
    joint_y = np.array([sum(height[:i + 1]) for i in range(spans)])  # location of stack joint
    support1_y = joint_y - np.array(support1)  # location of stack joint
    if support2:  # secondary support is defined
        support2_y = support1_y - np.array(support2)
    else:
        support2_y = []
    hf_y = []
    if hf_loc:  # location of horizontal feature is defined
        for i in range(len(hf_loc)):
            try:
                for hf in hf_loc[i]:  # try if multiple features on one span
                    hf_y.append(joint_y[i] - hf)
            except TypeError:
                hf_y.append(joint_y[i] - hf_loc[i])  # single feature on one span
    nodes_y = np.unique(np.concatenate(([0], joint_y, support1_y, support2_y, hf_y)))  # drop the duplicated node
    nodes_y.sort()  # sort the nodes from bottom to top
    nl = list(nodes_y)  # list of y-coordinates of sorted nodes
    nodes = [[0, y] for y in nodes_y]

    # Define Beam Set in format:
    beams = [[i, i + 1] for i in range(len(nodes) - 1)]

    # Record the mapping of Beam Id to mullion span
    joint_id = [nl.index(y) for y in joint_y]  # node id of stack joint
    sn_ = 0
    mu = {}
    for k in range(len(beams)):
        if k >= joint_id[sn_]:
            sn_ += 1
        mu[k] = sn_

    # Create list of beam object
    b = [Beam2(*x, 1, 1, 1, nodes) for x in beams]

    # Define restrain in format: {node:[x, y, rotate]...}, 1=restrained
    rp1 = [nl.index(y) for y in support1_y]  # node id for primary support
    restr = {p: [1, 1, 0] for p in rp1}
    restr.update({0: [1, 0, 0]})
    if support2:
        rp2 = [nl.index(y) for y in support2_y if y not in support1_y]  # node id for secondary support
        restr.update({p: [1, 0, 0] for p in rp2})  # add restrains of secondary support

    # Define end release of each beam in format: {beam:[(start_T, start_R), (end_T, end_R)]...},  1=released>>>
    jp = [nl.index(y) for y in joint_y]  # node id for stack joint
    brels = {i: [(1, 1), (0, 0)] for i in range(len(beams)) if beams[i][0] in jp}

    # Model 1, Load Case 1: unit udl on beams:
    Q = {i: (0, 1) for i in range(len(beams))}
    model_wl = Bsolver(b, nodes, restr, brels, distributed_load=Q)
    model_wl.solve()

    # Model 1, Load Case 2: unit ual on beams:
    Qa = {i: (-1, 0) for i in range(len(beams))}  # dl always downward
    b_dl = [Beam2(*x, 1, 1, 1, nodes) for x in beams]  # new set of beam elements
    model_dl = Bsolver(b_dl, nodes, restr, brels, distributed_load=Qa)
    model_dl.solve()

    simple_case = True  # flag of simple case of linear model, extreme values occur at same location

    # Model1, Load Case 3: unit node force in format: {node:[Fx, Fy, Mz]...}
    if hf_loc and wlhf and bhf:  # only when horizontal feature is existing and loaded
        lp = [nl.index(y) for y in hf_y if y not in joint_y]  # node id for horizontal feature
        F = {i: [0, 1, -(0.5 * bhf + hf_gap)] for i in lp}  # by unit uplifting force
        b_hf = [Beam2(*x, 1, 1, 1, nodes) for x in beams]  # new set of beam elements
        model_hf = Bsolver(b_hf, nodes, restr, brels, node_forces=F)
        model_hf.solve()
        load_hf = True
        simple_case = False
    else:
        load_hf = False

    # # Model 2, unit udl on beams:
    if wlvf and bvf:  # vertical feature is existing and loaded
        load_vf = True
        if support2 and (not support2_lateral):  # only when secondary support doesn't provide lateral res
            b_vf = [Beam2(*x, 1, 1, 1, nodes) for x in beams]  # new set of beam elements
            restr_vf = {p: [1, 1, 0] for p in rp1}  # new retrain condition
            restr_vf.update({0: [1, 0, 0]})
            model_vf = Bsolver(b_vf, nodes, restr_vf, brels, distributed_load=Q)
            model_vf.solve()
            simple_case = False
    else:
        load_vf = False

    # endregion

    # Define combination matrix
    CF = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1],
                   [0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 0, 1, 0, 1],
                   [0, 0, 1, 1, 0, 0, 1, 1],
                   [0, 0, 1, 1, 0, 0, 1, 1],
                   [1, 1, 0, 0, 1, 1, 0, 0],
                   [1, 1, 0, 0, 1, 1, 0, 0]])

    CFA = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1, 1, 0, 0]])

    if (not load_vf) and load_hf:  # no vf case
        CF = np.delete(CF, [1, 3, 5, 7], axis=1)
        CFA = np.delete(CFA, [1, 3, 5, 7], axis=1)
    elif load_vf and (not load_hf):  # no hf case
        CF = np.delete(CF, [2, 3, 6, 7], axis=1)
        CFA = np.delete(CFA, [2, 3, 6, 7], axis=1)
    elif (not load_vf) and (not load_hf):  # not vf and no hf
        CF = np.delete(CF, [1, 2, 3, 5, 6, 7], axis=1)
        CFA = np.delete(CFA, [1, 2, 3, 5, 6, 7], axis=1)

    # Define load factor for each sections
    LF = [fw_s if section_mat[sec][0] == 'S' else fw_a for sec in secs]  # load factor in order of secs
    LFAD = [(fd_s, fd_sb) if section_mat[sec][0] == 'S' else (fd_a, fd_ab) for sec in load_app[2:4]]  # (adv, benef)
    if load_hf:
        LFAW = [fw_s if section_mat[sec][0] == 'S' else fw_a for sec in load_app[2:4]]

    # Index of loads in matrix
    wp1_loc = (locsec(load_app[0], group_y) + len(group_x), 0)
    wp2_loc = (locsec(load_app[1], group_y) + len(group_x), 1)
    ws1_loc = (locsec(load_app[0], group_y) + len(group_x), 2)
    ws2_loc = (locsec(load_app[1], group_y) + len(group_x), 3)
    dl1_loc = (secs.index(load_app[2]), 0)
    dl2_loc = (secs.index(load_app[3]), 1)
    if load_vf:
        wlvf_loc = (locsec(load_app[4], group_x), 4 if wlvf < 0 else 5)
        wlvf_flip_loc = (locsec(load_app[4], group_x), 5 if wlvf < 0 else 4)
    if load_hf:
        wlhf1_loc = (locsec(load_app[5], group_y) + len(group_x), 6 if wlhf > 0 else 8)
        wlhf2_loc = (locsec(load_app[6], group_y) + len(group_x), 7 if wlhf > 0 else 9)
        ahf1_loc = (secs.index(load_app[2]), 2 if wlhf > 0 else 4)
        ahf2_loc = (secs.index(load_app[3]), 3 if wlhf > 0 else 5)
        wlhf1_flip_loc = (locsec(load_app[5], group_y) + len(group_x), 8 if wlhf > 0 else 6)
        wlhf2_flip_loc = (locsec(load_app[6], group_y) + len(group_x), 9 if wlhf > 0 else 7)
        ahf1_flip_loc = (secs.index(load_app[2]), 4 if wlhf > 0 else 2)
        ahf2_flip_loc = (secs.index(load_app[3]), 5 if wlhf > 0 else 3)

    # region <Quick calculation for max member force & deflection of simple linear case>
    if simple_case:
        print("Quick calculation is applied for simple case...\n")
        wp, ws = wl
        # moment due to wp on left panel, (max. negative, max, positive)
        M_wp1 = -wp * b1 / 2000 * np.array(model_wl.get_summary("Moment", envelop=True))  # +Fy causes -Mx
        # shear due to wp on left panel, (max. above support, max. below support)
        V_wp1 = wp * b1 / 2000 * np.array(model_wl.get_summary("Shear", envelop=True))
        # deflection coefficient due to wp on left panel, (max. at joint, max. in span)
        d_wp1 = wp * b1 / 2000 * np.array(model_wl.get_summary("Deflection", envelop=True))  # per unit stiffness
        # moment due to wp on right panel, (max. negative, max, positive)
        M_wp2 = M_wp1 * b2 / b1
        # shear due to wp on right panel, (max. above support, max. below support)
        V_wp2 = V_wp1 * b2 / b1
        # deflection coefficient due to wp on right panel, (max. at joint, max. in span)
        d_wp2 = d_wp1 * b2 / b1
        # Calculate member force & deflection coefficient due to wind suction on panel
        M_ws1 = M_wp1 * ws / wp
        V_ws1 = V_wp1 * ws / wp
        d_ws1 = d_wp1 * ws / wp
        M_ws2 = M_wp2 * ws / wp
        V_ws2 = V_wp2 * ws / wp
        d_ws2 = d_wp2 * ws / wp
        # Calculate member force & deflection coefficient due to wind load on vertical feature
        if load_vf:
            M_vf = M_wp1 * wlvf * bvf * 2 / (-wp * b1)  # +Fx causes +My
            V_vf = V_wp1 * wlvf * bvf * 2 / (wp * b1)
            d_vf = d_wp1 * wlvf * bvf * 2 / (wp * b1)
        # Axial force due to dead load from left side
        N_d1 = panel_dens1 * b1 / 2000 * np.array(model_dl.get_summary("Axial Force", envelop=True))
        # Axial force due to dead load from right side
        N_d2 = panel_dens2 * b2 / (panel_dens1 * b1) * N_d1

        # Factored axial stress due to dead load
        stress_d1 = N_d1 * LFAD[0][0] / sec_lib[load_app[2]]['A']  # left
        stress_d2 = N_d2 * LFAD[1][0] / sec_lib[load_app[3]]['A']  # right

        # index of max. magnitude value for shear and deflection
        V_i = 0 if abs(V_wp1[0]) >= abs(V_wp1[1]) else 1
        def_i = 0 if abs(d_wp1[0]) >= abs(d_wp1[1]) else 1

        # shear matrix due to load combinations
        V_all = np.zeros((group_len, 10))
        V_all[wp1_loc] = V_wp1[V_i]
        V_all[wp2_loc] = V_wp2[V_i]
        V_all[ws1_loc] = V_ws1[V_i]
        V_all[ws2_loc] = V_ws2[V_i]
        if load_vf:  # load from vertical feature
            V_all[wlvf_loc] = V_vf[V_i]
            if wlvf_flip:  # consider the flipped case too
                V_all[wlvf_flip_loc] = -V_vf[V_i]
        sec_v = eta_v @ V_all @ CF

        # negative moment matrix due to load combinations
        Mn_all = np.zeros((group_len, 10))
        Mn_all[wp1_loc] = M_wp1[0]
        Mn_all[wp2_loc] = M_wp2[0]
        Mn_all[ws1_loc] = M_ws1[0]
        Mn_all[ws2_loc] = M_ws2[0]
        if load_vf:  # load from vertical feature
            Mn_all[wlvf_loc] = M_vf[0]
            if wlvf_flip:  # consider the flipped case too
                Mn_all[wlvf_flip_loc] = -M_vf[0]
        sec_mn = eta_m @ Mn_all @ CF

        # positive moment matrix due to load combinations
        sec_mp = sec_mn * M_wp1[1] / M_wp1[0]

        # deflection matrix due to load combinations
        d_all = np.zeros((group_len, 10))
        d_all[wp1_loc] = d_wp1[def_i]
        d_all[wp2_loc] = d_wp2[def_i]
        d_all[ws1_loc] = d_ws1[def_i]
        d_all[ws2_loc] = d_ws2[def_i]
        if load_vf:  # load from vertical feature
            d_all[wlvf_loc] = d_vf[def_i]
            if wlvf_flip:  # consider the flipped case too
                d_all[wlvf_flip_loc] = -d_vf[def_i]
        defs = es @ eta_v @ d_all @ CF

        # Result: Shear force, bending moment, stress and deflection on sections
        axial = {}
        shear = {}
        moment_n = {}
        moment_p = {}
        stress = {}
        deflection = {}
        for i in range(ns):  # Sort values in matrix into dict for output
            shear[secs[i]] = [(sec_v[2 * i, j] * LF[i], sec_v[2 * i + 1, j] * LF[i])
                              for j in range(sec_v.shape[1])]

            if secs[i] == load_app[2]:  # for section takes dl from left panel
                axial[secs[i]] = tuple(N_d1 * LFAD[0][0])
                stress_axial = stress_d1
            elif secs[i] == load_app[3]:  # for section takes dl from right panel
                axial[secs[i]] = tuple(N_d2 * LFAD[1][0])
                stress_axial = stress_d2
            else:
                axial[secs[i]] = (0, 0)  # no axial force
                stress_axial = 0

            # conservatively add max stress due to axial force and bending together
            # which may occur at difference position on a mullion
            stress_1 = np.array([biabend(sec_lib[secs[i]], sec_mn[2 * i + 1, j] * LF[i],
                                         sec_mn[2 * i, j] * LF[i])
                                 for j in range(sec_mn.shape[1])]) + stress_axial  # at support

            stress_2 = np.array([biabend(sec_lib[secs[i]], sec_mp[2 * i + 1, j] * LF[i],
                                         sec_mp[2 * i, j] * LF[i])
                                 for j in range(sec_mp.shape[1])]) + stress_axial  # at span
            stress_envelop = np.where(abs(stress_1) >= abs(stress_2), stress_1, stress_2)  # get enveloped stress
            stress[secs[i]] = [tuple(s) for s in stress_envelop]  # format to tuple of list

            # Sort moment causing max negative stress or max positive stress respectively
            _m1 = [(sec_mn[2 * i + 1, j] * LF[i], sec_mn[2 * i, j] * LF[i]) for j in range(sec_mn.shape[1])]  # (Mx, My)
            _m2 = [(sec_mp[2 * i + 1, j] * LF[i], sec_mp[2 * i, j] * LF[i]) for j in range(sec_mp.shape[1])]  # (Mx, My)
            moment_n[secs[i]] = []
            moment_p[secs[i]] = []
            for j in range(stress_1.shape[0]):
                moment_n[secs[i]].append(_m1[j] if abs(stress_1[j, 0]) >= abs(stress_2[j, 0]) else _m2[j])
                moment_p[secs[i]].append(_m1[j] if abs(stress_1[j, 1]) >= abs(stress_2[j, 1]) else _m2[j])

            # Sort deflection
            deflection[secs[i]] = [(defs[i, j], defs[ns + i, j]) for j in range(defs.shape[1])]

        # Output
        if summary:  # return summary
            max_stress = [0, 0]  # [max. stress on alum member, max stress on steel member]
            for s in secs:
                stress_list = [abs(x) for case in stress[s] for x in case]
                max_s = max(stress_list)
                index = 1 if section_mat[s][0] == 'S' else 0
                if max_stress[index] < max_s:
                    max_stress[index] = max_s
            # [max. deflection in x, max. deflection in y]
            max_deflection = [abs(defs[0:ns, :].flatten()).max(), abs(defs[ns:, :].flatten()).max()]
            return Mullion_summary(max_stress, max_deflection)

        else:  # return detail results
            # output items: N, V, Mn, Mp, D, axial, shear, moment_n, moment_p, stress, deflection
            return (Mullion_model((N_d1, N_d2), V_all, Mn_all, Mn_all * M_wp1[1] / M_wp1[0], d_all),
                    Mullion_verify(axial, shear, moment_n, moment_p, stress, deflection))

    # endregion

    # region <Member Force and Deflection Details>

    # Expression for member force & deflection due to wind pressure on panel
    wp, ws = wl
    # Function: moment due to wp on left panel, x=location, n=beam no.
    M_wp1 = lambda x, n: -wp * b1 / 2000 * model_wl.elements[n].get_M(x)  # +force causes -moment
    # Function: shear due to wp on left panel, x=location, n=beam no.
    V_wp1 = lambda x, n: wp * b1 / 2000 * model_wl.elements[n].get_V(x)
    # Function: deflection coefficient due to wp on left panel, x=location, n=beam no.
    d_wp1 = lambda x, n: wp * b1 / 2000 * model_wl.elements[n].get_Defy(x)  # per unit stiffness
    # Function: moment due to wp on right panel, x=location, n=beam no.
    M_wp2 = lambda x, n: -wp * b2 / 2000 * model_wl.elements[n].get_M(x)
    # Function: shear due to wp on right panel, x=location, n=beam no.
    V_wp2 = lambda x, n: wp * b2 / 2000 * model_wl.elements[n].get_V(x)
    # Function: deflection coefficient due to wp on right panel, x=location, n=beam no.
    d_wp2 = lambda x, n: wp * b2 / 2000 * model_wl.elements[n].get_Defy(x)  # per unit stiffness

    # Expression for member force & deflection coefficient due to wind suction on panel
    M_ws1 = lambda x, n: -ws * b1 / 2000 * model_wl.elements[n].get_M(x)
    V_ws1 = lambda x, n: ws * b1 / 2000 * model_wl.elements[n].get_V(x)
    d_ws1 = lambda x, n: ws * b1 / 2000 * model_wl.elements[n].get_Defy(x)
    M_ws2 = lambda x, n: -ws * b2 / 2000 * model_wl.elements[n].get_M(x)
    V_ws2 = lambda x, n: ws * b2 / 2000 * model_wl.elements[n].get_V(x)
    d_ws2 = lambda x, n: ws * b2 / 2000 * model_wl.elements[n].get_Defy(x)

    # Expression for member force due to dead load, x=location, n=beam no.
    N_d1 = lambda x, n: panel_dens1 * b1 / 2000 * model_dl.elements[n].get_N(x)  # left
    N_d2 = lambda x, n: panel_dens2 * b2 / 2000 * model_dl.elements[n].get_N(x)  # right

    # Expression for member force & deflection coefficient due to wind load on vertical feature
    if load_vf:
        if support2 and (not support2_lateral):  # use different model for vertical feature
            M_vf = lambda x, n: wlvf * bvf / 1000 * model_vf.elements[n].get_M(x)
            V_vf = lambda x, n: wlvf * bvf / 1000 * model_vf.elements[n].get_V(x)
            d_vf = lambda x, n: wlvf * bvf / 1000 * model_vf.elements[n].get_Defy(x)
        else:  # use same model as wind pressure
            M_vf = lambda x, n: wlvf * bvf / 1000 * model_wl.elements[n].get_M(x)
            V_vf = lambda x, n: wlvf * bvf / 1000 * model_wl.elements[n].get_V(x)
            d_vf = lambda x, n: wlvf * bvf / 1000 * model_wl.elements[n].get_Defy(x)

    # Expression for member force & deflection coefficient due to wind load on horizontal feature
    if load_hf:
        N_hf1 = lambda x, n: wlhf * bhf * b1 / 2000 * model_hf.elements[n].get_N(x)  # left
        N_hf2 = lambda x, n: wlhf * bhf * b2 / 2000 * model_hf.elements[n].get_N(x)  # right
        M_hf1 = lambda x, n: wlhf * bhf * b1 / 2000 * model_hf.elements[n].get_M(x)  # left
        M_hf2 = lambda x, n: wlhf * bhf * b2 / 2000 * model_hf.elements[n].get_M(x)  # right
        V_hf1 = lambda x, n: wlhf * bhf * b1 / 2000 * model_hf.elements[n].get_V(x)  # left
        V_hf2 = lambda x, n: wlhf * bhf * b2 / 2000 * model_hf.elements[n].get_V(x)  # right
        d_hf1 = lambda x, n: wlhf * bhf * b1 / 2000 * model_hf.elements[n].get_Defy(x)  # left
        d_hf2 = lambda x, n: wlhf * bhf * b2 / 2000 * model_hf.elements[n].get_Defy(x)  # right

    # Total matrix of all functions about shear due to load combinations
    V_all = np.empty(shape=(group_len, 10), dtype=object)
    V_all[wp1_loc] = V_wp1
    V_all[wp2_loc] = V_wp2
    V_all[ws1_loc] = V_ws1
    V_all[ws2_loc] = V_ws2
    if load_vf:  # load from vertical feature
        V_all[wlvf_loc] = V_vf
        if wlvf_flip:  # consider the flipped case too
            V_all[wlvf_flip_loc] = lambda x, n: -V_vf(x, n)
    if load_hf:  # load from horizontal feature
        V_all[wlhf1_loc] = V_hf1
        V_all[wlhf2_loc] = V_hf2
        if wlhf_flip:  # consider the flipped case too
            V_all[wlhf1_flip_loc] = lambda x, n: -V_hf1(x, n)
            V_all[wlhf2_flip_loc] = lambda x, n: -V_hf2(x, n)

    # Factory Function: return functions calculate shears on sections due to specified load combination
    def get_shear_function(sub_shear_matrix, on_section=True):

        f_loc = list(zip(*np.where(sub_shear_matrix)))

        def Vs_lc(x, n):  # function for calculating shear on sections at position x of element n
            Vs_xn = np.zeros((group_len, 5))
            for j, k in f_loc:
                Vs_xn[j, k] = sub_shear_matrix[j, k](x, n)
            return (eta_v @ Vs_xn).sum(axis=1)

        def V_lc(x, n):  # function for calculating shear as mullion member force at position x of element n
            V_xn = np.zeros((group_len, 5))
            for j, k in f_loc:
                V_xn[j, k] = sub_shear_matrix[j, k](x, n)
            return V_xn

        if on_section:
            return Vs_lc
        else:
            return V_lc

    # list of functions for calculating shear due to various load combinations
    Vs_LC = [get_shear_function(np.delete(V_all, np.where(CF[:, lcn] == 0), axis=1)) for lcn in range(CF.shape[1])]
    V_LC = [get_shear_function(np.delete(V_all, np.where(CF[:, lcn] == 0), axis=1), on_section=False)
            for lcn in range(CF.shape[1])]

    # Total matrix of all functions about moment due to load combinations
    M_all = np.empty(shape=(group_len, 10), dtype=object)
    M_all[wp1_loc] = M_wp1
    M_all[wp2_loc] = M_wp2
    M_all[ws1_loc] = M_ws1
    M_all[ws2_loc] = M_ws2
    if load_vf:  # load from vertical feature
        M_all[wlvf_loc] = M_vf
        if wlvf_flip:  # consider the flipped case too
            M_all[wlvf_flip_loc] = lambda x, n: -M_vf(x, n)
    if load_hf:  # load from horizontal feature
        M_all[wlhf1_loc] = M_hf1
        M_all[wlhf2_loc] = M_hf2
        if wlhf_flip:  # consider the flipped case too
            M_all[wlhf1_flip_loc] = lambda x, n: -M_hf1(x, n)
            M_all[wlhf2_flip_loc] = lambda x, n: -M_hf2(x, n)

    # Factory Function: return functions calculate moments on sections due to specified load combination
    def get_moment_function(sub_moment_matrix, on_section=True):

        f_loc = list(zip(*np.where(sub_moment_matrix)))

        def Ms_lc(x, n):  # function for calculating moment on section at position x of element n
            Ms_xn = np.zeros((group_len, 5))
            for j, k in f_loc:
                Ms_xn[j, k] = sub_moment_matrix[j, k](x, n)
            return (eta_m @ Ms_xn).sum(axis=1)

        def M_lc(x, n):  # function for calculating moment as mullion member force at position x of element n
            M_xn = np.zeros((group_len, 5))
            for j, k in f_loc:
                M_xn[j, k] = sub_moment_matrix[j, k](x, n)
            return M_xn

        if on_section:
            return Ms_lc
        else:
            return M_lc

    # list of functions for calculating moment due to various load combinations
    Ms_LC = [get_moment_function(np.delete(M_all, np.where(CF[:, lcn] == 0), axis=1)) for lcn in range(CF.shape[1])]
    M_LC = [get_moment_function(np.delete(M_all, np.where(CF[:, lcn] == 0), axis=1), on_section=False)
            for lcn in range(CF.shape[1])]

    # Total matrix of all functions about axial force due to load combinations
    N_all = np.empty(shape=(ns, 6), dtype=object)
    N_all[dl1_loc] = N_d1
    N_all[dl2_loc] = N_d2
    if load_hf:
        N_all[ahf1_loc] = N_hf1
        N_all[ahf2_loc] = N_hf2
        if wlhf_flip:  # consider the flipped case too
            N_all[ahf1_flip_loc] = lambda x, n: -N_hf1(x, n)
            N_all[ahf2_flip_loc] = lambda x, n: -N_hf2(x, n)

    # Factory Function: return function calculate factored axial force on sections due to specified load combination
    def get_axial_function(sub_axial_matrix, alum=True, on_section=True):

        f_loc = list(zip(*np.where(sub_axial_matrix)))

        if alum:  # section is alum
            LCFA = [fd_a, fd_a, fw_a, fw_a]  # adverse dl
            LCFA_ = [fd_ab, fd_ab, fw_a, fw_a]  # beneficial dl
        else:  # section is steel
            LCFA = [fd_s, fd_s, fw_s, fw_s]  # adverse dl
            LCFA_ = [fd_sb, fd_sb, fw_s, fw_s]  # beneficial dl

        def Ns_lc(x, n):  # function for factored axial on section at position x of element n, assuming dl is adverse
            Ns_xn = np.zeros((ns, 4))
            for j, k in f_loc:
                Ns_xn[j, k] = sub_axial_matrix[j, k](x, n)
            return Ns_xn @ LCFA

        def Ns_lc_(x, n):  # function for factored calculating axial on section at position x of element n, assuming dl is beneficial
            Ns_xn = np.zeros((ns, 4))
            for j, k in f_loc:
                Ns_xn[j, k] = sub_axial_matrix[j, k](x, n)
            return Ns_xn @ LCFA_

        def N_lc(x, n):  # function for non-factored axial member force at position x of element n
            N_xn = np.zeros((ns, 4))
            for j, k in f_loc:
                N_xn[j, k] = sub_axial_matrix[j, k](x, n)
            return N_xn

        if on_section:
            return Ns_lc, Ns_lc_
        else:
            return N_lc

    # list of functions for calculating axial force due to various load combinations
    Ns_LC_a = [get_axial_function(np.delete(N_all, np.where(CFA[:, lcn] == 0), axis=1), alum=True)
               for lcn in range(CFA.shape[1])]
    Ns_LC_s = [get_axial_function(np.delete(N_all, np.where(CFA[:, lcn] == 0), axis=1), alum=False)
               for lcn in range(CFA.shape[1])]
    N_LC = [get_axial_function(np.delete(N_all, np.where(CFA[:, lcn] == 0), axis=1), on_section=False)
            for lcn in range(CFA.shape[1])]

    # Total matrix of all functions about deflection due to load combinations
    d_all = np.empty(shape=(group_len, 10), dtype=object)
    d_all[wp1_loc] = d_wp1
    d_all[wp2_loc] = d_wp2
    d_all[ws1_loc] = d_ws1
    d_all[ws2_loc] = d_ws2
    if load_vf:  # load from vertical feature
        d_all[wlvf_loc] = d_vf
        if wlvf_flip:  # consider the flipped case too
            d_all[wlvf_flip_loc] = lambda x, n: -d_vf(x, n)
    if load_hf:  # load from horizontal feature
        d_all[wlhf1_loc] = d_hf1
        d_all[wlhf2_loc] = d_hf2
        if wlhf_flip:  # consider the flipped case too
            d_all[wlhf1_flip_loc] = lambda x, n: -d_hf1(x, n)
            d_all[wlhf2_flip_loc] = lambda x, n: -d_hf2(x, n)

    # Factory Function: return functions calculate deflection of sections due to specified load combination
    def get_deflection_function(sub_deflection_matrix, on_section=True):

        f_loc = list(zip(*np.where(sub_deflection_matrix)))

        def ds_lc(x, n):  # function for calculating deflection of sections at position x of element n
            ds_xn = np.zeros((group_len, 5))
            for j, k in f_loc:
                ds_xn[j, k] = sub_deflection_matrix[j, k](x, n)
            return (es @ eta_v @ ds_xn).sum(axis=1)

        def d_lc(x, n):  # function for calculating overall deflection of mullion member at position x of element n
            d_xn = np.zeros((group_len, 5))
            for j, k in f_loc:
                d_xn[j, k] = sub_deflection_matrix[j, k](x, n)
            return d_xn

        if on_section:
            return ds_lc
        else:
            return d_lc

    # list of functions for calculating deflection due to various load combinations
    ds_LC = [get_deflection_function(np.delete(d_all, np.where(CF[:, lcn] == 0), axis=1)) for lcn in range(CF.shape[1])]
    d_LC = [get_deflection_function(np.delete(d_all, np.where(CF[:, lcn] == 0), axis=1), on_section=False)
            for lcn in range(CF.shape[1])]

    # endregion

    # Find Max Total Fiber Stress on Each Section Due to Load Combinations
    stress = {}
    critical_M = {}
    Mn = {}
    Mp = {}
    moment_n = {}
    moment_p = {}
    for i in range(ns):  # for different section
        stress[secs[i]] = []  # initialize the list of stress
        critical_M[secs[i]] = []  # initialize the list of critical position on members
        Mn[secs[i]] = []  # initialize the list of moment matrix for max negative stress
        Mp[secs[i]] = []  # initialize the list of moment matrix for max positive stress
        moment_n[secs[i]] = []  # initialize the list of moment on section causing ma negative stress
        moment_p[secs[i]] = []  # initialize the list of moment on section causing ma positive stress
        area = sec_lib[secs[i]]['A']  # section area
        for c in range(CF.shape[1]):  # for different load combinations
            local_max = [0., 0.]  # [max. negative stress, max. positive stress]
            position_s1 = (0, 0)  # (x, n)
            position_s2 = (0, 0)  # (x, n)
            fs_axial = Ns_LC_s[c] if section_mat[secs[i]][0] == 'S' else Ns_LC_a[c]  # function for axial stress

            for bn in range(len(beams)):  # for different span
                s1 = minimize(lambda x: biabend(sec_lib[secs[i]], Ms_LC[c](x, bn)[i * 2 + 1], Ms_LC[c](x, bn)[i * 2])[0] * LF[i]
                              + fs_axial[0](x, bn)[i] / area, 0.5, bounds=((0, 1),))
                s1_ = minimize(lambda x: biabend(sec_lib[secs[i]], Ms_LC[c](x, bn)[i * 2 + 1], Ms_LC[c](x, bn)[i * 2])[0] * LF[i]
                               + fs_axial[1](x, bn)[i] / area, 0.5, bounds=((0, 1),))
                s2 = minimize(lambda x: -(biabend(sec_lib[secs[i]], Ms_LC[c](x, bn)[i * 2 + 1], Ms_LC[c](x, bn)[i * 2])[1] * LF[i]
                              + fs_axial[0](x, bn)[i] / area), 0.5, bounds=((0, 1),))
                s2_ = minimize(lambda x: -(biabend(sec_lib[secs[i]], Ms_LC[c](x, bn)[i * 2 + 1], Ms_LC[c](x, bn)[i * 2])[1] * LF[i]
                               + fs_axial[1](x, bn)[i] / area), 0.5, bounds=((0, 1),))
                x = (s1.x[0] if s1.fun < s1_.fun else s1_.x[0],
                     s2.x[0] if s2.fun < s2_.fun else s2_.x[0])
                extreme = [min(s1.fun, s1_.fun), max(-s2.fun, -s2_.fun)]
                # record the corresponding moments and location
                if extreme[0] < local_max[0]:  # record the max. negative stress
                    local_max[0] = extreme[0]
                    position_s1 = (x[0], bn)
                if extreme[1] > local_max[1]:  # record the max. positive stress
                    local_max[1] = extreme[1]
                    position_s2 = (x[1], bn)

            stress[secs[i]].append(local_max)
            critical_M[secs[i]].append([position_s1, position_s2])
            Mn[secs[i]].append(M_LC[c](*position_s1))  # moment matrix for max negative stress
            Mp[secs[i]].append(M_LC[c](*position_s2))   # moment matrix for max positive stress
            # Sort moment on section causing max negative stress or max positive stress respectively
            moment_n[secs[i]].append((Ms_LC[c](*position_s1)[i * 2 + 1] * LF[i], Ms_LC[c](*position_s1)[i * 2] * LF[i]))
            moment_p[secs[i]].append((Ms_LC[c](*position_s2)[i * 2 + 1] * LF[i], Ms_LC[c](*position_s2)[i * 2] * LF[i]))

    # Find Max Deflection on Each Section Due to Load Combinations
    deflection = {}
    critical_disp = {}
    D = {}
    for i in range(ns):  # for different section
        deflection[secs[i]] = []  # initialize the list of deflection
        critical_disp[secs[i]] = []   # initialize the list of critical position on members
        D[secs[i]] = []  # initialize the list of deflection matrix for critical cases
        for c in range(CF.shape[1]):  # for different load combinations
            local_max = [0., 0.]  # [max. deflection in x, max. deflection in y]
            position_dx = (0, 0)  # (x, n)
            position_dy = (0, 0)  # (x, n)
            for bn in range(len(beams)):  # for different span
                x_dx = minimize(lambda x: -abs(ds_LC[c](x, bn)[i]), 0.5, bounds=((0, 1),)).x[0]
                x_dy = minimize(lambda x: -abs(ds_LC[c](x, bn)[i + ns]), 0.5, bounds=((0, 1),)).x[0]
                extreme = [ds_LC[c](x_dx, bn)[i], ds_LC[c](x_dy, bn)[i+ns]]
                if abs(extreme[0]) > abs(local_max[0]):  # record the max. deflection in x
                    local_max[0] = extreme[0]
                    position_dx = (x_dx, bn)
                if abs(extreme[1]) > abs(local_max[1]):  # record the max. deflection in y
                    local_max[1] = extreme[1]
                    position_dy = (x_dy, bn)
            deflection[secs[i]].append(local_max)
            critical_disp[secs[i]].append([position_dx, position_dy])
            D[secs[i]].append((d_LC[c](*position_dx), d_LC[c](*position_dy)))

    # Output
    if summary:  # return summary
        max_stress = [0, 0]  # [max. stress on alum member, max stress on steel member]
        max_deflection = [0, 0]  # [max. deflection in x, max. deflection in y]
        for s in secs:
            stress_list = [abs(x) for case in stress[s] for x in case]
            deflection_list = list(zip(*[np.abs(x) for x in deflection[s]]))
            max_s = max(stress_list)
            max_dx = max(deflection_list[0])
            max_dy = max(deflection_list[1])
            index = 1 if section_mat[s][0] == 'S' else 0
            if max_stress[index] < max_s:
                max_stress[index] = max_s
            if max_dx > max_deflection[0]:
                max_deflection[0] = max_dx
            if max_dy > max_deflection[1]:
                max_deflection[1] = max_dy
        return Mullion_summary(max_stress, max_deflection)

    else:  # return detail results

        # Find Max Shear on Each Section Due to Load Combinations
        shear = {}
        critical_V = {}
        V = {}
        for i in range(ns):  # for different section
            shear[secs[i]] = []  # initialize the list of deflection
            critical_V[secs[i]] = []  # initialize the list of critical position on members
            V[secs[i]] = []   # initialize the list of shear matrix for critical cases
            for c in range(CF.shape[1]):  # for different load combinations
                local_max = [0., 0.]  # [max. shear in x, max. shear in y]
                position_Vx = (0, 0)  # (x, n)
                position_Vy = (0, 0)  # (x, n)
                for bn in range(len(beams)):  # for different span
                    x_Vx = minimize(lambda x: -abs(Vs_LC[c](x, bn)[i * 2]), 0.5, bounds=((0, 1),)).x[0]
                    x_Vy = minimize(lambda x: -abs(Vs_LC[c](x, bn)[i * 2 + 1]), 0.5, bounds=((0, 1),)).x[0]
                    extreme = [Vs_LC[c](x_Vx, bn)[i * 2] * LF[i], Vs_LC[c](x_Vy, bn)[i * 2 + 1] * LF[i]]
                    if abs(extreme[0]) > abs(local_max[0]):  # record the max. shear in x
                        local_max[0] = extreme[0]
                        position_Vx = (x_Vx, bn)
                    if abs(extreme[1]) > abs(local_max[1]):  # record the max. shear in y
                        local_max[1] = extreme[1]
                        position_Vy = (x_Vy, bn)
                shear[secs[i]].append(tuple(local_max))
                critical_V[secs[i]].append((position_Vx, position_Vy))
                V[secs[i]].append((V_LC[c](*position_Vx), V_LC[c](*position_Vy)))

        # Find Max Axial Force on Each section Due to Load Combinations
        # Note: this max axial force may not same as the axial force combining with bending at critical section
        axial = {}
        critical_N = {}
        N = {}
        for i in range(ns):  # for different section
            axial[secs[i]] = []  # initialize the list of axial force
            critical_N[secs[i]] = []   # initialize the list of critical position on members
            N[secs[i]] = []  # initialize the list of axial force matrix for critical cases
            for c in range(CF.shape[1]):  # for different load combinations
                local_max = [0., 0.]  # [max. compression, max. tension]
                position_c = (0, 0)   # (x, n)
                position_t = (0, 0)   # (x, n)
                fs_axial = Ns_LC_s[c] if section_mat[secs[i]][0] == 'S' else Ns_LC_a[c]  # function for axial stress
                for bn in range(len(beams)):  # for different span
                    cpn = minimize(lambda x: fs_axial[0](x, bn)[i], 0.5, bounds=((0, 1),))
                    cpn_ = minimize(lambda x: fs_axial[1](x, bn)[i], 0.5, bounds=((0, 1),))
                    ten = minimize(lambda x: -fs_axial[0](x, bn)[i], 0.5, bounds=((0, 1),))
                    ten_ = minimize(lambda x: -fs_axial[1](x, bn)[i], 0.5, bounds=((0, 1),))
                    x = (cpn.x[0] if cpn.fun < cpn_.fun else cpn_.x[0],
                         ten.x[0] if ten.fun < ten_.fun else ten_.x[0])
                    extreme = [min(cpn.fun, cpn_.fun), max(-ten.fun, -ten_.fun)]
                    # record the corresponding moments and location
                    if extreme[0] < local_max[0]:  # record the max. compression
                        local_max[0] = extreme[0]
                        position_c = (x[0], bn)
                    if extreme[1] > local_max[1]:  # record the max. tension
                        local_max[1] = extreme[1]
                        position_t = (x[1], bn)
                axial[secs[i]].append(tuple(local_max))
                critical_N[secs[i]].append((position_c, position_t))
                N[secs[i]].append((N_LC[c](*position_c), N_LC[c](*position_t)))

        # Output items: N, V, Mn, Mp, D, axial, shear, moment_n, moment_p, stress, deflection
        return (Mullion_model(N, V, Mn, Mp, D),
                Mullion_verify(axial, shear, moment_n, moment_p, stress, deflection),
                Mullion_critical(critical_N, critical_V, critical_M, critical_disp))


def check_mullions_varsec(section_lib, section_mat, section_combs, height, b1, b2, load_app, wl, panel_dens1, panel_dens2,
                          support1, support2=None, support2_lateral=False, wlvf=0.0, wlhf=0.0, bvf=0.0, bhf=0.0, hf_gap=0.0,
                          hf_loc=None, wlvf_flip=True, wlhf_flip=True, summary=False):
    """Verification of mullion of combined sections inconsistently.

    :param section_lib: str or dict, section library stating section properties and boundary information. In form of
                        str as path and name of json file created by ``pyacad.Acad.seclib``, or a nested dict in the
                        same form.
    :param section_mat: dict, material of involved sections, {sec_name_1: material_1, sec_name_2: material_2, ...}
    :param section_combs: nested list of section combinations, [comb1, comb2, comb3 ...] in same order as mullion span
                          **from Bottom to Top**, each combination is in the form of
                          [[sec_name_1, sec_name_2, comb_indicator],...]. The *comb_indicator* is one of below:

                          | 'x', 'X' or 1 - combine in x-direction
                          | 'y', 'Y' or 2 - combine in y-direction
                          | 'xy', 'XY' or 0 - combine in both directions

    :param height: list of float, height of mullions from lower to upper, unit = mm.
    :param b1: float, width of panel at left side of mullion, unit = mm.
    :param b2: float, width of panel at right side of mullion, unit = mm.
    :param load_app: list of str, name of sections takes loads
                    [for_wl1, for_wl2, for_dl1, for_dl2, for_wlvf, for_wlhf1, for_wlhf2]
                    the last three can be 'None' when corresponding load does not exist
    :param wl: list of float, design wind load on panel, [pressure, suction], unit = kPa, positive as pressure,
               negative as suction
    :param panel_dens1: float, design area density of panel at left side of mullion, unit = kN/m :superscript:`2`.
    :param panel_dens2: float, design area density of panel at right side of mullion, unit = kN/m :superscript:`2`.
    :param support1: list of float, distance from top end of mullion to primary support, unit = mm.
    :param support2: list of float, distance from primary support to secondary support, unit = mm.
                     No secondary support in that span if corresponding item in list is 0.
    :param support2_lateral: bool, secondary support also provides lateral restrain.
    :param wlvf: float, design wind load on vertical feature, unit = kPa, positive as rightward, negative as leftward.
    :param wlhf: float, design wind load on horizontal feature, unit = kPa, positive as uplifting, negative as downward.
    :param bvf: float, windward breadth of vertical feature, unit = mm.
    :param bhf: float, windward breadth of horizontal feature, unit = mm.
    :param hf_gap: float, gap between horizontal feature and centroid of connected section, unit = mm.
    :param hf_loc: list, distance of horizontal features to top end of mullion at each span, unit = mm.
                   No horizontal feature in that span if corresponding item in list is 0.
    :param wlvf_flip: bool, also check the case that wind load direction on vertical feature is flipped.
    :param wlhf_flip: bool, also check the case that wind load direction on horizontal feature is flipped.
    :param summary: bool, output the verification summary only.

    :return: if *summary* = True, return a namedtuple ``Mullion_summary``. otherwise return 3 namedtuple,
             ``Mullion_model``, ``Mullion_verify`` and ``Mullion_critical``.

    """

    # Define load factor for alum. member
    fd_a = 1.2  # adverse dl
    fd_ab = 0.8  # beneficial dl
    fw_a = 1.4  # wl

    # Define load factor for steel member
    fd_s = 1.4  # adverse dl
    fd_sb = 1.0  # beneficial dl
    fw_s = 1.4  # wl

    # Initialize section properties
    if type(section_lib) == str and section_lib[-4:] == 'json':  # read from specified jason file
        with open(section_lib) as f:
            sec_lib = json.load(f)
    elif type(section_lib) == dict:  # read directly as a sorted dictionary
        sec_lib = section_lib
    else:
        raise ValueError("Unsupported Type of Section Library")

    # Record section component name of each span, from bottom to top
    total_secs = list(section_mat.keys())  # all related section order follows the input dict of section material
    secs = []  # sections on each span
    ns = []  # number of sections involved in each span
    for sc in section_combs:
        _sl = sorted(list(set([s for pair in sc for s in pair[:2]])), key=total_secs.index)
        secs.append(_sl)
        ns.append(len(_sl))

    # Calculate combination-related data of each span
    spans = len(height)  # number of total spans
    group_x, group_y, eta_v, eta_m, es = zip(*[combsec(sec_lib, {ind: section_mat[ind] for ind in secs[i]},
                                                       section_combs[i])
                                               for i in range(spans)])

    group_len = [len(group_x[i])+len(group_y[i]) for i in range(spans)]

    def locsec(section, grp):
        for g in grp:
            if section in g:
                return grp.index(g)
        else:
            raise ValueError(f"Can't find <{section}> in <{grp}>.")

    # Calculate equivalent moment of inertia of combined sections, based on E=1
    I_eq = []
    for i in range(spans):
        I_wl1 = 1/(es[i][secs[i].index(load_app[0])+ns[i],:] @ eta_v[i][:,locsec(load_app[0], group_y[i]) + len(group_x[i])])
        I_wl2 = 1/(es[i][secs[i].index(load_app[1])+ns[i],:] @ eta_v[i][:,locsec(load_app[1], group_y[i]) + len(group_x[i])])
        if load_app[4]:
            I_wlvf = 1 / (es[i][secs[i].index(load_app[4]), :] @ eta_v[i][:, locsec(load_app[4], group_x[i])])
        else:
            I_wlvf = None
        if load_app[5]:
            I_wlhf1 = 1 / (es[i][secs[i].index(load_app[5]) + ns[i], :] @ eta_v[i][:,
                                                                        locsec(load_app[5], group_y[i]) + len(
                                                                            group_x[i])])
        else:
            I_wlhf1 = None
        if load_app[6]:
            I_wlhf2 = 1 / (es[i][secs[i].index(load_app[6]) + ns[i], :] @ eta_v[i][:,
                                                                        locsec(load_app[6], group_y[i]) + len(
                                                                            group_x[i])])
        else:
            I_wlhf2 = None
        I_eq.append([I_wl1, I_wl2, I_wlvf, I_wlhf1, I_wlhf2])
    I_eq = np.array(I_eq).T  # sort by load cases as row order and spans as column order

    # region <Calculation Model of Mullion>  -->> model_wl1, model_wl2, model_dl, model_vf, model_hf1, model_hf2

    # Define Coordinate of Node in format: [[x1,y1],[x2,y2]...]
    joint_y = np.array([sum(height[:i + 1]) for i in range(spans)])  # location of stack joint
    support1_y = joint_y - np.array(support1)  # location of stack joint
    if support2:  # secondary support is defined
        support2_y = support1_y - np.array(support2)
    else:
        support2_y = []
    hf_y = []
    if hf_loc:  # location of horizontal feature is defined
        for i in range(len(hf_loc)):
            try:
                for hf in hf_loc[i]:  # try if multiple features on one span
                    hf_y.append(joint_y[i] - hf)
            except TypeError:
                hf_y.append(joint_y[i] - hf_loc[i])  # single feature on one span
    nodes_y = np.unique(np.concatenate(([0], joint_y, support1_y, support2_y, hf_y)))  # drop the duplicated node
    nodes_y.sort()  # sort the nodes from bottom to top
    nl = list(nodes_y)  # list of y-coordinates of sorted nodes
    nodes = [[0, y] for y in nodes_y]

    # Define Beam Set:
    beams = [[i, i + 1] for i in range(len(nodes) - 1)]

    # Record the mapping of Beam Id to mullion span
    joint_id = [nl.index(y) for y in joint_y]  # node id of stack joint
    sn_ = 0
    mu = {}
    for k in range(len(beams)):
        if k >= joint_id[sn_]:
            sn_ += 1
        mu[k] = sn_

    # Create list of beam object for each load application [for_wl1, for_wl2, for_wlvf, for_wlhf1, for_wlhf2]
    bs = []
    for Is in I_eq:  # for each load case
        bs.append([Beam2(*x, 1, Is[mu[x[0]]], 1, nodes) for x in beams])

    # Define restrain in format: {node:[x, y, rotate]...}, 1=restrained
    rp1 = [nl.index(y) for y in support1_y]  # node id for primary support
    restr = {p: [1, 1, 0] for p in rp1}
    restr.update({0: [1, 0, 0]})
    if support2:
        rp2 = [nl.index(y) for y in support2_y if y not in support1_y]  # node id for secondary support
        restr.update({p: [1, 0, 0] for p in rp2})  # add restrains of secondary support

    # Define end release of each beam in format: {beam:[(start_T, start_R), (end_T, end_R)]...},  1=released>>>
    jp = [nl.index(y) for y in joint_y]  # node id for stack joint
    brels = {i: [(1, 1), (0, 0)] for i in range(len(beams)) if beams[i][0] in jp}

    # *Model for wl1 - PRESSURE: udl on beams:
    Q = {i: (0, wl[0] * b1 / 2000) for i in range(len(beams))}
    model_wp1 = Bsolver(bs[0], nodes, restr, brels, distributed_load=Q)
    model_wp1.solve()

    # *Model for wl2 - PRESSURE: udl on beams:
    Q = {i: (0, wl[0] * b2 / 2000) for i in range(len(beams))}
    model_wp2 = Bsolver(bs[1], nodes, restr, brels, distributed_load=Q)
    model_wp2.solve()

    # *Model for dl1 & dl2: UNIT ual on beams:
    Qa = {i: (-1, 0) for i in range(len(beams))}  # dl always downward
    b_dl = [Beam2(*x, 1, 1, 1, nodes) for x in beams]  # new set of beam elements
    model_dl = Bsolver(b_dl, nodes, restr, brels, distributed_load=Qa)
    model_dl.solve()

    # *Model for wlhf1 & wlhf2: node force in format: {node:[Fx, Fy, Mz]...}
    if hf_loc and wlhf and bhf:  # only when horizontal feature is existing and loaded
        lp = [nl.index(y) for y in hf_y if y not in joint_y]  # node id for horizontal feature
        F1 = {i: [0, wlhf * bhf * b1 / 2000, -(0.5 * bhf + hf_gap) * wlhf * bhf * b1 / 2000] for i in lp}
        model_hf1 = Bsolver(bs[3], nodes, restr, brels, node_forces=F1)
        model_hf1.solve()
        F2 = {i: [0, wlhf * bhf * b2 / 2000, -(0.5 * bhf + hf_gap) * wlhf * bhf * b2 / 2000] for i in lp}
        model_hf2 = Bsolver(bs[4], nodes, restr, brels, node_forces=F2)
        model_hf2.solve()
        load_hf = True
    else:
        load_hf = False

    # *Model for wlvf:  udl on beams
    if wlvf and bvf:  # vertical feature is existing and loaded
        load_vf = True
        Q = {i: (0, wlvf * bvf / 1000) for i in range(len(beams))}
        if support2 and (not support2_lateral):  # only when secondary support doesn't provide lateral res
            restr_vf = {p: [1, 1, 0] for p in rp1}  # new retrain condition
            restr_vf.update({0: [1, 0, 0]})
            model_vf = Bsolver(bs[2], nodes, restr_vf, brels, distributed_load=Q)
        else:
            model_vf = Bsolver(bs[2], nodes, restr, brels, distributed_load=Q)
        model_vf.solve()
    else:
        load_vf = False

    # endregion

    # Define combination matrix
    CF = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1],
                   [0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 1, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 0, 1, 0, 1],
                   [0, 0, 1, 1, 0, 0, 1, 1],
                   [0, 0, 1, 1, 0, 0, 1, 1],
                   [1, 1, 0, 0, 1, 1, 0, 0],
                   [1, 1, 0, 0, 1, 1, 0, 0]])

    CFA = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [1, 1, 0, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1, 1, 0, 0]])

    if (not load_vf) and load_hf:  # no vf case
        CF = np.delete(CF, [1, 3, 5, 7], axis=1)
        CFA = np.delete(CFA, [1, 3, 5, 7], axis=1)
    elif load_vf and (not load_hf):  # no hf case
        CF = np.delete(CF, [2, 3, 6, 7], axis=1)
        CFA = np.delete(CFA, [2, 3, 6, 7], axis=1)
    elif (not load_vf) and (not load_hf):  # not vf and no hf
        CF = np.delete(CF, [1, 2, 3, 5, 6, 7], axis=1)
        CFA = np.delete(CFA, [1, 2, 3, 5, 6, 7], axis=1)

    # Define load factor for each sections
    LF = {sec: fw_s if section_mat[sec][0] == 'S' else fw_a for sec in total_secs}  # load factor in order of total_secs
    LFAD = [(fd_s, fd_sb) if section_mat[sec][0] == 'S' else (fd_a, fd_ab) for sec in load_app[2:4]]  # (adv, benef)
    if load_hf:
        LFAW = [fw_s if section_mat[sec][0] == 'S' else fw_a for sec in load_app[2:4]]

    # Index of loads in matrix
    wp1_loc = []
    wp2_loc = []
    ws1_loc = []
    ws2_loc = []
    dl1_loc = []
    dl2_loc = []
    wlvf_loc =[]
    wlvf_flip_loc = []
    wlhf1_loc = []
    wlhf2_loc = []
    wlhf1_flip_loc = []
    wlhf2_flip_loc = []
    ahf1_loc = []
    ahf2_loc = []
    ahf1_flip_loc = []
    ahf2_flip_loc = []
    for i in range(spans):  # sorted by different span
        wp1_loc.append((locsec(load_app[0], group_y[i]) + len(group_x[i]), 0))
        wp2_loc.append((locsec(load_app[1], group_y[i]) + len(group_x[i]), 1))
        ws1_loc.append((locsec(load_app[0], group_y[i]) + len(group_x[i]), 2))
        ws2_loc.append((locsec(load_app[1], group_y[i]) + len(group_x[i]), 3))
        dl1_loc.append((secs[i].index(load_app[2]), 0))
        dl2_loc.append((secs[i].index(load_app[3]), 1))

        if load_vf:
            wlvf_loc.append((locsec(load_app[4], group_x[i]), 4 if wlvf < 0 else 5))
            wlvf_flip_loc.append((locsec(load_app[4], group_x[i]), 5 if wlvf < 0 else 4))
        if load_hf:
            wlhf1_loc.append((locsec(load_app[5], group_y[i]) + len(group_x[i]), 6 if wlhf > 0 else 8))
            wlhf2_loc.append((locsec(load_app[6], group_y[i]) + len(group_x[i]), 7 if wlhf > 0 else 9))
            wlhf1_flip_loc.append((locsec(load_app[5], group_y[i]) + len(group_x[i]), 8 if wlhf > 0 else 6))
            wlhf2_flip_loc.append((locsec(load_app[6], group_y[i]) + len(group_x[i]), 9 if wlhf > 0 else 7))
            ahf1_loc.append((secs[i].index(load_app[2]), 2 if wlhf > 0 else 4))
            ahf2_loc.append((secs[i].index(load_app[3]), 3 if wlhf > 0 else 5))
            ahf1_flip_loc.append((secs[i].index(load_app[2]), 4 if wlhf > 0 else 2))
            ahf2_flip_loc.append((secs[i].index(load_app[3]), 5 if wlhf > 0 else 3))

    # region <Member Force and Deflection Details>

    # Function: moment due to wp on left panel, x=location, n=beam no.
    M_wp1 = lambda x, n: -model_wp1.elements[n].get_M(x)  # +force causes -moment
    # Function: shear due to wp on left panel, x=location, n=beam no.
    V_wp1 = lambda x, n: model_wp1.elements[n].get_V(x)
    # Function: deflection due to wp on left panel, x=location, n=beam no.
    d_wp1 = lambda x, n: model_wp1.elements[n].get_Defy(x) * I_eq[0][mu[n]]  # per unit stiffness
    # Function: moment due to wp on right panel, x=location, n=beam no.
    M_wp2 = lambda x, n: -model_wp2.elements[n].get_M(x)
    # Function: shear due to wp on right panel, x=location, n=beam no.
    V_wp2 = lambda x, n: model_wp2.elements[n].get_V(x)
    # Function: deflection coefficient due to wp on right panel, x=location, n=beam no.
    d_wp2 = lambda x, n: model_wp2.elements[n].get_Defy(x) * I_eq[1][mu[n]]  # per unit stiffness

    # Expression for member force & deflection coefficient due to wind suction on panel
    M_ws1 = lambda x, n: - wl[1] / wl[0] * model_wp1.elements[n].get_M(x)
    V_ws1 = lambda x, n: wl[1] / wl[0] * model_wp1.elements[n].get_V(x)
    d_ws1 = lambda x, n: wl[1] / wl[0] * model_wp1.elements[n].get_Defy(x) * I_eq[0][mu[n]]
    M_ws2 = lambda x, n: -wl[1] / wl[0] * model_wp2.elements[n].get_M(x)
    V_ws2 = lambda x, n: wl[1] / wl[0] * model_wp2.elements[n].get_V(x)
    d_ws2 = lambda x, n: wl[1] / wl[0] * model_wp2.elements[n].get_Defy(x) * I_eq[1][mu[n]]

    # Expression for member force due to dead load, x=location, n=beam no.
    N_d1 = lambda x, n: panel_dens1 * b1 / 2000 * model_dl.elements[n].get_N(x)  # left
    N_d2 = lambda x, n: panel_dens2 * b2 / 2000 * model_dl.elements[n].get_N(x)  # right

    # Expression for member force & deflection coefficient due to wind load on vertical feature
    if load_vf:
        M_vf = lambda x, n: model_vf.elements[n].get_M(x)
        V_vf = lambda x, n: model_vf.elements[n].get_V(x)
        d_vf = lambda x, n: model_vf.elements[n].get_Defy(x) * I_eq[2][mu[n]]

    # Expression for member force & deflection coefficient due to wind load on horizontal feature
    if load_hf:
        N_hf1 = lambda x, n: model_hf1.elements[n].get_N(x)  # left
        N_hf2 = lambda x, n: model_hf2.elements[n].get_N(x)  # right
        M_hf1 = lambda x, n: model_hf1.elements[n].get_M(x)  # left
        M_hf2 = lambda x, n: model_hf2.elements[n].get_M(x)  # right
        V_hf1 = lambda x, n: model_hf1.elements[n].get_V(x)  # left
        V_hf2 = lambda x, n: model_hf2.elements[n].get_V(x)  # right
        d_hf1 = lambda x, n: model_hf1.elements[n].get_Defy(x) * I_eq[3][mu[n]]  # left
        d_hf2 = lambda x, n: model_hf2.elements[n].get_Defy(x) * I_eq[4][mu[n]]  # right

    # Total matrix of all functions about shear due to load combinations
    V_all = []
    for i in range(spans):
        V_all_ = np.empty(shape=(group_len[i], 10), dtype=object)
        V_all_[wp1_loc[i]] = V_wp1
        V_all_[wp2_loc[i]] = V_wp2
        V_all_[ws1_loc[i]] = V_ws1
        V_all_[ws2_loc[i]] = V_ws2
        if load_vf:  # load from vertical feature
            V_all_[wlvf_loc[i]] = V_vf
            if wlvf_flip:  # consider the flipped case too
                V_all_[wlvf_flip_loc[i]] = lambda x, n: -V_vf(x, n)
        if load_hf:  # load from horizontal feature
            V_all_[wlhf1_loc[i]] = V_hf1
            V_all_[wlhf2_loc[i]] = V_hf2
            if wlhf_flip:  # consider the flipped case too
                V_all_[wlhf1_flip_loc[i]] = lambda x, n: -V_hf1(x, n)
                V_all_[wlhf2_flip_loc[i]] = lambda x, n: -V_hf2(x, n)
        V_all.append(V_all_)

    # Factory Function: return functions calculate shears on sections due to specified load combination
    def get_shear_function(span_no, sub_shear_matrix, on_section=True):

        f_loc = list(zip(*np.where(sub_shear_matrix)))

        def Vs_lc(x, n):  # function for calculating shear on sections at position x of element n
            Vs_xn = np.zeros(sub_shear_matrix.shape)
            for j, k in f_loc:
                Vs_xn[j, k] = sub_shear_matrix[j, k](x, n)
            return (eta_v[span_no] @ Vs_xn).sum(axis=1)

        def V_lc(x, n):  # function for calculating shear as mullion member force at position x of element n
            V_xn = np.zeros(sub_shear_matrix.shape)
            for j, k in f_loc:
                V_xn[j, k] = sub_shear_matrix[j, k](x, n)
            return V_xn

        if on_section:
            return Vs_lc
        else:
            return V_lc

    # list of functions for calculating shear due to various load combinations
    Vs_LC = []
    V_LC = []
    for i in range(spans):
        Vs_LC.append([get_shear_function(i, np.delete(V_all[i], np.where(CF[:, lcn] == 0), axis=1))
                      for lcn in range(CF.shape[1])])
        V_LC.append([get_shear_function(i, np.delete(V_all[i], np.where(CF[:, lcn] == 0), axis=1), on_section=False)
                     for lcn in range(CF.shape[1])])

    # Total matrix of all functions about moment due to load combinations
    M_all = []
    for i in range(spans):
        M_all_ = np.empty(shape=(group_len[i], 10), dtype=object)
        M_all_[wp1_loc[i]] = M_wp1
        M_all_[wp2_loc[i]] = M_wp2
        M_all_[ws1_loc[i]] = M_ws1
        M_all_[ws2_loc[i]] = M_ws2
        if load_vf:  # load from vertical feature
            M_all_[wlvf_loc[i]] = M_vf
            if wlvf_flip:  # consider the flipped case too
                M_all_[wlvf_flip_loc[i]] = lambda x, n: -M_vf(x, n)
        if load_hf:  # load from horizontal feature
            M_all_[wlhf1_loc[i]] = M_hf1
            M_all_[wlhf2_loc[i]] = M_hf2
            if wlhf_flip:  # consider the flipped case too
                M_all_[wlhf1_flip_loc[i]] = lambda x, n: -M_hf1(x, n)
                M_all_[wlhf2_flip_loc[i]] = lambda x, n: -M_hf2(x, n)
        M_all.append(M_all_)

    # Factory Function: return functions calculate moments on sections due to specified load combination
    def get_moment_function(span_no, sub_moment_matrix, on_section=True):

        f_loc = list(zip(*np.where(sub_moment_matrix)))

        def Ms_lc(x, n):  # function for calculating moment on section at position x of element n
            Ms_xn = np.zeros(sub_moment_matrix.shape)
            for j, k in f_loc:
                Ms_xn[j, k] = sub_moment_matrix[j, k](x, n)
            return (eta_m[span_no] @ Ms_xn).sum(axis=1)

        def M_lc(x, n):  # function for calculating moment as mullion member force at position x of element n
            M_xn = np.zeros(sub_moment_matrix.shape)
            for j, k in f_loc:
                M_xn[j, k] = sub_moment_matrix[j, k](x, n)
            return M_xn

        if on_section:
            return Ms_lc
        else:
            return M_lc

    # list of functions for calculating moment due to various load combinations
    Ms_LC = []
    M_LC = []
    for i in range(spans):
        Ms_LC.append([get_moment_function(i, np.delete(M_all[i], np.where(CF[:, lcn] == 0), axis=1))
                      for lcn in range(CF.shape[1])])
        M_LC.append([get_moment_function(i, np.delete(M_all[i], np.where(CF[:, lcn] == 0), axis=1), on_section=False)
                     for lcn in range(CF.shape[1])])

    # Total matrix of all functions about axial force due to load combinations
    N_all = []
    for i in range(spans):
        N_all_ = np.empty(shape=(ns[i], 6), dtype=object)
        N_all_[dl1_loc[i]] = N_d1
        N_all_[dl2_loc[i]] = N_d2
        if load_hf:
            N_all_[ahf1_loc[i]] = N_hf1
            N_all_[ahf2_loc[i]] = N_hf2
            if wlhf_flip:  # consider the flipped case too
                N_all_[ahf1_flip_loc[i]] = lambda x, n: -N_hf1(x, n)
                N_all_[ahf2_flip_loc[i]] = lambda x, n: -N_hf2(x, n)
        N_all.append(N_all_)

    # Factory Function: return function calculate factored axial force on sections due to specified load combination
    def get_axial_function(sub_axial_matrix, alum=True, on_section=True):

        f_loc = list(zip(*np.where(sub_axial_matrix)))

        if alum:  # section is alum
            LCFA = [fd_a, fd_a, fw_a, fw_a]  # adverse dl
            LCFA_ = [fd_ab, fd_ab, fw_a, fw_a]  # beneficial dl
        else:  # section is steel
            LCFA = [fd_s, fd_s, fw_s, fw_s]  # adverse dl
            LCFA_ = [fd_sb, fd_sb, fw_s, fw_s]  # beneficial dl

        def Ns_lc(x, n):  # function for factored axial on section at position x of element n, assuming dl is adverse
            Ns_xn = np.zeros(sub_axial_matrix.shape)
            for j, k in f_loc:
                Ns_xn[j, k] = sub_axial_matrix[j, k](x, n)
            return Ns_xn @ LCFA

        def Ns_lc_(x, n):  # function for factored calculating axial on section at position x of element n, assuming dl is beneficial
            Ns_xn = np.zeros(sub_axial_matrix.shape)
            for j, k in f_loc:
                Ns_xn[j, k] = sub_axial_matrix[j, k](x, n)
            return Ns_xn @ LCFA_

        def N_lc(x, n):  # function for non-factored axial member force at position x of element n
            N_xn = np.zeros(sub_axial_matrix.shape)
            for j, k in f_loc:
                N_xn[j, k] = sub_axial_matrix[j, k](x, n)
            return N_xn

        if on_section:
            return Ns_lc, Ns_lc_
        else:
            return N_lc

    # list of functions for calculating axial force due to various load combinations
    Ns_LC_a = []
    Ns_LC_s = []
    N_LC = []
    for i in range(spans):
        Ns_LC_a.append([get_axial_function(np.delete(N_all[i], np.where(CFA[:, lcn] == 0), axis=1), alum=True)
                        for lcn in range(CFA.shape[1])])
        Ns_LC_s.append([get_axial_function(np.delete(N_all[i], np.where(CFA[:, lcn] == 0), axis=1), alum=False)
                        for lcn in range(CFA.shape[1])])
        N_LC.append([get_axial_function(np.delete(N_all[i], np.where(CFA[:, lcn] == 0), axis=1), on_section=False)
                     for lcn in range(CFA.shape[1])])

    # Total matrix of all functions about deflection due to load combinations
    d_all = []
    for i in range(spans):
        d_all_ = np.empty(shape=(group_len[i], 10), dtype=object)
        d_all_[wp1_loc[i]] = d_wp1
        d_all_[wp2_loc[i]] = d_wp2
        d_all_[ws1_loc[i]] = d_ws1
        d_all_[ws2_loc[i]] = d_ws2
        if load_vf:  # load from vertical feature
            d_all_[wlvf_loc[i]] = d_vf
            if wlvf_flip:  # consider the flipped case too
                d_all_[wlvf_flip_loc[i]] = lambda x, n: -d_vf(x, n)
        if load_hf:  # load from horizontal feature
            d_all_[wlhf1_loc[i]] = d_hf1
            d_all_[wlhf2_loc[i]] = d_hf2
            if wlhf_flip:  # consider the flipped case too
                d_all_[wlhf1_flip_loc[i]] = lambda x, n: -d_hf1(x, n)
                d_all_[wlhf2_flip_loc[i]] = lambda x, n: -d_hf2(x, n)
        d_all.append(d_all_)

    # Factory Function: return functions calculate deflection of sections due to specified load combination
    def get_deflection_function(span_no, sub_deflection_matrix, on_section=True):

        f_loc = list(zip(*np.where(sub_deflection_matrix)))

        def ds_lc(x, n):  # function for calculating deflection of sections at position x of element n
            ds_xn = np.zeros(sub_deflection_matrix.shape)
            for j, k in f_loc:
                ds_xn[j, k] = sub_deflection_matrix[j, k](x, n)
            return (es[span_no] @ eta_v[span_no] @ ds_xn).sum(axis=1)

        def d_lc(x, n):  # function for calculating overall deflection of mullion member at position x of element n
            d_xn = np.zeros(sub_deflection_matrix.shape)
            for j, k in f_loc:
                d_xn[j, k] = sub_deflection_matrix[j, k](x, n)
            return d_xn

        if on_section:
            return ds_lc
        else:
            return d_lc

    # list of functions for calculating deflection due to various load combinations
    ds_LC = []
    d_LC = []
    for i in range(spans):
        ds_LC.append([get_deflection_function(i, np.delete(d_all[i], np.where(CF[:, lcn] == 0), axis=1)) for lcn in
                      range(CF.shape[1])])
        d_LC.append([get_deflection_function(i, np.delete(d_all[i], np.where(CF[:, lcn] == 0), axis=1), on_section=False)
                     for lcn in range(CF.shape[1])])

    # endregion

    # Find Max Total Fiber Stress on Each Section Due to Load Combinations
    # initialize the list of max stress
    stress = {sec_name: [[0, 0] for cnt in range(CF.shape[1])] for sec_name in total_secs}
    # initialize the list of critical position on members
    critical_M = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
    # initialize the list of moment matrix for max negative stress
    Mn = {sec_name: [None] * CF.shape[1] for sec_name in total_secs}
    # initialize the list of moment matrix for max positive stress
    Mp = {sec_name: [None] * CF.shape[1] for sec_name in total_secs}
    # initialize the list of moment on section causing ma negative stress
    moment_n = {sec_name: [None] * CF.shape[1] for sec_name in total_secs}
    # initialize the list of moment on section causing ma positive stress
    moment_p = {sec_name: [None] * CF.shape[1] for sec_name in total_secs}

    for sp in range(spans):
        for i in range(ns[sp]):  # for different section
            area = sec_lib[secs[sp][i]]['A']  # section area
            LF_sec = LF[secs[sp][i]]
            for c in range(CF.shape[1]):  # for different load combinations
                fs_axial = Ns_LC_s[sp][c] if section_mat[secs[sp][i]][0] == 'S' else Ns_LC_a[sp][c]  # function for axial stress
                for bn in [v for v in range(len(beams)) if mu[v] == sp]:  # for different beam element
                    s1 = minimize(
                        lambda x: biabend(sec_lib[secs[sp][i]], Ms_LC[sp][c](x, bn)[i * 2 + 1], Ms_LC[sp][c](x, bn)[i * 2])[0] * LF_sec
                                  + fs_axial[0](x, bn)[i] / area, 0.5, bounds=((0, 1),))
                    s1_ = minimize(
                        lambda x: biabend(sec_lib[secs[sp][i]], Ms_LC[sp][c](x, bn)[i * 2 + 1], Ms_LC[sp][c](x, bn)[i * 2])[0] * LF_sec
                                  + fs_axial[1](x, bn)[i] / area, 0.5, bounds=((0, 1),))
                    s2 = minimize(
                        lambda x: -(biabend(sec_lib[secs[sp][i]], Ms_LC[sp][c](x, bn)[i * 2 + 1], Ms_LC[sp][c](x, bn)[i * 2])[1] * LF_sec
                                    + fs_axial[0](x, bn)[i] / area), 0.5, bounds=((0, 1),))
                    s2_ = minimize(
                        lambda x: -(biabend(sec_lib[secs[sp][i]], Ms_LC[sp][c](x, bn)[i * 2 + 1], Ms_LC[sp][c](x, bn)[i * 2])[1] * LF_sec
                                    + fs_axial[1](x, bn)[i] / area), 0.5, bounds=((0, 1),))
                    resx = (s1.x[0] if s1.fun < s1_.fun else s1_.x[0],
                            s2.x[0] if s2.fun < s2_.fun else s2_.x[0])
                    extreme = [min(s1.fun, s1_.fun), max(-s2.fun, -s2_.fun)]
                    # record the corresponding moments and location
                    if extreme[0] < stress[secs[sp][i]][c][0]:  # record the max. negative stress
                        stress[secs[sp][i]][c][0] = extreme[0]  # -stress
                        critical_M[secs[sp][i]][c][0] = (resx[0], bn)  # (x, n)
                        Mn[secs[sp][i]][c] = M_LC[sp][c](resx[0], bn)  # matrix
                        moment_n[secs[sp][i]][c] = (Ms_LC[sp][c](resx[0], bn)[i * 2 + 1] * LF_sec,
                                                    Ms_LC[sp][c](resx[0], bn)[i * 2] * LF_sec)  # (Mx, My)
                    if extreme[1] > stress[secs[sp][i]][c][1]:  # record the max. positive stress
                        stress[secs[sp][i]][c][1] = extreme[1]  # +stress
                        critical_M[secs[sp][i]][c][1] = (resx[1], bn)  # (x, n)
                        Mp[secs[sp][i]][c] = M_LC[sp][c](resx[1], bn)  # matrix
                        moment_p[secs[sp][i]][c] = (Ms_LC[sp][c](resx[1], bn)[i * 2 + 1] * LF_sec,
                                                    Ms_LC[sp][c](resx[1], bn)[i * 2] * LF_sec)  # (Mx, My)

    # Find Max Deflection on Each Section Due to Load Combinations
    # initialize the list of max stress
    deflection = {sec_name: [[0, 0] for cnt in range(CF.shape[1])] for sec_name in total_secs}
    # initialize the list of critical position on members
    critical_disp = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
    # initialize the list of unit deflection matrix of critical case
    D = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
    for sp in range(spans):
        for i in range(ns[sp]):  # for different section
            for c in range(CF.shape[1]):  # for different load combinations
                for bn in [v for v in range(len(beams)) if mu[v] == sp]:  # for different beam element
                    x_dx = minimize(lambda x: -abs(ds_LC[sp][c](x, bn)[i]), 0.5, bounds=((0, 1),)).x[0]
                    x_dy = minimize(lambda x: -abs(ds_LC[sp][c](x, bn)[i + ns[sp]]), 0.5, bounds=((0, 1),)).x[0]
                    extreme = [ds_LC[sp][c](x_dx, bn)[i], ds_LC[sp][c](x_dy, bn)[i + ns[sp]]]
                    if abs(extreme[0]) > abs(deflection[secs[sp][i]][c][0]):  # record the max. deflection in x
                        deflection[secs[sp][i]][c][0] = extreme[0]  # def_x
                        critical_disp[secs[sp][i]][c][0] = (x_dx, bn)  # (x, n)
                        D[secs[sp][i]][c][0] = d_LC[sp][c](x_dx, bn)  # matrix
                    if abs(extreme[1]) > abs(deflection[secs[sp][i]][c][1]):  # record the max. deflection in y
                        deflection[secs[sp][i]][c][1] = extreme[1]  # def_y
                        critical_disp[secs[sp][i]][c][1] = (x_dy, bn)  # (x, n)
                        D[secs[sp][i]][c][1] = d_LC[sp][c](x_dy, bn)  # matrix

    # Output
    if summary:  # return summary
        max_stress = [0, 0]  # [max. stress on alum member, max stress on steel member]
        max_deflection = [0, 0]  # [max. deflection in x, max. deflection in y]
        for s in total_secs:
            stress_list = [abs(x) for case in stress[s] for x in case]
            deflection_list = list(zip(*[np.abs(x) for x in deflection[s]]))
            max_s = max(stress_list)
            max_dx = max(deflection_list[0])
            max_dy = max(deflection_list[1])
            index = 1 if section_mat[s][0] == 'S' else 0
            if max_stress[index] < max_s:
                max_stress[index] = max_s
            if max_dx > max_deflection[0]:
                max_deflection[0] = max_dx
            if max_dy > max_deflection[1]:
                max_deflection[1] = max_dy
        return Mullion_summary(max_stress, max_deflection)

    else:  # return detail results

        # Find Max Shear on Each Section Due to Load Combinations
        # initialize the list of max shear
        shear = {sec_name: [[0, 0] for cnt in range(CF.shape[1])] for sec_name in total_secs}
        # initialize the list of critical position on members
        critical_V = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
        # initialize the list of shear matrix of critical case
        V = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
        for sp in range(spans):
            for i in range(ns[sp]):  # for different section
                LF_sec = LF[secs[sp][i]]
                for c in range(CF.shape[1]):  # for different load combinations
                    for bn in [v for v in range(len(beams)) if mu[v] == sp]:  # for different beam element
                        x_Vx = minimize(lambda x: -abs(Vs_LC[sp][c](x, bn)[i * 2]), 0.5, bounds=((0, 1),)).x[0]
                        x_Vy = minimize(lambda x: -abs(Vs_LC[sp][c](x, bn)[i * 2 + 1]), 0.5, bounds=((0, 1),)).x[0]
                        extreme = [Vs_LC[sp][c](x_Vx, bn)[i * 2] * LF_sec, Vs_LC[sp][c](x_Vy, bn)[i * 2 + 1] * LF_sec]
                        if abs(extreme[0]) > abs(shear[secs[sp][i]][c][0]):  # record the max. shear in x
                            shear[secs[sp][i]][c][0] = extreme[0]  # Vx
                            critical_V[secs[sp][i]][c][0] = (x_Vx, bn)  # (x, n)
                            V[secs[sp][i]][c][0] = V_LC[sp][c](x_Vx, bn)  # matrix
                        if abs(extreme[1]) > abs(shear[secs[sp][i]][c][1]):  # record the max. shear in y
                            shear[secs[sp][i]][c][1] = extreme[1]  # Vy
                            critical_V[secs[sp][i]][c][1] = (x_Vy, bn)  # (x, n)
                            V[secs[sp][i]][c][1] = V_LC[sp][c](x_Vy, bn)  # matrix

        # Find Max Axial Force on Each section Due to Load Combinations
        # Note: this max axial force may not same as the axial force combining with bending at critical section
        # initialize the list of (max compression, max tension)
        axial = {sec_name: [[0, 0] for cnt in range(CF.shape[1])] for sec_name in total_secs}
        # initialize the list of critical position on members
        critical_N = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
        # initialize the list of shear matrix of critical case
        N = {sec_name: [[None, None] for cnt in range(CF.shape[1])] for sec_name in total_secs}
        for sp in range(spans):
            for i in range(ns[sp]):  # for different section
                # axial[secs[i]] = []  # initialize the list of axial force
                # critical_N[secs[i]] = []  # initialize the list of critical position on members
                # N[secs[i]] = []  # initialize the list of axial force matrix for critical cases
                LF_sec = LF[secs[sp][i]]
                for c in range(CF.shape[1]):  # for different load combinations
                    fs_axial = Ns_LC_s[sp][c] if section_mat[secs[sp][i]][0] == 'S' else Ns_LC_a[sp][c]
                    for bn in [v for v in range(len(beams)) if mu[v] == sp]:  # for different beam element
                        cpn = minimize(lambda x: fs_axial[0](x, bn)[i], 0.5, bounds=((0, 1),))
                        cpn_ = minimize(lambda x: fs_axial[1](x, bn)[i], 0.5, bounds=((0, 1),))
                        ten = minimize(lambda x: -fs_axial[0](x, bn)[i], 0.5, bounds=((0, 1),))
                        ten_ = minimize(lambda x: -fs_axial[1](x, bn)[i], 0.5, bounds=((0, 1),))

                        x = (cpn.x[0] if cpn.fun < cpn_.fun else cpn_.x[0],
                             ten.x[0] if ten.fun < ten_.fun else ten_.x[0])
                        extreme = [min(cpn.fun, cpn_.fun), max(-ten.fun, -ten_.fun)]
                        # record the corresponding moments and location
                        if extreme[0] < axial[secs[sp][i]][c][0]:  # record the max. compression
                            axial[secs[sp][i]][c][0] = extreme[0]  # Ft
                            critical_N[secs[sp][i]][c][0] = (x[0], bn)  # (x, n)
                            N[secs[sp][i]][c][0] = N_LC[sp][c](x[0], bn)  # matrix
                        if extreme[1] > axial[secs[sp][i]][c][1]:  # record the max. tension
                            axial[secs[sp][i]][c][1] = extreme[1]  # Fc
                            critical_N[secs[sp][i]][c][1] = (x[1], bn)  # (x, n)
                            N[secs[sp][i]][c][1] = N_LC[sp][c](x[1], bn)  # matrix

        # Output items: N, V, Mn, Mp, D, axial, shear, moment_n, moment_p, stress, deflection
        return (Mullion_model(N, V, Mn, Mp, D),
                Mullion_verify(axial, shear, moment_n, moment_p, stress, deflection),
                Mullion_critical(critical_N, critical_V, critical_M, critical_disp))


def build_frame(drawing=None, sec_mod=None, sec_area=None, sec_inert=None, auto_assign=False, geoacc=4,
                file_name=None):
    """Interactively build 2D frame model through AutoCAD.

    :param drawing: str. file name (and path) with extension of .dwg to open and operate on. Activate the specified
                    file if it is opened already. Otherwise, try to open the file from specified path.
    :param sec_mod: list of float, modulus of elasticity of each beam. unit = N/mm :superscript:`2`.
                    If not given, use unit value (1) as default.
    :param sec_area: list of float, section area of each beam, unit = mm :superscript:`2`.
                     If not given, use unit value (1) as default.
    :param sec_inert: list of float, moment of inertia of each beam, unit = mm :superscript:`4`.
                      If not given, use unit value (1) as default.
    :param auto_assign: bool. Automatically assign the provide modulus of elasticity, section area and moment of
                        inertia to beams in order. If the size of provided list is not enough, the last value in list
                        will be repeated. If False, interactive keyborad inputting will be requested.
    :param geoacc: int. number of decimal place to be kept when getting the nodes' coordinates.
    :param file_name:  str, file name for exporting model information, with extension of '.json'. If not specified,
                       exporting procedure will be skipped.
    :return: tuple. (``pyfacade.pyeng.Beam2`` object, ``pyfacade.pyeng.Bsolver`` object)
    """

    fr = CADFrame(drawing, geoacc)  # instance of 2D frame
    nodes = fr.nodes  # coordinate of nodes
    beams = fr.beams  # beam sets
    nb = len(beams)  # total number of beams
    restr = fr.set_restrain()  # Define restrain

    while fr.isbusy():
        time.sleep(0.2)
    try:
        brels = fr.set_release()  # Define end release
    except (KeyboardInterrupt, ValueError):
        print("Skip setting end release of model")
        brels = {}

    while fr.isbusy():
        time.sleep(0.2)
    try:  # apply udl on beams
        Q = fr.set_udl()
    except (KeyboardInterrupt, ValueError):
        print("Skip applying udl on model")
        Q = {}

    while fr.isbusy():
        time.sleep(0.2)
    try:  # apply point load on beams
        P = fr.set_pointload()
    except (KeyboardInterrupt, ValueError):
        print("Skip applying point load on model")
        P = {}

    if auto_assign:

        if not sec_mod:
            Es = [1] * nb  # use unit vale
        else:
            if len(sec_mod) < nb:
                Es = sec_mod + [sec_mod[-1]]*(nb-len(sec_mod))  # extend list to fit number of beams
            else:
                Es = sec_mod[:nb]  # get slice to fit number of beams

        if not sec_area:
            As = [1] * nb  # use unit vale
        else:
            if len(sec_area) < nb:
                As = sec_area + [sec_area[-1]]*(nb-len(sec_area))  # extend list to fit number of beams
            else:
                As = sec_area[:nb]  # get slice to fit number of beams

        if not sec_inert:
            Is = [1] * nb  # use unit vale
        else:
            if len(sec_inert) < nb:
                Is = sec_inert + [sec_inert[-1]]*(nb-len(sec_inert))  # extend list to fit number of beams
            else:
                Is = sec_inert[:nb]   # get slice to fit number of beams

    else:  # assign properties interactively
        while fr.isbusy():
            time.sleep(0.2)
        try:
            Es = fr.set_E({x: y for x, y in enumerate(sec_mod)})
        except KeyboardInterrupt:
            print("Skip assigning modulus of elasticity. Use unit value as default.")
            Es = [1] * nb

        while fr.isbusy():
            time.sleep(0.2)
        try:
            As = fr.set_A({x: y for x, y in enumerate(sec_area)})
        except KeyboardInterrupt:
            print("Skip assigning section area. Use unit value as default.")
            As = [1] * nb

        while fr.isbusy():
            time.sleep(0.2)
        try:
            Is = fr.set_I({x: y for x, y in enumerate(sec_inert)})
        except KeyboardInterrupt:
            print("Skip assigning moment of inertia. Use unit value as default.")
            Is = [1] * nb

    bs = [Beam2(*beams[x], As[x], Is[x], Es[x], nodes) for x in range(nb)]  # list of beam object
    model = Bsolver(bs, nodes, restr, brels, Q, P)  # setup solver instance
    model.solve()  # run solver

    if file_name:  # save as json file
        rec = {'nodes': nodes, 'beams': beams, 'A': As, 'I': Is, 'E': Es, 'restrain': restr, 'release': brels,
               'udl': Q, 'pl': P}
        with open(file_name, 'w') as f:
            json.dump(rec, f)
        print(f"Model information have been exported to <{file_name}>")

    return bs, model


def load_frame(file_name, build=True):
    """Load 2D frame model from file saved by ``build_frame`` .

    :param file_name: str, file name with extension of '.json' to be loaded.
    :param build: bool, build 2D frame based on loaded model information.
    :return: If *build* = True, return a tuple (``pyfacade.pyeng.Beam2`` object, ``pyfacade.pyeng.Bsolver`` object).
            Otherwise, return a namedtuple ``Frame_model``.
    """
    with open(file_name) as f:
        frame = json.load(f)

    if build:
        bs = [Beam2(*frame['beams'][x], frame['A'][x], frame['I'][x], frame['E'][x], frame['nodes'])
              for x in range(len(frame['beams']))]
        model = Bsolver(bs, frame['nodes'],
                        {int(k): frame['restrain'][k] for k in frame['restrain'].keys()},
                        {int(k): frame['release'][k] for k in frame['release'].keys()},
                        {int(k): frame['udl'][k] for k in frame['udl'].keys()},
                        {int(k): frame['P'][k] for k in frame['P'].keys()})
        model.solve()

        return bs, model

    else:
        return Frame_model(frame['nodes'],
                           frame['beams'],
                           frame['A'],
                           frame['I'],
                           frame['E'],
                           {int(k): frame['restrain'][k] for k in frame['restrain'].keys()},
                           {int(k): frame['release'][k] for k in frame['release'].keys()},
                           {int(k): frame['udl'][k] for k in frame['udl'].keys()},
                           {int(k): frame['P'][k] for k in frame['P'].keys()})

# todo: Function: transom of sole section check

# todo: Function: mullion of sole section check

# todo: Function: typical mullion quick builder


if __name__ == '__main__':

    # Sample: check transom
    # section_lib = "C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\section_lib_transoms.json"
    # section_comb = [['Bottom_transom', 'Top_transom', 'x'],
    #                 ['Bottom_transom', 'RF_D', 'xy'],
    #                 ['Top_transom', 'RF_C', 'xy']]  # group_x and group_y is according to this
    # section_mat = {'Top_transom': '6063-T6',
    #                'Bottom_transom': '6063-T6',
    #                'RF_C': 'S275-40',
    #                'RF_D': 'S275-16'}
    # applied = ["Bottom_transom", "Top_transom", "Bottom_transom", "Bottom_transom",
    #            "Bottom_transom", "Top_transom"]
    # #
    # output = check_transoms(section_lib, section_mat, section_comb, span=2000, h1=1600, h2=2500, load_app=applied,
    #                         wl=(2.0, -2.8), dl1=1600, ds1=2000 / 4, dl2=1000, ds2=0, four_side1=True, four_side2=True,
    #                         feature=350, wlf=4.0, wlf_flip=True, imp=-1000, imq=-1.0, summary=False)


    # Sample: Check Mullion of consistent section
    # section_lib = "C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\section_lib_mullion.json"
    # section_comb = [['Male_Mullion', 'Female_Mullion', 'y'],
    #                 ['Male_Mullion', 'RF_C', 'y']]  # group_x and group_y is according to this
    # section_mat = {'Male_Mullion': '6063-T6',
    #                'Female_Mullion': '6063-T6',
    #                'RF_C': 'S275-16'}
    # applied = ["Male_Mullion", "Female_Mullion", "Male_Mullion", "Female_Mullion", "Male_Mullion",
    #            "Male_Mullion", "Female_Mullion"]
    # time_start = time.time()
    # res1, res2 = check_mullions(section_lib, section_mat, section_comb, height=[3500] * 3, b1=1200, b2=1400, load_app=applied,
    #                      wl=(2.0, -2.8), panel_dens1=0.6, panel_dens2=0.6,
    #                      support1=[500] * 3, support2=[400] * 3, support2_lateral=True,
    #                      wlvf=-4.0, wlhf=4.0, bvf=200, bhf=200, hf_gap=50,
    #                      hf_loc=[800] * 3,
    #                      wlvf_flip=True, wlhf_flip=True, summary=False)


    # Sample: Check mullion of varies section, case 1
    # section_lib = "C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\section_lib_mullion.json"
    # section_comb1 = [['Male_Mullion', 'Female_Mullion', 'y'],
    #                  ['Female_Mullion', 'RF_C', 'x']]
    # section_comb2 = [['Male_Mullion', 'Female_Mullion', 'y'],
    #                  ['Male_Mullion', 'RF_C', 'y']]
    # section_mat = {'Male_Mullion': '6063-T6',
    #                'Female_Mullion': '6063-T6',
    #                'RF_C': 'S275-16'}
    # applied = ["Male_Mullion", "Female_Mullion", "Male_Mullion", "Female_Mullion", "Female_Mullion",
    #            "Male_Mullion", "Female_Mullion"]  # [for_wl1, for_wl2, for_dl1, for_dl2, for_wlvf, for_wlhf1, for_wlhf2]
    #
    # res1, res2, res3 = check_mullions_varsec(section_lib, section_mat, [section_comb1, section_comb2, section_comb1],
    #                                   height=[3500] * 3, b1=1200, b2=1400, load_app=applied,
    #                                   wl=(2.0, -2.8), panel_dens1=0.6, panel_dens2=0.6,
    #                                   support1=[500] * 3, support2=[400] * 3, support2_lateral=True,
    #                                   wlvf=-4.0, wlhf=4.0, bvf=200, bhf=200, hf_gap=50,
    #                                   hf_loc=[800] * 3,
    #                                   wlvf_flip=True, wlhf_flip=True, summary=False)

    # Sample: Check mullion of varies section, case 2
    # section_lib = "C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\section_lib_mullion.json"
    # section_comb1 = [['Male_Mullion', 'Female_Mullion', 'y']]
    # section_comb2 = [['Male_Mullion', 'Female_Mullion', 'y'],
    #                  ['Male_Mullion', 'RF_C', 'y']]
    # section_mat = {'Male_Mullion': '6063-T6',
    #                'Female_Mullion': '6063-T6',
    #                'RF_C': 'S275-16'}
    # applied = ["Male_Mullion", "Female_Mullion", "Male_Mullion", "Female_Mullion", "Female_Mullion",
    #            "Male_Mullion", "Female_Mullion"]  # [for_wl1, for_wl2, for_dl1, for_dl2, for_wlvf, for_wlhf1, for_wlhf2]
    #
    # res1, res2, res3 = check_mullions_varsec(section_lib, section_mat, [section_comb1, section_comb2, section_comb1],
    #                                   height=[3500] * 3, b1=1200, b2=1400, load_app=applied,
    #                                   wl=(2.0, -2.8), panel_dens1=0.6, panel_dens2=0.6,
    #                                   support1=[500] * 3, support2=[400] * 3, support2_lateral=False,
    #                                   wlvf=-4.0, wlhf=4.0, bvf=200, bhf=200, hf_gap=50,
    #                                   hf_loc=[800] * 3,
    #                                   wlvf_flip=True, wlhf_flip=True, summary=False)

    # Sample: build 2D frame model
    b, md = build_frame(sec_mod=[205000], sec_area=[1000, 1000], sec_inert=[200000], auto_assign=False,
                        file_name="C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\model.json")
    md.show_diag('Moment', DivN=20, Scale=2000, Accr=4)

    # Sample: read model and build up automatically
    bb, m = load_frame("C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\model.json")
    m.show_diag('Moment', DivN=20, Scale=2000, Accr=4)

    # Sample: read model information and build manually
    model = load_frame("C:\\Work File\\Python Code\\PycharmProjects\\pyfacade\\working_file\\model.json", build=False)
    bb = [Beam2(*model.beams[x], model.A[x], model.I[x], model.E[x], model.nodes) for x in range(len(model.beams))]
    m = Bsolver(bb, model.nodes, model.restrain, model.release, model.udl, model.pl)
    m.solve()
    m.show_diag('Moment', DivN=20, Scale=2000, Accr=4)


