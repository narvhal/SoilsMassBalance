
import warnings

import requests
import json
# import xarray as xr
import numpy as np
import pandas as pd 
import glob
import os
import pprint as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib as mpl
# import geopy.distance
# from geopy.distance import geodesic
import csv
# import scipy as scipy
import statsmodels.api as sm
from scipy.stats import linregress
from scipy.optimize import curve_fit
import plotly.express as px
import seaborn as sns

from mendeleev import element
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import * 
import uncertainties.unumpy as unumpy
from uncertainties import  ufloat_fromstr
import copy


#  plt.rcParams["font.family"] = "cursive"
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# maybe print pf to check how reasonable it is.





def generic_squares(list_of_partials, list_of_uncerts):
    temp_sum = 0
    for t, par in enumerate(list_of_partials):
        temp_sum += (par**2 )* (list_of_uncerts[t])**2
    return np.sqrt(temp_sum)

def f_rest(Inv, D, ltl):
    # Residence time:
    # ltl is lambda (1/yr)
    # D is at /cm2/yr
    #     rest = []
    #     for i,iinnvv in enumerate(Inv):
    #     for i in range(len(Inv)):
    #         rest.append((-1.0/ ltl) * log(1 - (ltl* Inv[i] / D)))   #t == yr
    #     return rest
    try:
        return (-1.0/ ltl) * log(1 - (ltl* Inv/ D))
    except:
        return ufloat(-1, .1)

def f_Inv(N,p_re,z):
    # Inventory = Integral(zb-z)(N*density)dz

    # N is conc Be in at/cm2
    # p_re is density of soil
    # z is soil depth (cm)
    # Inv is at/cm2
    Inv = N*p_re*z
    return Inv

def f_Inv_unc(N,p_re,z, N_unc, p_re_unc, z_unc):
    list_of_partials = [z*p_re, N*p_re, z*N]
    list_of_uncerts = [ N_unc, z_unc, p_re_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def f_Inv_unc_pde(df, N,p_re,z, N_unc, p_re_unc, z_unc):
    list_of_partials = [z*p_re, N*p_re, z*N]
    df["Inv_dI/dN"] = list_of_partials[0]
    df["Inv_dI/dz"] = list_of_partials[1]
    df["Inv_dI/dpre"] = list_of_partials[2]
    return df

def f_rest_unc(Inv, D, ltl, Inv_unc, D_unc):
    list_of_partials = [1/(D-ltl*Inv), -Inv/( (D**2 )*(1-ltl*Inv/D))]
    list_of_uncerts = [Inv_unc, D_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def f_rest_unc_pde(df, Inv, D, ltl, Inv_unc, D_unc):
    list_of_partials =  [1/(D-ltl*Inv), -Inv/( (D**2 )*(1-ltl*Inv/D))]
    df["rt_dt/dI"] = list_of_partials[0]
    df["rt_dt/dD"] = list_of_partials[1]
    return df

def f_erate(Inv, D, N, ltl, p_re):
    # Inv in at/cm2
    # N is uppermmost sample 10Be in at/g
    # D in at/cm2 /yr
    # p_re in g/cm3
    # ltl in 1/yr

    erate = (D- ltl *Inv)/ (N*p_re)*10*1000   # returns mm/kyr
    return erate

def f_erate_unc(z, D, N, ltl, p_re, z_unc, D_unc, N_unc, p_re_unc):
    # erate = (D- ltl *Inv)/ (N*p_re)*10*1000   # mm/kyr
    list_of_partials = [-D*1e4/(p_re * N**2), 1e4/(N * p_re), -ltl*1e4, -D *1e4 / (N*p_re**2)]
    list_of_uncerts = [N_unc, D_unc, z_unc, p_re_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def f_erate_unc_pde(df, z, D, N, ltl, p_re, z_unc, D_unc, N_unc, p_re_unc):
    list_of_partials =  [-D*1e4/(p_re * N**2), 1e4/(N * p_re), -ltl*1e4, -D *1e4 / (N*p_re**2)]
    df["erate_de/dN"] = list_of_partials[0]
    df["erate_de/dD"] = list_of_partials[1]
    df["erate_de/dz"] = list_of_partials[2]
    df["erate_de/dpre"] = list_of_partials[3]
    return df

###
# Volumetric Flux -- West, Eqn 26, w my modification
# Cre is conc, mass 10Be/mass sample
# Is (x-x0) dx here? It's what she says. So not distance downslope…
# West 14, Eq 26.
# q = D*m_a*(x-x0)/(Cre*pre)
# at/cm2/yr * g/at *cm/ (g/g * g/cm3)
# g/cm/yr  / g/cm3 = cm2/yr

def f_regflux(D, xpos, x0, p_re, N ):
    # Can be surface or depth avg conc. UNits out: cm2/yr
    # variations: depth-averaged concentration. Cumulative q downslope
    regflux = D* (xpos - x0) / (p_re*N)  # cm2/yr
    return regflux

def f_regflux_unc(D, xpos, x0, p_re, N, D_unc, p_re_unc, N_unc ):
    # qs = D* (xpos - x0) / (p_re*N)
    list_of_partials = [(xpos - x0) / (p_re*N), -D* (xpos - x0)/(N*p_re**2), -D* (xpos - x0)/(p_re* N**2)]
    list_of_uncerts = [D_unc, p_re_unc, N_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def flux_boxmodel( SD, res_t, ps):
    # units out: g/(cm2*yr)
    # soil formation rate (E: cm/yr) * psoil = soil depth * psoil / res time
    # SD in cm , res_t in yr, ps in g/cm3: OUT is  g/(cm2*yr)
    return SD*ps/res_t

def flux_boxmodel_unc( SD, res_t, p_re, SD_unc, res_t_unc, p_re_unc):
    # units out: g/(cm2*yr)
    # soil formation rate (E: cm/yr) * psoil = soil depth * psoil / res time
    list_of_partials = [p_re/res_t, -SD*p_re/(res_t**(2)), SD/res_t]
    list_of_uncerts = [SD_unc, res_t_unc, p_re_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def v_reg(D, xpos,x0, p_re, N_B, N_A):
    # velocity, p 40 notebook for written out units. based on Jungers' desc.
    v=D*(xpos-x0)/(p_re*(N_B - N_A))

def f_mass_balance_for_dust(F_fines, F_coarse, F_br, DF):
    # out in g/cm2/yr
    F_dust = F_fines + (F_coarse - F_br)/ (1+ DF)
    return F_dust

def f_mass_balance_for_dust_unc(F_fines, F_coarse, F_br, DF, F_fines_unc, F_coarse_unc, F_br_unc, DF_unc):
    list_of_partials = [1, 1/(1+DF), -1/(1+DF) , -(F_coarse - F_br)*(1+DF)**(-2)]
    list_of_uncerts = [F_fines_unc, F_coarse_unc, F_br_unc, DF_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def f_coarse_flux(coarse_mass, coarse_area, max_coarse_residence_time):
    # Res time in yr, out: g/cm2/yr
    return coarse_mass/coarse_area / max_coarse_residence_time

def f_coarse_flux_unc(coarse_mass, coarse_area, max_coarse_residence_time, coarse_mass_unc, coarse_area_unc, max_coarse_residence_time_unc):
    # Res time in yr, out: g/cm2/yr
    # coarse_mass/clast_area / res time
    list_of_partials = [1/(coarse_area*max_coarse_residence_time), -coarse_mass/(coarse_area**2) /max_coarse_residence_time, -coarse_mass/(max_coarse_residence_time**2) /coarse_area]
    list_of_uncerts = [coarse_mass_unc, coarse_area_unc, max_coarse_residence_time_unc]
    return generic_squares(list_of_partials, list_of_uncerts)

def f_br_flux(br_E_rate, p_br):
    # in: Erate in mm/kyr, out: g/cm2/yr
    # mm/kyr * g/cm3 * 1 ky/1000 yr * 1 cm/ 10 mm
    return br_E_rate*p_br/10/1000

def f_br_flux_unc(br_E_rate, p_br, br_E_rate_unc):
    # in: Erate in mm/kyr, out: g/cm2/yr
    #
    list_of_partials = [p_br/10/1000]
    list_of_uncerts = [br_E_rate_unc]
    return generic_squares(list_of_partials, list_of_uncerts)



def get_nomval(uf):
    # returns val, stdev
    if isinstance(uf, uncertainties.core.Variable) | isinstance(uf, uncertainties.core.AffineScalarFunc):
        return uf.nominal_value, uf.std_dev
    else:  # not a single number....
        return unumpy.nominal_values(uf), unumpy.std_devs(uf)

def redef_uf(uf):
    # returns val, stdev
    if isinstance(uf, uncertainties.core.Variable) | isinstance(uf, uncertainties.core.AffineScalarFunc):
        return ufloat(uf.nominal_value, uf.std_dev)
    else:  # not a single number....
        ufa = []
        for i, ufv in enumerate(uf):
            ufa.append(ufloat(unumpy.nominal_values(ufv), unumpy.std_devs(ufv)))
        return np.array(ufa) # ufloat(unumpy.nominal_values(uf), unumpy.std_devs(uf))

def get_vals_uf(uf):
    # returns val, stdev
    if isinstance(uf, uncertainties.core.Variable) | isinstance(uf, uncertainties.core.AffineScalarFunc):
        return uf.nominal_value, uf.std_dev
    else:  # not a single number....
        return  unumpy.nominal_values(uf), unumpy.std_devs(uf)



def set_up_df_for_flux_results(df, Ncol, N_unc_col, zcol, z_unc, I_desc, p_re, p_re_unc,p_br,  D_Desc, D, D_unc, DF,DF_unc, br_E_rate, br_E_rate_unc, coarse_mass, coarse_mass_unc, coarse_area,coarse_area_unc, max_coarse_residence_time,max_coarse_residence_time_unc):

    # Flat values for all rows
    dd = {'N_col': Ncol,  'N_unc_col': N_unc_col, 'z_col': zcol, 'z_unc': z_unc, 'Inv_Desc': I_desc, 'p_re': p_re, 'p_re_unc': p_re_unc,'p_br': p_br,  'D_Desc': D_Desc, 'D': D, 'D_unc': D_unc, 'DF': DF, 'DF_unc': DF_unc, 'br_E_rate':br_E_rate, 'br_E_rate_unc': br_E_rate_unc, 'coarse_mass':coarse_mass, 'coarse_mass_unc': coarse_mass_unc, 'coarse_area': coarse_area, 'coarse_area_unc': coarse_area_unc, 'max_coarse_residence_time': max_coarse_residence_time, 'max_coarse_residence_time_unc': max_coarse_residence_time_unc }
    if zcol in ['local_sd_min', 'local_sd_max', 'local_sd_avg']:
        dft = df[['sample_region', 'sample_location', 'sample_position','sample_id','s_dx (cm)','s_dx (cm) upslope','inv_sum_flag','flow acc m2', Ncol,N_unc_col, zcol]].copy()

    else:
        dft = df[['sample_region', 'sample_location', 'sample_position','sample_id','s_dx (cm)','s_dx (cm) upslope','inv_sum_flag','flow acc m2', Ncol,N_unc_col]].copy()

    for i, di in enumerate(dd.keys()):
        dft[di] = dd[di]

    if zcol in ['local_sd_min', 'local_sd_max', 'local_sd_avg']:
        SD = dft[zcol]
    else:
        SD = zcol
    dft['z'] = SD
    dft['z_unc'] = z_unc

    if I_desc == 'site':
        sitecols = []
        dft['z_site'] = dft.groupby('inv_sum_flag')['z'].transform('sum')
    #         dft['Inv_site'] = dft.groupby('inv_sum_flag')['Inv'].transform('sum')
    #         dft['Inv'] = dft['Inv_site']
    #         if zcol in ['local_sd_min', 'local_sd_max', 'local_sd_avg']:
    #             SD = dft.groupby('inv_sum_flag')[zcol].transform('sum')
        dft['N_site'] = dft.groupby('inv_sum_flag')[Ncol].transform('mean')
        dft['N_site_unc'] = dft.groupby('inv_sum_flag')[N_unc_col].transform('mean')
        dft['N'] = dft.apply(lambda x: ufloat(x['N_site'], x['N_site_unc'], "N"), axis = 1)
        N = dft['N']
        dft['N_unc'] = dft['N_site_unc']

        dft['z'] = dft['z_site']

    else:
        dft['N'] = dft.apply(lambda x: ufloat(x[Ncol], x[N_unc_col], "N"), axis = 1)
        N = dft['N']

    zl = []
    for r, ro in enumerate(dft['z']):
        zl.append(ufloat(ro, dft['z_unc'].iloc[r]))
    dft['z'] = zl

    return dft, SD, N

# PDE
def solve_rt(dft, flag_pde = False):
    rtruf = []
    rt_pde_Inv = []
    rt_pde_D = []
    for r, ro in enumerate(dft['Inv']):
        v1 = redef_uf(dft['D'].iloc[r])
        ro = redef_uf(ro)
        rt_i = f_rest(ro, v1, ltl)
        rt_redef =redef_uf(rt_i)
        rtruf.append(rt_redef)
        try:
            drv = rt_i.derivatives[ro]
            rt_pde_Inv.append(drv)
            rt_pde_D.append(rt_i.derivatives[v1])
        except:
        #             print('Exception, solve rt: rt_i = ', rt_i)
            pass
    dft['rt'] = rtruf
    res_t = dft['rt']
    if flag_pde:
        dft['rt_pdeInv'] = rt_pde_Inv
        dft['rt_pdeD'] = rt_pde_D
    return dft, res_t

def solve_E_fines(dft):
    #     dft, E_fines = solve_E_fines(dft)
    Et = []
    Et_pdez = []
    Et_pdeD = []
    Et_pdeN = []
    Et_pdepre = []
    for r, ro in enumerate(dft['z']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['D'].iloc[r])
        v3 = redef_uf(dft['N'].iloc[r])
        v4 = redef_uf(dft['p_re'].iloc[r])
        Et_i = f_erate(v1, v2, v3, ltl, v4)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)
        Et_pdez.append(Et_i.derivatives[v1])
        Et_pdeD.append(Et_i.derivatives[v2])
        Et_pdeN.append(Et_i.derivatives[v3])
        Et_pdepre.append(Et_i.derivatives[v4])

    dft['E_fines'] = Et
    #dft.apply(lambda x: f_erate(x['z'],x['D'], x['N'], ltl, x['p_re']), axis = 1)
    E_fines = dft['E_fines']

    dft['E_fines_pdez'] = Et_pdez
    # dft.apply(lambda x: x['E_fines'].derivatives[x['z']], axis = 1)
    dft['E_fines_pdeD'] = Et_pdeD
    # dft.apply(lambda x: x['E_fines'].derivatives[x['D']], axis = 1)
    dft['E_fines_pdeN'] = Et_pdeN
    # dft.apply(lambda x: x['E_fines'].derivatives[x['N']], axis = 1)
    dft['E_fines_pdepre'] = Et_pdepre
    # dft.apply(lambda x: x['E_fines'].derivatives[x['p_re']], axis = 1)
    return dft, E_fines

def solve_F_fines(dft):
    #     dft['F_fines_boxmodel'] = flux_boxmodel( SD, res_t, p_re)
    #     F_fines = dft['F_fines_boxmodel']
    #     dft['E_fines_pdez'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['z']], axis = 1)
    #     dft['E_fines_pdeD'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['D']], axis = 1)
    #     dft['E_fines_pdeN'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['N']], axis = 1)
    #     dft['E_fines_pdepre'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['p_re']], axis = 1)
    Et = []
    Et_pdez = []
    Et_pdeD = []
    Et_pdeN = []
    Et_pdepre = []
    for r, ro in enumerate(dft['z']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['rt'].iloc[r])  # is this supposed to be in *yrs* or ky?
        v3 = redef_uf(dft['p_re'].iloc[r])
        Et_i = flux_boxmodel(v1, v2, v3)    # SD in cm , res_t in yr, ps in g/cm3: OUT is  g/(cm2*yr)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)
        Et_pdez.append(Et_i.derivatives[v1])
        Et_pdeD.append(Et_i.derivatives[v2])
        Et_pdepre.append(Et_i.derivatives[v3])

    dft['F_fines_boxmodel'] = Et
    F_fines = dft['F_fines_boxmodel']

    dft['F_fines_boxmodel_pdez'] = Et_pdez
    dft['F_fines_boxmodel_pdeD'] = Et_pdeD
    dft['F_fines_boxmodel_pdepre'] = Et_pdepre
    return dft, F_fines

def solve_F_coarse(dft):  # dft, F_coarse  = solve_F_coarse(dft)
    Et = []
    Et_pdecm = []
    Et_pdeca = []
    Et_pdemcrt = []
    for r, ro in enumerate(dft['coarse_mass']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['coarse_area'].iloc[r])
        v3 = redef_uf(dft['max_coarse_residence_time'].iloc[r])
        Et_i = f_coarse_flux(v1, v2, v3)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)
        Et_pdecm.append(Et_i.derivatives[v1])
        Et_pdeca.append(Et_i.derivatives[v2])
        Et_pdemcrt.append(Et_i.derivatives[v3])

    dft['F_coarse'] = copy.copy(Et)
    #     print(type(copy.copy(Et[0])))
    F_coarse = dft['F_coarse']

    dft['F_coarse_boxmodel_pdecm'] = Et_pdecm
    dft['F_coarse_boxmodel_pdeca'] = Et_pdeca
    dft['F_coarse_boxmodel_pdemcrt'] = Et_pdemcrt
    return dft, F_coarse

def solve_F_br(dft):  # dft, F_br  = solve_F_br(dft)
    Et = []
    Et_pdebrE = []
    Et_pdepbr = []

    for r, ro in enumerate(dft['br_E_rate']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['p_br'].iloc[r])
        Et_i = f_br_flux(v1, v2)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)
        Et_pdebrE.append(Et_i.derivatives[v1])
        Et_pdepbr.append(Et_i.derivatives[v2])

    dft['F_br'] = Et
    F_br = dft['F_br']

    dft['F_br_boxmodel_pdebrE'] = Et_pdebrE
    dft['F_br_boxmodel_pdepbr'] = Et_pdepbr
    return dft, F_br

def solve_F_dust(dft):  # dft, F_dust  = solve_F_dust(dft)
    Et = []
    Et_pdeFf = []
    Et_pdeFc = []
    Et_pdeFbr = []
    Et_pdeDF = []

    for r, ro in enumerate(dft['F_fines_boxmodel']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['F_coarse'].iloc[r])
        v3 = redef_uf(dft['F_br'].iloc[r])
        v4 = redef_uf(dft['DF'].iloc[r])
        Et_i = f_mass_balance_for_dust(v1, v2, v3, v4)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)
        Et_pdeFf.append(Et_i.derivatives[v1])
        Et_pdeFc.append(Et_i.derivatives[v2])
        Et_pdeFbr.append(Et_i.derivatives[v3])
        Et_pdeDF.append(Et_i.derivatives[v4])

    dft['F_dust'] = Et
    F_dust = dft['F_dust']

    dft['F_dust_boxmodel_pdeFf'] = Et_pdeFf
    dft['F_dust_boxmodel_pdeFc'] = Et_pdeFc
    dft['F_dust_boxmodel_pdeFbr'] = Et_pdeFbr
    dft['F_dust_boxmodel_pdeDF'] = Et_pdeDF
    return dft, F_dust

def define_unc_cols(dft):
    # Now calculate Uncertainties using the functions I made....
    # Get nominal values so unc funcs don't freak out

    Iu = []
    rtu = []
    efu = []
    fbru = []
    fcu = []
    ff = []
    fd = []

    ium = []
    rum = []
    fbum = []
    fcum = []
    ffbum = []
    fdum = []
    for iu, u in enumerate(dft['N']):
        N, N_unc = get_nomval(dft['N'].iloc[iu])
        p_re, p_re_unc = get_nomval(dft['p_re'].iloc[iu])
        SD, z_unc =  get_nomval(dft['z'].iloc[iu])
        D, D_unc =  get_nomval(dft['D'].iloc[iu])
        DF, DF_unc =  get_nomval(dft['DF'].iloc[iu])
        br_E_rate, br_E_rate_unc =  get_nomval(dft['br_E_rate'].iloc[iu])
        coarse_mass, coarse_mass_unc = get_nomval(dft['coarse_mass'].iloc[iu])
        coarse_area, coarse_area_unc = get_nomval(dft['coarse_area'].iloc[iu])
        max_coarse_residence_time, max_coarse_residence_time_unc = get_nomval(dft['max_coarse_residence_time'].iloc[iu])
        res_t, rtu_up = get_nomval(dft['rt'].iloc[iu])
        F_fines,jj = get_nomval(dft['F_fines_boxmodel'].iloc[iu])
    #         print(F_fines)
        F_coarse,jj = get_nomval(dft['F_coarse'].iloc[iu])
        F_br,jj = get_nomval(dft['F_br'].iloc[iu])
        F_d,jj = get_nomval(dft['F_dust'].iloc[iu])
        Inv, Inv_unc = get_nomval(dft['Inv'].iloc[iu])
    #         print(f"{foo=} {bar=}")
    #         lsss = [N,p_re,SD, N_unc, p_re_unc, z_unc]
    #         print('Inv unc: ', [f"{i}{lsss[i]=}" for i,x in enumerate(lsss) if not (isinstance(x, float) | isinstance(x, int))])
        ium.append(Inv)
        rum.append(res_t)
        fbum.append(F_br)
        fcum.append(F_coarse)
        ffbum.append(F_fines)
        fdum.append(F_d)

        Inv_unc = f_Inv_unc(N,p_re,SD, N_unc, p_re_unc, z_unc)
        Iu.append(Inv_unc)
    #         Invi, hh = get_nomval(Inv)
    #         lsss = [Inv, D, ltl, Inv_unc, D_unc]
    #         print('rt unc: ', [f"{i}{lsss[i]=}" for i,x in enumerate(lsss) if not (isinstance(x, float) | isinstance(x, int))])

        res_t_unc = f_rest_unc(Inv, D, ltl, Inv_unc, D_unc)
        rtu.append(res_t_unc)
        efu.append(f_erate_unc(SD, D, N, ltl, p_re, z_unc, D_unc, N_unc, p_re_unc))
        F_br_unc = f_br_flux_unc(br_E_rate, p_br, br_E_rate_unc)   #   in: Erate in mm/kyr, out: g/cm2/yr
        fbru.append(F_br_unc)
        F_coarse_unc = f_coarse_flux_unc(coarse_mass, coarse_area, max_coarse_residence_time, coarse_mass_unc, coarse_area_unc, max_coarse_residence_time_unc)
        fcu.append(F_coarse_unc)
        F_fines_unc = flux_boxmodel_unc( SD, res_t, p_re, z_unc, res_t_unc, p_re_unc)
        ff.append(F_fines_unc)
    #         print('fd unc: ', [f"{x=}" for x in [F_fines, F_coarse, F_br, DF, F_fines_unc, F_coarse_unc, F_br_unc, DF_unc] if not (isinstance(x, float) | isinstance(x, int))])

        fd.append(f_mass_balance_for_dust_unc(F_fines, F_coarse, F_br, DF, F_fines_unc, F_coarse_unc, F_br_unc, DF_unc))

    # sanity check
    dft['Inv_main'] = ium
    dft['rt_main'] = rum  # array? why?

    dft['F_br_main'] = fbum
    dft['F_coarse_main' ] = fcum
    dft['F_fines_boxmodel_main'] = ffbum
    dft['F_dust_by_mass_balance_main'] =fdum # array? whty

    dft['Inv_unc'] = Iu
    dft['rt_unc'] = rtu  # array? why?
    dft['E_fines_unc'] = efu
    dft['F_br_unc'] = fbru
    dft['F_coarse_unc' ] = fcu
    dft['F_fines_boxmodel_unc'] = ff
    dft['F_dust_by_mass_balance_unc'] =fd # array? whty

    return dft
