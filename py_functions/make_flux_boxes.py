
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import BytesIO
import matplotlib
import pandas as pd

import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unumpy
from uncertainties import  ufloat_fromstr
import copy





def fltr(df, scenario, selcol, selcolval, sample_id = None):
    dft = df[(df.default_scenario == scenario) & (df.select_col == selcol) & (df.select_col_val == selcolval)  ].copy()
    if sample_id != None:
        dft = dft[dft.sample_id.isin(sample_id)].copy()
    return dft
def fltr2(df,  selcol, selcolval, sample_id = None):
    # dft = df[(df.select_col == selcol) & (df.select_col_val == selcolval)  ].copy()
    # if sample_id != None:
    dft = df[df.sample_id.isin(sample_id)].copy()
    return dft

def vcols(listofcols):
    return [c + '_val' for c in listofcols]

def findcols(df, schstr, notstr = None):
    lst = df.columns.to_list()
    if notstr != None:
        for ns in notstr:
            lst = [c for c in lst if not (ns in c)]
    return [c for c in lst if schstr in c]

def wrap_flux_box_visual(df, saveloc , scenario ,selcol,selcolval):
    saveloc2 = saveloc + '\\flux_box_figs_'+ scenario + '_'+selcol+ '_'+ str(selcolval)
    if not os.path.exists(saveloc2):
        os.mkdir(saveloc2)
    for si in ['MT120', 'MT130', 'MT140',  'NQT0', 'NQCC2']:
        dft = fltr(df, scenario, selcol, selcolval, sample_id = [si])
        list_of_tuplelists,ft,fst = make_into_area(dft, flag_model = 'not', height = 3)

        plot_patches(list_of_tuplelists, dft, ft,fst, height = 3)
        filenametag = si +'_boxflux_w_dust'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')
    for si in ['MT120', 'MT130', 'MT140',  'NQT0', 'NQCV2']:
        dft = fltr(df, scenario, selcol, selcolval, sample_id = [si])

        list_of_tuplelists,ft,fst = make_into_area(dft, flag_model = 'simple', height = 3)

        plot_patches(list_of_tuplelists, dft, ft,fst, height = 3)
        filenametag = si +'_boxflux_simple'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')

def proc(key):
    st.info(st.session_state[key])


def mtfmt(mt):   # functions to provide vals for 'model_type'
    mtd = {'simple':"Solve for dissolved flux, no dust ('simple')",
    'wdust':  "Solve for dust flux (dissolved flux constrained by calcite mass balance)" }
    return mtd[mt]


def wrap_flux_box_streamlit(dft, selval_dict):
    fig, ax = plt.subplots()
    height = selval_dict['model_shape']
    flag_model = selval_dict['model_type']
    list_of_tuplelists,ft,fst, height , L, H, XY, fst, YC = make_into_area_streamlit(dft, flag_model = flag_model, height = height, scale = selval_dict["boxscale"] , shape_buffer = selval_dict["shape_buffer"])
    plot_patches(list_of_tuplelists, dft, ft, L, H, XY,YC, fst, height = height,
        flag_model =flag_model, newfig = False,flag_annot = False)
    fig.set_size_inches(selval_dict['figwidth'], selval_dict['figheight'])
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width = selval_dict["pixelwidth"])
    return fig


def deal_w_ustring(val):
    if isinstance(val, str):
        numstr = ufloat_fromstr(val)
    else:
        numstr = val
    return numstr

def make_into_area_streamlit(df, flag_model= 'simple', height = 'auto', scale = 1, shape_buffer = .75):
    if flag_model == 'simple':
        fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,
            'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
        ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
        spacerloc = 0
    else:
        fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
             'F_coarse_g_m2_yr' ,
             'F_fines_from_br_g_m2_yr' ,
             'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
        ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']
        spacerloc = 1


    H = []  # Vertical
    L = []  # Horiz
    csum = shape_buffer
    fst = []
    Fbr_L = df[fmcols[0]].to_numpy()[0]   # BR value
    XC = [csum]


    for i, col in enumerate(fmcols[0:]):
        colval = df[col].to_numpy()[0]
        colval = deal_w_ustring(colval)
        if colval > 0:
            if height == 'Uniform height':  # About half of sqrt (FBr_L)
                htt = 0.3 * (Fbr_L)**(0.5)  # bedrock length where Fb_L**2 = Fb
                L1 = colval/htt*scale
            elif isinstance(height, float):
                htt = height
                L1 = colval/htt*scale
            else:  # squares
                L1 = (colval*scale)**(0.5)
                htt = L1
        else:
            L1 = 0

        if i == spacerloc:
            csum = csum + shape_buffer
        # st.write("Make area L1xH: {:.1f}x {:.1f} = {:.1f}".format(float(L1), float(htt), float(L1)*float(htt)))
        # st.write(" {:s}   Orig Area: {:.1f}".format(str(np.round(colval, 1) ==np.round(float(L1)*float(htt), 1) ), colval))
        L.append(L1)
        fst.append(colval)
        H.append(htt)
        csum = L1+csum +shape_buffer
        XC.append( csum)
    # Now need to make each corners
    list_of_tuplelists= []
    for i, (x,y) in enumerate(zip(L,H)):
        # y0 is 0.1 basically
        YC = []
        flag_along_baseline = False
        if flag_along_baseline:
            x0 = XC[i]
            x1 = x0 + x
            y0 = 0.1
            y1 = y0+y
        else: # along centerline
            midy = np.max(H)/2
            x0 = XC[i]
            x1 =  x0 + x
            y0 = midy-(y/2)
            y1 = midy+(y/2)
        YC.append((y0, y1))
        DL = (x0,y0)
        UL = (x0, y1)
        DR = (x1, y0)
        UR = (x1, y1)
        # st.write("Make area from points: {:.1f}x {:.1f} = {:.1f}".format(float(x1-x0), float(y1-y0), float((x1-x0) *(y1-y0))))
        # st.write("Make area from x y : {:.1f}x {:.1f} = {:.1f}".format(float(x1-x0), float(y1-y0),float(x*y)))
        # colval = df[fmcols[i]].to_numpy()[0]
        # st.write(" {:s}   Orig Area: {:.1f}".format(str(np.round(colval, 1) ==np.round(x*y, 1) ), colval))

        list_of_tuplelists.append([DL] + [UL] + [UR]+[DR] +[DL])
    return list_of_tuplelists, ft, fst, height, L, H, XC, fst, YC


def plot_patches(list_of_tuplelist, df, ft, L, H, XC, YC, fst,add_conc = 'auto',  height = 'auto', flag_model = 'simple',newfig = True, flag_annot = True, set_maxy = None, xoffset = 0):
    if newfig:
        fig, ax = plt.subplots()
    else:
        ax = plt.gca()
        fig = plt.gcf()
    equals_locx = 0

    if height != 'Uniform height':
        maxy = H[0]   # bedrock height
        maxx = XC[-1]
    elif isinstance(height, float):
        maxy = height
        maxx = XC[-1]
    else:# squares
        maxy = np.max(H)
        maxx = XC[-1]
    # st.write("XY",XY)
    mxo = maxx
    if flag_model == 'simple':
        hch = ['x', '', '', '']
        bxc = ['grey', 'rosybrown', 'indianred', 'lightcyan']
    else:
        hch = ['x','', '', '', '', '']
        bxc = ['grey','burlywood', 'rosybrown', 'indianred', 'lightcyan', 'burlywood']
    flag_tilt_label = False

    ## Make subscripts smaller??
    matplotlib._mathtext.SHRINK_FACTOR = 0.5

    mxx = []
    for i, points in enumerate(list_of_tuplelist):
        flag_along_baseline = False
        if flag_along_baseline: pass
        else:
            midy = np.max(H)/2
            adjx = [points[p][0] + xoffset for p in np.arange(len(points))]
            y = [points[p][1] for p in np.arange(len(points))]
            npp = list(zip(adjx, y))
            ax.add_patch(mpatches.Polygon(npp, ec = 'dimgrey', fc = bxc[i], hatch = hch[i], ls = '-', lw = .5))  # df.cdict.iloc[0]
            # npn = ( (npp[0][1]+npp[1][1])/2 ,  (npp[1][0] + npp[0][0])/2 )  # Find x and y-midpoint
            npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2, midy )  # Find x and y-midpoint
            # st.write(npp)
            # st.write(f"Area {ft[i]}: {(npp[1][1]-npp[1][0])* (npp[2][0]-npp[1][0])}")
            # st.write(f"Orig: {fst[i]}" )
            # st.write("npp Points:",npp[1])
            if flag_model == 'simple':
                pct_denom = fst[0]  # just bedrock flux
            else:
                pct_denom = fst[0] + fst[1] # bedrock + dust flux
            fst_as_pct = np.round(fst[i]/pct_denom*100, 0)
            if flag_annot:
                if (points[3][0] - points[0][0])<=maxx/20:
                    # st.write("narrow box, "+ft[i]+'   : {:0.1f}'.format(fst[i]))
                    plt.annotate(' '+ft[i]+' : {:0.1f}'.format(fst[i]), npp[1], rotation = 45, fontsize = 15)
                    # if (points[3][0] - points[0][0])>=.6:
                        # plt.annotate(''+ft[i], npn, va = 'center')
                    flag_tilt_label = True
                else: # LABEL boxes in middle
                    # st.write("wide box, " +ft[i])
                    # st.write(npn)
                    # st.write(npp[0])
                    # st.write(points[0])
                    # st.write()

                    plt.annotate(ft[i], npn, va = 'center', fontsize = 13, ha = 'center')
                    plt.annotate('\n{:0.1f}'.format(fst[i]), npn, va = 'top', ha = 'center')
                    plt.annotate('\n\n {:0.0f}%'.format(fst_as_pct), npn, va = 'top', ha = 'center')
            else:
                # npn = (npn[0], npn[1]+ midy*1.9)
                npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  npn[1]+ midy*1.9 ) # Find x and y-midpoint

                # npn = (npp[0][0] , npn[1]+ midy*1.9)
                plt.annotate(ft[i], npn, va = 'center', fontsize = 15, ha = 'center')
                if i == 0:
                    plt.annotate('\n {:0.1f}\n g/m$^2$/yr'.format(fst[i]), npn, fontsize = 10,  va = 'top', ha = 'center')
                else:
                    plt.annotate('\n {:0.1f}'.format(fst[i]), npn,fontsize = 10, va = 'top', ha = 'center')

                npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  (npp[0][1]+npp[1][1])/2 ) # Find x and y-midpoint
                plt.annotate('{:0.0f}%'.format(fst_as_pct), npn, va = 'center', ha = 'center', fontsize = 8 ) # , fontweight = "bold")
            # plt.annotate(f"LxH = Area\n{L[i]} x {H[i]} \n\t= {fst[i]}", (points[0][0], 0.1), va = "center", rotation = 20)
            # Add equation stuff to nearby box
            if i>0:
                spacex = (npp[0][0] - (list_of_tuplelist[i-1][3][0] + xoffset))/2
                if flag_model == 'simple':
    #                 print(i, points)
                    syms = [' ', '=', '+', '+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 ),ha='center', va = 'center')

                else:
                    syms = [' ','+', '=', '+', '+','+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 ),ha='center', va = 'center')
                # Also write between F labels
                if not flag_annot:
                    npn2 = (npp[1][0]-spacex,  npn[1]+ midy*1.9 ) # Find x and y-midpoint
                    plt.annotate(sy, npn2,ha='center', va = 'center')

        mxx.append(adjx)
    maxx2 = np.max(np.array(mxx))
    frame1 = plt.gca()
    if set_maxy !=None:
        maxx = set_maxy
        plt.xlim(0, set_maxy+0.3)
        plt.ylim(0, maxy*2) #set_maxy/3+0.1 )

        frame1 = plt.gca()
    else:
        # st.write("TWO max x",maxx)
        # st.write("TWO max Y",maxy)
        xl = maxx2*1.1+0.3
        yl = maxy*2+0.1
        # st.write(f"xy lims: {xl}, {yl}" )
        plt.xlim(0, xl)
        plt.ylim(0, yl )
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    plt.annotate(df.sample_id.iloc[0] +"\n" + df.sample_region.iloc[0], (0, maxy+ 13/14*maxy), fontsize = 13) #(0, npp[1][1]+4.5))
    #npp[1][1]+3.5))

    frame1.axis('off')
    return


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



def f_rest_unc(Inv, D, ltl, Inv_unc, D_unc):
    list_of_partials = [1/(D-ltl*Inv), -Inv/( (D**2 )*(1-ltl*Inv/D))]
    list_of_uncerts = [Inv_unc, D_unc]
    return generic_squares(list_of_partials, list_of_uncerts)


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
    for r, ro in enumerate(dft['Inv']):
        v1 = redef_uf(dft['D'].iloc[r])
        ro = redef_uf(ro)
        rt_i = f_rest(ro, v1, ltl)
        rt_redef =redef_uf(rt_i)
        rtruf.append(rt_redef)

    dft['rt'] = rtruf
    res_t = dft['rt']

    return dft, res_t

def solve_E_fines(dft):
    #     dft, E_fines = solve_E_fines(dft)
    Et = []
    for r, ro in enumerate(dft['z']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['D'].iloc[r])
        v3 = redef_uf(dft['N'].iloc[r])
        v4 = redef_uf(dft['p_re'].iloc[r])
        Et_i = f_erate(v1, v2, v3, ltl, v4)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)

    dft['E_fines'] = Et
    #dft.apply(lambda x: f_erate(x['z'],x['D'], x['N'], ltl, x['p_re']), axis = 1)
    E_fines = dft['E_fines']
    return dft, E_fines

def solve_F_fines(dft):
    #     dft['F_fines_boxmodel'] = flux_boxmodel( SD, res_t, p_re)
    #     F_fines = dft['F_fines_boxmodel']
    #     dft['E_fines_pdez'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['z']], axis = 1)
    #     dft['E_fines_pdeD'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['D']], axis = 1)
    #     dft['E_fines_pdeN'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['N']], axis = 1)
    #     dft['E_fines_pdepre'] = dft.apply(lambda x: x['F_fines_boxmodel'].derivatives[x['p_re']], axis = 1)
    Et = []
    for r, ro in enumerate(dft['z']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['rt'].iloc[r])  # is this supposed to be in *yrs* or ky?
        v3 = redef_uf(dft['p_re'].iloc[r])
        Et_i = flux_boxmodel(v1, v2, v3)    # SD in cm , res_t in yr, ps in g/cm3: OUT is  g/(cm2*yr)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)

    dft['F_fines_boxmodel'] = Et
    F_fines = dft['F_fines_boxmodel']
    return dft, F_fines

def solve_F_coarse(dft):  # dft, F_coarse  = solve_F_coarse(dft)
    Et = []
    for r, ro in enumerate(dft['coarse_mass']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['coarse_area'].iloc[r])
        v3 = redef_uf(dft['max_coarse_residence_time'].iloc[r])
        Et_i = f_coarse_flux(v1, v2, v3)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)
    dft['F_coarse'] = copy.copy(Et)
    #     print(type(copy.copy(Et[0])))
    F_coarse = dft['F_coarse']
    return dft, F_coarse

def solve_F_br(dft):  # dft, F_br  = solve_F_br(dft)
    Et = []

    for r, ro in enumerate(dft['br_E_rate']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['p_br'].iloc[r])
        Et_i = f_br_flux(v1, v2)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)

    dft['F_br'] = Et
    F_br = dft['F_br']
    return dft, F_br

def solve_F_dust(dft):  # dft, F_dust  = solve_F_dust(dft)
    Et = []
    for r, ro in enumerate(dft['F_fines_boxmodel']):
        v1 = redef_uf(ro)
        v2 = redef_uf(dft['F_coarse'].iloc[r])
        v3 = redef_uf(dft['F_br'].iloc[r])
        v4 = redef_uf(dft['DF'].iloc[r])
        Et_i = f_mass_balance_for_dust(v1, v2, v3, v4)
        Et_ri = redef_uf(Et_i)
        Et.append(Et_ri)

    dft['F_dust'] = Et
    F_dust = dft['F_dust']

    return dft, F_dust

def set_up_df_for_flux_results3(df,Ncol = 'calc-10Be_at_g', N_unc_col = 'calc-uncert_10Be_at-g'):
    # not updating z_unc

    sitecols = []
    dft['z_site'] = dft.groupby('inv_sum_flag')['z'].transform('sum')
    dft['N_site'] = dft.groupby('inv_sum_flag')[Ncol].transform('mean')
    dft['N_site_unc'] = dft.groupby('inv_sum_flag')[N_unc_col].transform('mean')
    dft['N'] = dft.apply(lambda x: ufloat(x['N_site'], x['N_site_unc'], "N"), axis = 1)
    N = dft['N']
    dft['N_unc'] = dft['N_site_unc']

    dft['z'] = dft['z_site']


    return dft, SD, N


def get_new_df_results_w_unc2(df,dff, col_overwrite = False, val_overwrite = 0, flag_br_equiv = False, flag_coarse_subsurface = 10, flag_pde = False):
    # col_overwrite = ['coarse_mass'], val_overwrite= 2000,

    AZ_D_graly = D_graly(400, 31.2)
    SP_D_graly = D_graly(510, 39.1)

    #     print('Default df cols: ', dff.columns.to_list())
    #     print('L4: dff[\'coarse_mass\'].iloc[0]', dff['coarse_mass'].iloc[0])
    #     print('L4: dff[\'coarse_area\'].iloc[0]', dff['coarse_area'].iloc[0])
    if isinstance(col_overwrite, list):  # Then we wanna calculate non-default vals
        for i, ovrc in enumerate(col_overwrite):
            ovrval = val_overwrite[i] # Will write uniform value for all samples
            if ovrc not in dff.columns.to_list():
                print('whoops column to overwrite isnt in the default df, adding col anyways: ', ovrc )
            dff[ovrc] = ovrval

    dft, SD, N = set_up_df_for_flux_results3(df,dff)  # does not output ufloats yet
    #     print('L13: debug: ', dft.columns.to_list())
    #     print('L14: dft[\'z\'].iloc[0]', dft['z'].iloc[0])
    #     print('L14: dft[\'coarse_mass\'].iloc[0]', dft['coarse_mass'].iloc[0])
    #     print('L14: dft[\'coarse_area\'].iloc[0]', dft['coarse_area'].iloc[0])

    ## run through and modify anything that isn't ufloat, need to follow uncertainties....
    # Is ufcols just the cols of dff?
    ufcols = ['p_re','p_br', 'br_E_rate', 'coarse_mass', 'coarse_area', 'max_coarse_residence_time', 'z', 'D', 'DF']
    for i, ucol in enumerate(ufcols):#isinstance(uf, uncertainties.core.Variable) | isinstance(uf, uncertainties.core.AffineScalarFunc)\
    #         if ucol == 'coarse_mass':
    #             print('before coarsemass: ', dft[ucol].iloc[0])
        if isinstance(dft[ucol].iloc[0], uncertainties.core.Variable): # already done, don't do anything
    #             print('TRUE: isinstance(dft[ucol].iloc[0], uncertainties.core.Variable)')
            pass
        elif isinstance(dft[ucol].iloc[0],  uncertainties.core.AffineScalarFunc):
    #                print('TRUE: isinstance(dft[ucol].iloc[0],  uncertainties.core.AffineScalarFunc)')
            dft[ucol] = redef_uf(dft[ucol])
        else: # if any of those ufcols isn't already ufloat, then we need to change it and assume the unc col is there else 0
    #             print('TRUE: else, i.e. not a affine scalar func or variable')
            uncol = ucol + '_unc'
            if uncol not in dft.columns.to_list():
    #                 print('uncol: ', uncol)
                dft[uncol] = 0
            tarr = []
            for j, uval in enumerate(dft[ucol]): # redef uf doesn't need to loop through each row but ufloat does
                 tarr.append(ufloat(uval, dft[uncol].iloc[j], ucol))
            dft[ucol] = tarr
    #         if ucol == 'coarse_mass':
    #             print('after ufloate coarsemass: ', dft[ucol].iloc[0])

   # Post-df formation calculations. E.g. coarse mass in subsurface? need to redefine soil depth, bc soil depth is actually FINE soil depth....
    if flag_coarse_subsurface != False:
        if isinstance(flag_coarse_subsurface, float) | isinstance(flag_coarse_subsurface, int):  #flag_coarse_subsurface = percent_subsurface_by_vol_is_coarse
            # print(flag_coarse_subsurface, 'flag coarse subsurface is a float or int')


    dft['Inv'] = dft.apply(lambda x: f_Inv(x['N'],x['p_re'], x['z']), axis = 1)


    dft,res_t = solve_rt(dft)
    dft, E_fines = solve_E_fines(dft)
    # in mm/kyr
    # need mass fluxes --> unc sensitive to res time

    dft, F_fines = solve_F_fines(dft)
    dft, F_coarse  = solve_F_coarse(dft)
    dft, F_br  = solve_F_br(dft)
    dft, F_dust  = solve_F_dust(dft)

    dft['F_fines_from_br'] = dft['F_fines_boxmodel'] - dft['F_dust']
    dft['F_dissolved'] = (dft['F_fines_boxmodel'] - dft['F_dust']) * dft['DF']

    # These should be equivalent: LHS = RHS of mass balance
    dft['F_br_plus_F_dust'] = dft['F_br'] + dft['F_dust']
    dft['F_coarse_plus_F_fines_plus_F_dissolved']= dft['F_coarse'] + dft['F_fines_boxmodel'] + dft['F_dissolved']

    DF = dft['DF']
    p_re = dft['p_re']

    to_m2_cols = [co for co in dft.columns.to_list() if co.startswith('F_')]
    # ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust', 'F_br_solids_after_chem_wx','F_fines_from_br', 'F_dissolved', 'F_br_plus_F_dust','F_coarse_plus_F_fines_plus_F_dissolved','F_coarse_normbr', 'F_fines_from_br_normbr','F_fines_boxmodel_normbr', 'F_dust_normbr', 'F_dissolved_normbr', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']
    dft = dft.copy()
    # Change fluxes to m2
    # g/cm2/yr  * 100*100cm2/m2
    for c,cc in enumerate(to_m2_cols):
        dft[cc + '_g_m2_yr'] = dft[cc].apply(lambda x: x*10000).copy()
    dft = dft.copy()

    dft['rt_ky'] = dft['rt'].copy() /1000 # ky
    dft = dft.copy()
    # At very end, run through all columns and make sure none are Affine Scalar Function
    for c, col in enumerate(dft.columns.to_list()):
        fv = dft[col].iloc[0]
        dft = dft.copy()
    # Actually I think remaking all ufloats will improve those horribly long decimals        if isinstance(fv, uncertainties.core.Variable):
    #             dft[col + '_val'], dft[col + '_unc'] = get_vals_uf(dft[col])
        if isinstance(fv, uncertainties.core.AffineScalarFunc) | isinstance(fv, uncertainties.core.Variable):
    #             print('Line 96:' , col, dft[col].iloc[0],'\n', dft[col])
            dft = dft.copy()
            dft[col] = redef_uf(dft[col])

            dft[col + '_val'], dft[col + '_unc'] = get_vals_uf(dft[col])
            dft = dft.copy()

    #             print(col, dft[col].iloc[1])
        else: pass

    return dft



def modify_start_subsoil_coarse_seds(dft, percent_subsurface_by_vol_is_coarse):
    coarse_mass = dft['coarse_mass'].iloc[0]
    coarse_area= dft['coarse_area'].iloc[0]
    p_re = dft['p_re'].iloc[0]
    SD = dft['z'].copy()
    p_br = dft['p_br'].iloc[0]
    #     print('In modify_start_subsoil_coarse_seds: \ncoarse mass: ', coarse_mass,'\ncoarse area' ,coarse_area, '\np_re: ' , p_re, '\nSD: ', SD, '\np_br', p_br)
    SDnew = SD-SD*percent_subsurface_by_vol_is_coarse/100   # h_fines
    coarse_massnew = coarse_mass+SD*percent_subsurface_by_vol_is_coarse/100*p_br*coarse_area   # equivalent mass of subsurface clasts
    #p_renew = some kind of table Poesen
    return SDnew, coarse_massnew
def D_graly(P,L, flag_P_is_mm_per_yr = True):
    # Graly 2010a
    # latitude = L
    # P = precip in cm/yr
    # flux in 10^4 at/cm3  but elsewhere in the paper it's written at/cm2/yr....
    if flag_P_is_mm_per_yr:
        P = P/10
    flux = P*(1.44/(1+np.exp((30.7-L)/4.36))+0.63)*10**4
    return flux

def htcvstuff():
        # F:\Ch4_Soils\hilltop_cv_recalc
    dict_nh_6 = {'MT120': -0.038960774739689605, 'MT130': -0.014048258463692646, 'MT140': -0.0039367675781761475, 'MT150': -0.02202351888028153, 'MT160': -0.002990722656353995, 'MT170': 0.0003051757813126851, 'NQT0': -0.025085449218782006, 'NQCC2': 0.019994099934862595, 'NQCV2': 0.01474761962887685, 'NQCV4': 0.04274241129554823, 'NQPR0': 0.1418329874674209}
    dict_nh_10 ={'MT120': -0.03911655970982906, 'MT130': -0.012145996093766386, 'MT140': -0.0047119140625133905, 'MT150': -0.015209089006695816, 'MT160': -0.004703194754474845, 'MT170': -0.004143415178590641, 'NQT0': -0.01717158726283599, 'NQCC2': -0.0025072370256786973, 'NQCV2': -0.00048522949219118776, 'NQCV4': -0.007194301060272058, 'NQPR0': 0.06417105538504028}
    dict_nh_18={'MT120': -0.03511895073784176, 'MT130': -0.012553271449137033, 'MT140': -0.004107586749181472, 'MT150': -0.010360974655399313, 'MT160': -0.00732709582806795, 'MT170': -0.006098664225488383, 'NQT0': -0.0171899561586119, 'NQCC2': -0.005148413778781948, 'NQCV2': -0.008519760285962767, 'NQCV4': -0.006003663768096233, 'NQPR0': 0.0159514345992028}
    dict_nh_26={'MT120': -0.03135276963871658, 'MT130': -0.01397734158119797, 'MT140': -0.004119676053365337, 'MT150': -0.007644201876851472, 'MT160': -0.007458052389321181, 'MT170': -0.006491658506760771, 'NQT0': -0.015380514636356259, 'NQCC2': -0.0003897803903206735, 'NQCV2': -0.012826356570488513, 'NQCV4': -0.0012080726749591534, 'NQPR0': 0.008571178882152648}
    dict_nh_30={'MT120': -0.029847367789513832, 'MT130': -0.014693424460757123, 'MT140': -0.00437018984840149, 'MT150': -0.0072553227364439395, 'MT160': -0.0073637722535687036, 'MT170': -0.006574322092870191, 'NQT0': -0.014767275964152931, 'NQCC2': -0.0004559564374692479, 'NQCV2': -0.01396707100386399, 'NQCV4': 0.00022294262929375627, 'NQPR0': 0.009526058312917179}

    dft['ht_cv_actual_topo_nh6m'] = dft.sample_id.map(dict_nh_6)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh10m'] = dft.sample_id.map(dict_nh_10)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh18m'] = dft.sample_id.map(dict_nh_18)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh26m'] = dft.sample_id.map(dict_nh_26)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh30m'] = dft.sample_id.map(dict_nh_30)  # from Ch 3 table 1
    dft['ht_cv_actual_topo'] = dft['ht_cv_actual_topo_nh18m'].copy()
    # dft['ht_cv_actual_topo'] = dft.sample_id.map( {'MT120':0.06, 'MT130': 0.0067, 'MT150': 0.0004, 'NQT0':0.02, 'NQCV2':0.0021, 'NQCV4': 0.0026})

    # These are from cv8sm: {'MT120':0.009, 'MT130': 0.0067, 'MT150': 0.0004, 'NQT0':0.0075, 'NQCV2':0.0021, 'NQCV4': 0.0026})    # Slightly diff locations... these are from Ch 3 --- super high compared to profile curvature... from col DV in 10Be Data reduction....{'MT120': 0.058, 'NQT0': 0.018, 'NQCV2': 0.018})  # from Ch 3 table 1

    # Amtof br that goes to dissolved   / total input
    # dft['CDF'] = ( (dft['F_br'] - dft['F_coarse'])*(1-1/dft['DF'])  - dft['F_coarse'])/dft['F_br']  ##  ## Scaling by density ratios,  needs to incorporate coarse sediment as well
    # dft['CDF_dust'] = ( (dft['F_br'] - dft['F_coarse'])*(1-1/dft['DF'])  - dft['F_coarse'] - dft['F_dust'])/(dft['F_br'] + dft['F_dust'])
    # new CDF Formulation
    dft['CDF'] = dft['F_dissolved'] /dft['F_br']
    dft['CDF_dust'] = dft['F_dissolved'] /(dft['F_br'] + dft['F_dust'])


    # IF FLUXES ARE IN g/m2/yr
    #g/m2/yr/cm2*yr*cm3/g = cm/m2 = cm*1m/100cm/m2 = 1/m/100

    ## IF FLUXES ARE IN g/cm2/yr
    # K is in cm2/yr so it's OK that Fluxes are in g/cm2/yr as well // actually need x100
    # g/cm2/yr / (g/cm3) / (cm2/yr) = cm/yr / (cm2/yr) = 1/cm
    # if ht cv should be in m, then 1/cm * 100 cm/1m = 100 / m
    # p_re should incorporate the coarse sediments as well....
    ht_cv = -(dft['F_br'] + dft['F_dust'])*(1-dft['CDF'] )/p_re / dft['K']

    dft['ht_cv_BenAsher'] = ht_cv*100

    dft['ht_cv_nodust'] = -(dft['F_br'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100
    # This one makes no sense: CDF already takes into account the dissolutb dft['ht_cv_dust_minus_dissolution'] = -(dft['F_br']+ dft['F_dust'] - dft['F_dissolved'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100

    dft['ht_cv_noCDF_noDust'] =  -(dft['F_br'] )/p_re / dft['K'] * 100

    dft['ht_cv_dustnoCDF'] = -(dft['F_br']*(1-dft['CDF'] ) +  dft['F_dust'])  /p_re / dft['K'] * 100


    dft['ht_cv_br_dust_nocdf'] =  -(dft['F_br'] + dft['F_dust'])/p_re / dft['K'] * 100

    # See excel sheet --- these simplify the CDF * solid matl as above so probably same
    dft['ht_cv_v1_dust'] = -(dft['F_coarse'] + dft['F_fines_from_br'] + dft['F_dust'] )/ dft['K'] / p_re *100
    dft['ht_cv_v2_brwx'] = -(dft['F_coarse'] + dft['F_fines_from_br'])/ dft['K'] / p_re *100
    dft['ht_cv_v3_nowx_dust'] = -(dft['F_br'] + dft['F_dust'])/ dft['K'] / p_re *100

    ht_cv = -(dft['F_br'] + dft['F_dust'])*(1-dft['CDF_dust'] )/p_re / dft['K']

    dft['ht_cv_BenAsher_CDF_dust'] = ht_cv*100

    dft['ht_cv_nodust_CDF_dust'] = -(dft['F_br'] )*(1-dft['CDF_dust'] )/p_re / dft['K'] * 100
    # This one makes no sense: CDF already takes into account the dissolutb dft['ht_cv_dust_minus_dissolution'] = -(dft['F_br']+ dft['F_dust'] - dft['F_dissolved'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100

    dft['ht_cv_dustnoCDF_CDF_dust'] = -(dft['F_br']*(1-dft['CDF_dust'] ) +  dft['F_dust'])  /p_re / dft['K'] * 100



    # Simple mass balance model:
    # Fbr = Fcoarse + Ffines + Fdissolved, ---> Fbr and Fcoarse and Ffines are the same as in more complexs model, but LHS is smaller so RHS also has to be smaller...
    # RHS simple mass balance model:
    dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']  =dft['F_br']  - dft['F_coarse'] - dft['F_fines_boxmodel']


    dft['ht_cv_actual_minus_nodustcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_nodust']
    dft['ht_cv_actual_minus_dustcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_BenAsher']
    dft['ht_cv_actual_minus_dustnoCDFcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_dustnoCDF']
    dft['ht_cv_actual_minus_noCDFcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_noCDF_noDust']
    # dft['ht_cv_actual_minus_simpledissolutioncv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_F_br_minus_simpledissolution']

    dft['ht_cv_diff2_nodustcv_dust_cv'] = dft['ht_cv_actual_minus_nodustcv'] -dft['ht_cv_actual_minus_dustcv']
    dft['ht_cv_diff2_nodustcv_dustnoCDF_cv'] =  dft['ht_cv_actual_minus_nodustcv'] - dft['ht_cv_actual_minus_dustnoCDFcv']
    dft['ht_cv_diff2_dustcv_dustnoCDF_cv'] =  dft['ht_cv_actual_minus_dustcv'] - dft['ht_cv_actual_minus_dustnoCDFcv']

    #  ['ht_cv_diff2_nodustcv_dust_cv', 'ht_cv_diff2_nodustcv_dustnoCDF_cv', 'ht_cv_diff2_dustcv_dustnoCDF_cv']

    # ht_cv_dust_minus_dissolution_val  ht_cv_F_br_minus_simpledissolution_val
    # Simple nodust model : (F_dissolved + F_fines)/F_fines

    dft['Simple_model_apparent_DF'] = (dft['F_br']  - dft['F_coarse'] )/ dft['F_fines_boxmodel']
    dft['Dust_model_apparent_DF'] = (dft['F_br']  - dft['F_coarse'] )/ dft['F_fines_from_br']

    # Can solve for DF if we assume Fd:
    # (F_dust - F_fines) (DF + 1) = F_coarse - F_br
    # F_dust*DF - F_fines*DF + F_dust -F_fines = F_coarse - F_br
    # DF(F_dust - F_fines) + F_dust - F_fines = F_coarse - F_br
    # F_br + F_dust = F_coarse  + F_fines + DF(F_fines-F_dust )
        #DF = (F_coarse - F_br)/(F_dust - F_fines) - 1
    dft['DF_if_Fd_is_set_to_1'] = (F_coarse - F_br)/(1/10000 - F_fines) -1     # g/m2/yr
    dft['DF_if_Fd_is_set_to_5'] = (F_coarse - F_br)/(5/10000 - F_fines) -1     # g/m2/yr
    dft['DF_if_Fd_is_set_to_10'] = (F_coarse - F_br)/(10/10000 - F_fines) -1
    dft['DF_if_Fd_is_set_to_15'] = (F_coarse - F_br)/(15/10000 - F_fines) -1
    dft['DF_if_Fd_is_set_to_20'] = (F_coarse - F_br)/(20/10000 - F_fines) -1



def write_defaults_to_df2(df, col_overwrite= 'none', dict_val_overwrite = [], Dcol_overwrite = 'none', dcol_dict_val = []):
    # Rules:
    # D --> if MT, then Graly AZ, else Graly SP
    AZ_D_graly = D_graly(400, 31.2)
    SP_D_graly = D_graly(510, 39.1)
    # AZ_D_graly = D_graly(450, 31.2)
    # SP_D_graly = D_graly(450, 39.1)

    if col_overwrite == 'D':
        D_dflt_dct = dict_val_overwrite
    else:
        D_dflt_dct = {'AZ': AZ_D_graly, 'SP': SP_D_graly}

    if Dcol_overwrite != 'none':
        D_dflt_dct = dcol_dict_val

    # DF --> based on cao soil/cao br; SP: ~4% cao in soil so 25 DF, MT: ~ 6% cao in soil so ~ 17 DF. BUT some soils in both samples have ~ 20% cao AND if include MgO then MT 9, 30%, and SP is 7,8, 26%. But I think these are probably sm rock fragments elevating these vals.... so use lowest
    if col_overwrite == 'DF':
        DF_dflt_dct =dict_val_overwrite
    else:
        DF_dflt_dct = {'AZ': 17, 'SP': 25}
    # p_re --> 1.4
    # p_re, p_br, coarse_area, max_coarse_residence_time, p_crs, coarse_mass
    if col_overwrite == 'p_re':
        p_re = dict_val_overwrite
    else:
        p_re = 1.4
    # p_br --> 2.6
    if col_overwrite == 'p_br':
        p_br = dict_val_overwrite
    else:
        p_br = 2.6
    # coarse area
    if col_overwrite == 'coarse_area':
        coarse_area = dict_val_overwrite
    else:
        coarse_area = 300
    # max coarse residence time.
    if col_overwrite == 'coarse_area':
        max_coarse_residence_time = dict_val_overwrite
    else:
        max_coarse_residence_time = 15500
    # density of coarse sediment modify? but if anything I want to overestimate the coarse sedments SHOULD be 2.2 or so (Poesen Laveee 1994)
    p_crs = 2.2
    # coarse mass
    if col_overwrite == 'coarse_mass':
        coarse_mass = dict_val_overwrite
    else:
        coarse_mass = 1133



    if col_overwrite == 'K': # Sed transport coeff
        K_sed = dict_val_overwrite
    else:
    # dft['K'] = dft.sample_region.map()
        K_sed = {'AZ':2.6, 'SP':14.6}
    # # br E rate: use closest E rate to sample....(br or clast), N14 for all meteoric except NQCV4 --> N04 br, NQPR0 --> C01 or BR04-C
        # Br, BRunc, ID, Name  (bulk as measured)
        # 10.2  2.4 13  BH04-B
        # 47.3  8.6 3   Bic-1
        # 24.2  4.3 1   Bo1-2-2-15
        #                             12.9  2.8 5   BR04-C
        #                           11.8    2.6 6   C012-15.85
        #                           11.5    2.6 7   C01gt15.85
        #                 7.6   1.7 14  MT1
        #                 16.2  3.5 15  MT6
        #                           15.6    3.4 8   N04-BR
        # 14.5  3.1 9   N08-BR
        # 19.7  4.1 10  N11-Cgt15.85
        #                           11.5    2.6 11  N14-BR
        # 13.0  2.8 12  N14-C-gt15
        # 40.4  7.2 4   R01c-2-15
        # 57.1  10.3    2   RioG-1-2-15


    MT1_E = [7.6,1.7]
    MT6_E = [16.2,3.5]
    N14br_E = [11.5, 2.6]
    N04br_E = [15.6,3.4]
    C01_E = [11.8,2.6]
    if col_overwrite == 'br_E':
        br_E_dflt_dct = dict_val_overwrite
    else:
        br_E_dflt_dct = {'MT120':MT1_E,  'MT130':MT1_E, 'MT140':MT1_E,  'MT150':MT6_E,'MT160':MT6_E, 'MT170':MT6_E, 'NQT0':N14br_E, 'NQCC2':N14br_E,'NQCV2':N14br_E, 'NQCV4':N04br_E, 'NQPR0':C01_E}

    samp_dflt_dct = {'MT120':'AZ',  'MT130':'AZ', 'MT140':'AZ',  'MT150':'AZ','MT160':'AZ', 'MT170':'AZ', 'NQT0':'SP', 'NQCC2':'SP','NQCV2':'SP', 'NQCV4':'SP', 'NQPR0':'SP'}
    count = 0



    for key in br_E_dflt_dct:
        # make new dict, then append as df to big df
        sampreg = df['sample_region'].loc[df['sample_id'] == key].copy().to_list()[0]
        dct = {'K': K_sed[sampreg], 'D': D_dflt_dct[sampreg],'DF':DF_dflt_dct[sampreg], 'p_re':p_re,  'p_br':p_br, 'coarse_area': coarse_area, 'max_coarse_residence_time':max_coarse_residence_time,'p_crs':p_crs, 'coarse_mass': coarse_mass, 'br_E_rate': br_E_dflt_dct[key][0], 'br_E_rate_unc': br_E_dflt_dct[key][1], 'zcol': 'local_sd_avg' }


        dft = pd.DataFrame(dct, index = [key])
        dft['sample_id']=key
        dft['sample_region'] = samp_dflt_dct[key]
        if count == 0:
            dfdflt = dft.copy()
        else:
            dfdflt = pd.concat([dfdflt, dft])
        count +=1
    return dfdflt


def prep_initial_df(df):
    cols = ['calc-uncert_10Be_at-g', 'calc-10Be_at_g', 'inv_sum_flag', 'sample_region', 'sample_location', 'sample_position', 'sample_top', 'sample_bot', 'sample_vol_cm', 'sample_type', 'sample_id', 'sample_is_transect', 'sample_is_profile', 'local_soildepth1','local_soildepth2', 'local_soildepth3', 'local_soildepth4','local_soildepth_avg', 's_lat', 's_lon', 's_elev', 's_elev_src', 'slpcombined', 's_cv', 's_dx', 's_dx (cm)', 'Inventory (at/cm2) each site', 'cv_prof_sm8m', 'cv_prof_sm30m','cv_tot_sm30m', 'gradient_sm8_in_degrees', 's_dx (cm) upslope', 'Inventory each site_localdepth', 'depth_avg_conc_each_site', 'Inventory[at/cm2]']
    dft = df[cols].copy()
    # local soildepth min/max
    loc_sd = ['local_soildepth1','local_soildepth2', 'local_soildepth3', 'local_soildepth4']
    loc_sd_agg  = ['local_sd_avg', 'local_sd_min', 'local_sd_max']

    df['local_sd_avg'] = df[loc_sd].mean(axis=1)
    df['local_sd_min'] = df[loc_sd].min(axis=1)
    df['local_sd_max'] = df[loc_sd].max(axis=1)
    # df[['local_sd_avg','local_sd_min', 'local_sd_max']]
    df['local_sd_avg'] = df['local_sd_avg'].fillna(df['local_soildepth_avg']).fillna(df['sample_vol_cm'])
    df['local_sd_min'] = df['local_sd_min'].fillna(df['local_soildepth_avg']).fillna(df['sample_vol_cm'])
    df['local_sd_max'] = df['local_sd_max'].fillna(df['local_soildepth_avg']).fillna(df['sample_vol_cm'])
    # df[['local_sd_avg','local_sd_min', 'local_sd_max']]
    return df



def D_graly(P,L, flag_P_is_mm_per_yr = True):
    # Graly 2010a
    # latitude = L
    # P = precip in cm/yr
    # flux in 10^4 at/cm3  but elsewhere in the paper it's written at/cm2/yr....
    if flag_P_is_mm_per_yr:
        P = P/10
    flux = P*(1.44/(1+np.exp((30.7-L)/4.36))+0.63)*10**4
    return flux


# D = 1.8e6 # at/cm2/yr
# p_re = 1.4 # g/cm3
# lamba = 5.1e-7 # 1/yr
# atom10Bemass = 1.6634e-23 #g/at
# AZ_D_graly = D_graly(400, 31.2)
# SP_D_graly = D_graly(510, 39.1) # used to be 450 mm/yr precip...
# #lambda
# ltl = 5.1e-7 # , 7.20e-7]  #1/yr lambda [Nishiizumi et al., 2007], Korschinek et al. (2010)
# df = pd.read_excel(r'C:\Users\nariv\OneDrive\Research\Landuse\SampleWorkup\Be-10-Samples\10Be_Data_Reduction_Miller.xlsx', sheet_name = '10BeCalculations', skiprows = 9, nrows = 29)


# df = prep_initial_df(df)

# dff = write_defaults_to_df(df)


# #     run_all_vars_4_23(df, saveloc,dff_overwrite, dff_val, Dcol_overwrite, dcol_dict_val, dirn, saveloc2, plotstuff = False)
# vals_arr =  [0,25,50,75]
# col_overwrite = 'D'   # not doing anything, as long as it's not isinstance, list, I think.

# table_name = 'CoarseSeds_subsurface_pct'
# summary_val_name = 'Coarse_seds_subsurface'
# printdfshort = False

# flag_coarse_subsurface = True
# tn = table_name
# dff_overwrite = dff_overwrite
# dff_val = dff_val
# Dcol_overwrite = Dcol_overwrite
# dcol_dict_val = dcol_dict_val
# print_df_short = printdfshort
# summary_val_name = summary_val_name
# col_overwrite =col_overwrite
# vals_arr = vals_arr
# flag_coarse_subsurface = flag_coarse_subsurface
# saveloc = saveloc
# dirn = dirn


def wrap_newdf2(dfto,K = 14.6, print_df_short = False, summary_val_name = 'none',  col_overwrite = 'none', Dcol_overwrite = 'none', dcol_dict_val = [], vals_arr = 0,dff_overwrite = 'none', dff_val = [], flag_br_equiv = False, flag_coarse_subsurface = False, flag_pde = False, tn = 'TableName',saveloc = 'F:\Ch4_Soils\Tables', dirn = '\\24_26_01', dict_of_fmt = None):
    if summary_val_name != 'none':
        col = summary_val_name
        # print('col is summary_val_name')
    else:
        col = col_overwrite[0]

    dff = write_defaults_to_df2(dfto,col_overwrite= dff_overwrite, dict_val_overwrite =dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val)
    #     print('Default df cols: ', dff.columns.to_list())
    #     print('L4: dff[\'coarse_mass\'].iloc[0]', dff['coarse_mass'].iloc[0])
    #     print('L4: dff[\'coarse_area\'].iloc[0]', dff['coarse_area'].iloc[0])
    if isinstance(dff_overwrite, list):  # Then we wanna calculate non-default vals
        for i, ovrc in enumerate(dff_overwrite):
            ovrval = dff_val[i] # Will write uniform value for all samples
            if ovrc not in dff.columns.to_list():
                print('whoops column to overwrite isnt in the default df, adding col anyways: ', ovrc )
            dff[ovrc] = ovrval

    dff.drop(['sample_region', 'sample_id'], axis = 1, inplace = True)

    dft = get_new_df_results_w_unc2(dfto,dff, col_overwrite = col_overwrite,
        val_overwrite = [val],
        flag_br_equiv = flag_br_equiv,
        flag_coarse_subsurface = flag_coarse_subsurface,
        flag_pde = flag_pde)

    return dfn
