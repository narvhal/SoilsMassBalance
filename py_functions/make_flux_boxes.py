
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
    mtd = {'simple':"Model A: Solve for dissolved flux, mass balance, no dust ('simple')",
    'carbbalance': "Model C: Solve for dissolved flux, carbonate balance",
    'wdust':  "Model B: Solve for dust flux, mass balance, carbonate balance " }
    return mtd[mt]


def wrap_flux_box_streamlit(dft, selval_dict):
    fig, ax = plt.subplots()
    height = selval_dict['model_shape']
    flag_model = selval_dict['model_type']
    list_of_tuplelists,ft,fst, height , L, H, XY, fst, YC = make_into_area_streamlit(dft, selval_dict, flag_model = flag_model, height = height, scale = selval_dict["boxscale"] , shape_buffer = selval_dict["shape_buffer"])
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

def make_into_area_streamlit(df,selval_dict, flag_model= 'simple', height = 'auto', scale = 1, shape_buffer = .75):
    if flag_model == 'simple':
        spacerloc = 0

    elif flag_model == "carbbalance":
        spacerloc = 0

    else:
        spacerloc = 1
    fmcols = selval_dict["fmcols"]
    ft = selval_dict["ft"]
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
        ec = 'dimgrey'
    elif flag_model == 'carbbalance':
        hch = ['+', '+', '+', '+']
        bxc = ['grey', 'rosybrown', 'indianred', 'lightcyan']
        ec = 'dimgrey'
    else:
        hch = ['x','', '', '', '', '']
        bxc = ['grey','burlywood', 'rosybrown', 'indianred', 'lightcyan', 'burlywood']
        ec = 'dimgrey'

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
            ax.add_patch(mpatches.Polygon(npp, ec = ec, fc = bxc[i], hatch = hch[i], ls = '-', lw = .5))  # df.cdict.iloc[0]
            # npn = ( (npp[0][1]+npp[1][1])/2 ,  (npp[1][0] + npp[0][0])/2 )  # Find x and y-midpoint
            npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2, midy )  # Find x and y-midpoint
            # st.write(npp)
            # st.write(f"Area {ft[i]}: {(npp[1][1]-npp[1][0])* (npp[2][0]-npp[1][0])}")
            # st.write(f"Orig: {fst[i]}" )
            # st.write("npp Points:",npp[1])
            if flag_model == 'simple':
                pct_denom = fst[0]  # just bedrock flux
            elif flag_model == 'carbbalance':
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
                elif flag_model == 'carbbalance':
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

## meh, this equation turns negative.... idk
def other_flux(df):
    lda = 5.1*1e4   # 1/yr
    N = dft['N'].values[0]
    p_re = dft['p_re'].values[0]
    z = dft['z'].values[0]

    df['otherflux'] = (df['D'] - lda*df['Inv']) / (df['N']/df['z'])*df['p_re']*1e4  # g/m2/yr
    st.write("E*p_re = (D-lambda*I)/(N/volume)*p_re")
    st.write("{:0.2f} = ({:0.1e}-lambda*{:0.2e})/({:0.2e}/{:0.1f})*{:0.2f}".format(df['otherflux'].values[0], df['D'].values[0], df['Inv'].values[0], df['N'].values[0], df['z'].values[0], df['p_re'].values[0]))
    return df  


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
    return uf
    # # returns val, stdev
    # if isinstance(uf, uncertainties.core.Variable) | isinstance(uf, uncertainties.core.AffineScalarFunc):
    #     return ufloat(uf.nominal_value, uf.std_dev)
    # else:  # not a single number....
    #     ufa = []
    #     for i, ufv in enumerate(uf):
    #         ufa.append(ufloat(unumpy.nominal_values(ufv), unumpy.std_devs(ufv)))
    #     return np.array(ufa) # ufloat(unumpy.nominal_values(uf), unumpy.std_devs(uf))

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
def solve_rt(dft, flag_pde = False, ltl = 5.1e-7):
    # rtruf = []
    # for r, ro in enumerate(dft['Inv']):
    #     # v1 = redef_uf(dft['D'].iloc[r])
    #     # ro = redef_uf(ro)
    #     v1 =dft['D'].iloc[r]
    #     rtruf.append(f_rest(ro, v1, ltl))

    dft['rt'] = (-1.0/ ltl) * log(1 - (ltl* dft['Inv']/ dft['D']))
    res_t = dft['rt']

    return dft, res_t

def solve_E_fines(dft, ltl = 5.1e-7):
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


# def wrap_newdf2(dfto,K = 14.6, print_df_short = False, summary_val_name = 'none',
#     col_overwrite = 'none', Dcol_overwrite = 'none', dcol_dict_val = [],
#     vals_arr = 0,dff_overwrite = 'none', dff_val = [], flag_br_equiv = False,
#     flag_coarse_subsurface = False, flag_pde = False, tn = 'TableName',
#     saveloc = 'F:\Ch4_Soils\Tables', dirn = '\\24_26_01', dict_of_fmt = None):
#     if summary_val_name != 'none':
#         col = summary_val_name
#         # print('col is summary_val_name')
#     else:
#         col = col_overwrite[0]

#     dff = write_defaults_to_df2(dfto,col_overwrite= dff_overwrite, dict_val_overwrite =dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val)
#     #     print('Default df cols: ', dff.columns.to_list())
#     #     print('L4: dff[\'coarse_mass\'].iloc[0]', dff['coarse_mass'].iloc[0])
#     #     print('L4: dff[\'coarse_area\'].iloc[0]', dff['coarse_area'].iloc[0])
#     if isinstance(dff_overwrite, list):  # Then we wanna calculate non-default vals
#         for i, ovrc in enumerate(dff_overwrite):
#             ovrval = dff_val[i] # Will write uniform value for all samples
#             if ovrc not in dff.columns.to_list():
#                 print('whoops column to overwrite isnt in the default df, adding col anyways: ', ovrc )
#             dff[ovrc] = ovrval

#     dff.drop(['sample_region', 'sample_id'], axis = 1, inplace = True)

#     dft = get_new_df_results_w_unc2(dfto,dff, col_overwrite = col_overwrite,
#         val_overwrite = [val],
#         flag_br_equiv = flag_br_equiv,
#         flag_coarse_subsurface = flag_coarse_subsurface,
#         flag_pde = flag_pde)

#     return dfn


def varvalsfmt(mt):   # functions to provide vals for 'model_type'
    varvalues_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": ["Regional Default", "0.5x $D_{AZ}$", "$D_{AZ}$", "1.5x $D_{AZ}$", "0.5 x $D_{Sp}$", "$D_{Sp}$", "1.5x $D_{Sp}$", "4x $D_{Sp}$"],
                "DF":[2.5, 5, 7.5, 15, 22.5],
                "p_re": [0.1, 0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5, 50, 70],
                "coarse_mass": [.75, 1.5, 2.25],
                "max_coarse_residence_time":[2, 5.5, 11., 16.5],
                "z": ["Site Measurement", 5, 10, 20, 50],
                "carb_br": [50, 90, 100],
                "carb_soil": [0, 5, 10, 15, 20, 25, 30]
                }
    return varvalues_dict[mt]


def Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six, index = 0):
    filtselcol = varnames_dict2[selcolkey]
    # st.write(f'Varvaldict {varvalues_dict[selcolkey]}')
    # st.write(f'Vars dict {vars_dict[selcolkey]}')
    # selcolkey = list(varnames_dict2.keys())[fix]
        # with colll[count]:
    # bc of the way I structured the df, there is no column for coarse seds subsurface, instead it is "select_col_val"
    # st.write(k, list(vlist[:]))
    # st.write(k in df_default.columns.to_list())
    # st.write(df_default[k])   index = def_ix,
    # vll = list(vlist[:])
    # def_ix = vll.index(default_dict[k])   # Lots of weird errors here as I try to set the default value "value" for the radio button. ugh.
    # if filtselcol in selcolu:
    keystr = str(selcolkey) + "_radioval_"+ str(six)

    # st.write("2, ")

    # st.dataframe(dft["D"])


    vvd =varvalues_dict[selcolkey]
    font_size = 20

    html_str = f"""
        <style>
        p.a {{
          font: bold {font_size}px Times New Roman;
        }}
        </style>
        <p class="a">{varnames_dict2[selcolkey]}</p>
        """
    rmd = f"**{varnames_dict2[selcolkey]}**"
    # st.markdown(html_str, unsafe_allow_html=True)
    val = st.radio(rmd, vvd, index = index,
        key = keystr, on_change=proc, args = (keystr,), horizontal = True)
    # st.write("3, ")
    # st.dataframe( dft["D"])
    st.write(" " )

    v2vdt = {varvalues_dict[selcolkey][ii]:vars_dict[selcolkey][ii] for ii in range(len(varvalues_dict[selcolkey]))}
    if selcolkey =="D":
        v2vdt["Regional Default"] = dft["D"].iloc[0]
    if selcolkey == "z":
        v2vdt["Site Measurement"] = dft["z"].iloc[0]
    # st.write("v2vdt: ", v2vdt)
    # st.write("val: ", val)
    selval_dict[selcolkey] = v2vdt[val]

    dft[selcolkey] = v2vdt[val]

    return dft, selval_dict


def simple_recalc(dft, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six,fmtcols, ltl = 5.1e-7):
    flag_coarse_subsurface = float(selval_dict['Coarse_seds_subsurface'])
    if flag_coarse_subsurface>0:
        SD, coarse_mass = modify_start_subsoil_coarse_seds(dft, flag_coarse_subsurface)
    else:
        SD =  dft['z'].copy()
        coarse_mass = dft['coarse_mass'].iloc[0]

    zl = []
    dft['z_old'] = dft['z'].copy()
    dft['coarse_mass_old'] = dft['coarse_mass'].copy()
#         print(dft['coarse_mass_old'])
    for j, vz in enumerate(dft['z']):
        vz = float(vz)
        zl.append(vz - vz*flag_coarse_subsurface/100)
    dft['z'] =zl
    dft['coarse_mass'] = coarse_mass

   # Post-df formation calculations. E.g. coarse mass in subsurface? need to redefine soil depth, bc soil depth is actually FINE soil depth....

    dft['Inv'] = dft.apply(lambda x: f_Inv(x['N'],x['p_re'], x['z']), axis = 1)
    
    dft['rt'] = (-1.0/ ltl) * log(1 - (ltl* dft['Inv']/ dft['D']))
    v1 = dft['z']
    v2 = dft['D']
    v3 = dft['N']
    v4 = dft['p_re']
    dft['E_fines'] =f_erate(v1, v2, v3, ltl, v4)

    v1 = dft['z']
    v2 = dft['rt']# is this supposed to be in *yrs* or ky?
    v3 = dft['p_re']
    # SD, res_t, ps)
    dft['F_fines_boxmodel'] =  flux_boxmodel(v1, v2, v3)
    # st.write("p_re:", v3)
    # st.write("rt:", v2)
    # st.write("z:", v1)
    # st.write("F_fines_boxmodel:", dft['F_fines_boxmodel'].iloc[0])
    # st.write(f"I = N*p_re*z")
    # st.write(f"rt = (-1/lambda)*log(1-(lambda*I/D))")
    # st.write(f"Ff = z*p_re/rt: {v1*v3/v2}" )


    v1 = dft['coarse_mass']
    v2 = dft['coarse_area']
    v3 = dft['max_coarse_residence_time']

    dft['F_coarse'] = f_coarse_flux(v1, v2, v3)



    v1 = dft['br_E_rate']
    v2 = dft['p_br']
    # st.write("before ", v1, v2, f_br_flux(v1, v2) )

    dft['F_br'] = f_br_flux(v1, v2)
    # st.write(v1, v2, f_br_flux(v1, v2) )



    v1 = dft['F_fines_boxmodel']
    v2 = dft['F_coarse']
    v3 = dft['F_br']
    v4 = dft['DF']
    dft['F_dust'] =  f_mass_balance_for_dust(v1, v2, v3, v4)
    # st.write("3   ", dft[fmtcols].iloc[0])


# #                     dft, E_fines = solve_E_fines(dft)
# #                     # in mm/kyr
# #                     # need mass fluxes --> unc sensitive to res time

# #                     dft, F_fines = solve_F_fines(dft)
# #                     dft, F_coarse  = solve_F_coarse(dft)
# #                     dft, F_br  = solve_F_br(dft)
# #                     dft, F_dust  = solve_F_dust(dft)

    dft['F_fines_from_br'] = dft['F_fines_boxmodel'] - dft['F_dust']
    dft['F_dissolved'] = (dft['F_fines_boxmodel'] - dft['F_dust']) * dft['DF']

#                 # These should be equivalent: LHS = RHS of mass balance
    dft['F_br_plus_F_dust'] = dft['F_br'] + dft['F_dust']
    dft['F_coarse_plus_F_fines_plus_F_dissolved']= dft['F_coarse'] + dft['F_fines_boxmodel'] + dft['F_dissolved']
    # dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr']
    dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']  =dft['F_br']  - dft['F_coarse'] - dft['F_fines_boxmodel']
    # fdissimple = dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines'].iloc[0]
    # st.write(f"F (diss, simple) = Fbr - Fc - Ff,{np.round(fdissimple,5)} = {np.round(dft['F_br'].iloc[0], 5)}  - {np.round(dft['F_coarse'].iloc[0],5)} - {np.round(dft['F_fines_boxmodel'].iloc[0], 5)}")
    # st.write(f"F (diss, simple) = {np.round(fdissimple,5)} = {np.round([dft['F_br'].iloc[0]- dft['F_coarse'].iloc[0] - dft['F_fines_boxmodel'].iloc[0]], 5)}")
# # #                     DF = dft['DF']
# # #                     p_re = dft['p_re']

    #########Carbonate mass balance
    # Recalc DF:
    # Fb = Fc + Ffb + Fdis
    # Fb = Fc + Ffb + DF*Ffb
    # We want DF to be true for DF*Ffb = Fdis
    # DF = Fdis/Ffb
    # Fdis_carbbalance
    if selval_dict['model_type'] == "carbbalance":
        dft['F_br_carbbalance']= dft['F_br'] *dft['carb_br']/100
        dft['F_coarse_carbbalance']= dft['F_coarse'] *dft['carb_br']/100
        dft['F_fines_boxmodel_carbbalance']= dft['F_fines_boxmodel'] *dft['carb_br']/100

        dft['F_dis_carbbalance'] = dft['F_br_carbbalance'] -dft['F_coarse_carbbalance']  - dft['F_fines_boxmodel_carbbalance']
        dft['DF_carbbalance'] = dft['F_dis_carbbalance'] / dft['F_fines_boxmodel']
        to_m2_cols = ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust', 'F_fines_from_br','F_dissolved', 'F_br_plus_F_dust', 'F_coarse_plus_F_fines_plus_F_dissolved', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines','F_br_carbbalance','F_coarse_carbbalance','F_fines_boxmodel_carbbalance', 'F_dis_carbbalance']
        st.write(f"DF (Carbonate Balance) is {np.round(dft['DF_carbbalance'].to_numpy(), 1)}")
        rhsmb = dft['F_fines_boxmodel'].to_numpy()+dft['F_coarse'].to_numpy()+ dft['F_dis_carbbalance'].to_numpy()
        st.write(f"RHS is {np.round(rhsmb*1e4, 1) }")
    else:
        # to_m2_cols = [co for co in dft.columns.to_list() if co.startswith('F_')]
        # st.write(to_m2_cols)
        to_m2_cols = ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust', 'F_fines_from_br',
        'F_dissolved', 'F_br_plus_F_dust', 'F_coarse_plus_F_fines_plus_F_dissolved', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']
    #                 # Change fluxes to m2
#                 # g/cm2/yr  * 100*100cm2/m2
    for c,cc in enumerate(to_m2_cols):
        dft[cc + '_g_m2_yr_val'] = dft[cc].apply(lambda x: x*10000).copy()
        # dft[cc + '_g_m2_yr_val'] = dft[cc].apply(lambda x: x*10000).copy()

    dft['rt_ky'] = dft['rt'].copy() /1000 # ky

    return dft, selval_dict

# def simple_carb_recalc(dft, df_chem, F_br,F_c, F_fines,  selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six,fmtcols, carbfmcols,carbfmd):
#     # Recalc DF



def display_massbalance_equations():
    tx = r'''\begin{equation}
        DF = F_{dis} / F_{fb}
        \end{equation}'''
    st.latex(tx)


# Substituting Equation \ref{eq:dissolutionfactorbalance} into Equation \ref{eq:fluxes_balance_nodust} yields:
#         \begin{equation} \label{eq:intermediate_derivation_mass_balance}
#             F_b = F_c + F_{fb}*(1+DF)
#         \end{equation}
#         Rearrange to isolate $F_{fb}$:
#         \begin{equation}
#            \label{eq:intermediate2_derivation_mass_balance}
#             -F_{fb} = \frac{F_c -F_b}{(1+DF)}
#         \end{equation}

#         Add unknown dust flux ($F_d$) to both sides, where $F_{d}$ is the mass flux of aeolian material, of any origin outside the hillslope and rearrange again:

#         \begin{equation} \label{eq:unknown_dust_both_sides}
#         F_d = F_d+ F_{fb} + \frac{F_c -F_b}{(1+DF)}
#         \end{equation}

#         Note that we do not include the dissolution of dust in this model.

#         $F_f$ represents the total flux of fine sediment (<2mm) out of the control volume, which is the sum of fine sediments produced by bedrock and those deposited by dust:

#         \begin{equation} \label{eq:define_Ff}
#             F_f =   F_{d} + F_{fb}
#         \end{equation}

#         Then, substituting Equation \ref{eq:define_Ff} into the unknown dust flux in Equation \ref{eq:unknown_dust_both_sides} :

#         \begin{equation} \label{eq:fluxes_dust_isolated}
#              F_{d}  = F_f + \frac{F_c  -F_b}{(1+DF)}
#         \end{equation}


#         In Equation \ref{eq:fluxes_dust_isolated}, the measurable quantities are isolated on the right side. Equations \ref{eq:fluxes_balance_nodust} and \ref{eq:fluxes_dust_isolated} are represented in Figure \ref{fig:fluxes_balanced_boxes}. The fluxes of fine sediment, coarse sediment, and bedrock can be evaluated through a combination of techniques, briefly described below and in more detail in Sections \ref{sec:Ffo}, \ref{sec:Fb}, and \ref{sec:Fc}.


#         \begin{figure}[htbp!]
#             \centering
#             \includegraphics[width=0.75\linewidth]{src/texts_6_12/June_12_Ch4/Figures/ConceptFigs/Mass_balance_visual_Equations.png}
#             \caption[Conceptual Visualization of Mass Balance of Regolith]{Top: A visual representation of the mass balance between the fluxes in and out of the regolith control volume, Equation \ref{eq:fluxes_balance_nodust}. Lower: The addition of dust means that the proportion of fine sediments derived from bedrock ($F_{bf}$) to dissolved material decreases ($F_{dis}$).}
#             \label{fig:fluxes_balanced_boxes}
#         \end{figure}

#         The mass flux from bedrock is estimated from the \cosmoCl{} in bedrock.  Mass flux (g/cm$^2$/yr) is related to lowering rate of bedrock ($E_b$, L/T, mm/kyr) by the density (g/cm$^3$), and so the flux of material released by the bedrock as it erodes can be calculated.

#         \begin{equation}
#             F_b = E_b*\rho_b
#             \label{eq:F_br}
#         \end{equation}


    # tx = '''Substituting Equation \ref{eq:dissolutionfactorbalance} into Equation \ref{eq:fluxes_balance_nodust} yields:'''
    # st.write(tx)

def write_postscript():
    tx = r'''    F_b = F_c + F_{fb}*(1+DF)'''
    st.latex(tx)

    tx = "Rearrange to isolate $F_{fb}$:"
    st.write(tx)

    tx = r''' -F_{fb} = \frac{F_c -F_b}{(1+DF)}'''
    st.latex(tx)

    tx = "Add unknown dust flux ($F_d$) to both sides, where $F_{d}$ is the mass flux of aeolian material, of any origin outside the hillslope and rearrange again:"
    st.write(tx)

    tx = r''' F_d = F_d+ F_{fb} + \frac{F_c -F_b}{(1+DF)}'''
    st.latex(tx)

    tx = "Note that we do not include the dissolution of dust in this model."
    st.write(tx)
    tx = r"$F_f$ represents the total flux of fine sediment (<2mm) out of the control volume, which is the sum of fine sediments produced by bedrock and those deposited by dust:"
    st.write(tx)

    tx = r'''   F_f =   F_{d} + F_{fb}   '''
    st.latex(tx)

    tx = '''Then, substituting Equation \ref{eq:define_Ff} into the unknown dust flux in Equation \ref{eq:unknown_dust_both_sides} :'''
    st.write(tx)


    tx =r'''F_{d}  = F_f + \frac{F_c  -F_b}{(1+DF)}'''
    st.latex(tx)


    tx = r'''In Equation \ref{eq:fluxes_dust_isolated}, the measurable quantities are isolated on the right side. Equations \ref{eq:fluxes_balance_nodust} and \ref{eq:fluxes_dust_isolated} are represented in Figure \ref{fig:fluxes_balanced_boxes}. The fluxes of fine sediment, coarse sediment, and bedrock can be evaluated through a combination of techniques, briefly described below and in more detail in Sections \ref{sec:Ffo}, \ref{sec:Fb}, and \ref{sec:Fc}.'''
    st.write(tx)

    tx = r'''The mass flux from bedrock is estimated from the \cosmoCl{} in bedrock.  Mass flux (g/cm$^2$/yr) is related to lowering rate of bedrock ($E_b$, L/T, mm/kyr) by the density (g/cm$^3$), and so the flux of material released by the bedrock as it erodes can be calculated.'''
    st.write(tx)

    tx = r'''F_b = E_b*\rho_b'''
    st.latex(tx)

    tx = r'''\begin{equation}
        I_{Be} = \int^z_{z_b} N \rho_{re} dz
    \end{equation}'''
    st.latex(tx)


    tx = r'''where I$_{Be}$ is the total inventory of \Be\ calculated for a given sample (at/cm$^2$), $z$ is the elevation of the surface, and $z_b$ is the elevation of the bedrock-soil interface. \textit{N} is the \Be\ concentration (at/g), $\rho_{re}$ is the depth-averaged regolith density of the sample (g/cm$^3$).

        Residence times assume the soil depth is stable through time and describe how long that sample would need to be exposed to the delivery rate to be in secular equilibrium, given that \Be{} decays and the soil is eroding. Residence times are dependent on the inventory of \Be{}, which in turn is dependent on the soil depth.  Soil residence time is the duration of an average particle of soil  on the hillslope \autocites{Jungers_2009, west_regolith_2013}, and to elucidate the processes involved in moving soil on the hillslope. For example, \textcite{Jungers_2009} identifies the zone where hillslope transport transitions from creep into advective transport by ephemeral channels.
        '''

    st.write(tx)

    tx = r'''\begin{equation}
        t=\left(\frac{-1}{\lambda}\right) ln\left(1-\frac{\lambda I_{{Be}}}{P_{^{10}Be}}\right)
    \end{equation}'''
    st.latex(tx)


    tx = r'''where $t$ is time in years, $\lambda$ is the radioactive decay constant for \Be{} (5.1 x 10$^{-7}$ yr$^{-1}$), $I_{Be}$ is the inventory of atoms of \Be{} in the soil at a site on a ridgetop (at/cm$^2$), and $P_{^{10}Be}$ is the delivery rate of \Be{} from the atmosphere (at/cm$^2$/yr).'''
    st.write(tx)

    tx = r'''\begin{equation}
        E=\frac{D-\lambda I_{Be}}{N_{surface}}    \end{equation}'''
    st.latex(tx)


    tx =r'''where E is erosion rate (cm/yr), $N_{surface}$ is the \Be{} concentration of the eroding material, i.e. the surface soil (at/cm$^3$) \autocite{West_2014}.
    '''

    st.write(tx)

    tx = r'''This erosion rate is independent of the assumptions involved in calculating bedrock erosion rates from \cosmoCl.
    '''
    st.write(tx)

    tx =r'''\Be{} provides an estimate of the residence time ($t_f$) of the fine fraction of regolith, because the accumulation of \Be{} is primarily in the soil, and is eroded with the soil \autocite{Brown_1992}. Combined with observations of soil depth, a simple model of mass flux of fine sediments can be created:'''
    st.write(tx)

    tx = r'''\begin{equation}
        F_f = h_f*\rho_f / t_f
            \end{equation}'''
    st.latex(tx)

    tx = r'''where $h_f$ represents the depth of fine sediment and pore space, measured in the field, $\rho_f$ is the density of the fine fraction of sediment, and $t_f$ is the residence time of the fine fraction.
    '''

    st.write(tx)



