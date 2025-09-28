## Functions tailored to Soils Mass Balance Web App (poster version)

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


def run_mkdn_style():
    st.markdown("""
        <style>
        
               /* Remove blank space at top and bottom */ 
               .block-container {
                   padding-top: 3rem;
                   padding-bottom: 0rem;
                   padding-left: 3rem;
                   padding-right: 3rem;
                }
               
                /* moves menu up a little */
                [data-testid="stSidebarNavItems"] {
                    padding-top: 3rem;
                }

                /* make sidebar narrower */
                [data-testid="stSidebar"] {
                    min-width: 174px;
                    width: 180px;
                    max-width: 180px;
                }

                /* make sidebar narrower */
                .stSidebar {
                    min-width: 174px;
                    max-width: 180px;
                    width: 200px;
                }

                /* Emphasize Expander Headers */
                .streamlit-expanderHeader {
                    background-color: grey;
                    color: black; 
                }
        </style>
        """, unsafe_allow_html=True)


def vcols(listofcols):
    return [c + '_val' for c in listofcols]


def proc(key):
    None
    #st.info(st.session_state[key])

def sliderrange(start, step, num):
    return start + np.arange(num)*step

def wrap_flux_box_streamlit(dft, selval_dict):
    fig, ax = plt.subplots()
    
    list_of_tuplelists,ft,fst, height , L, H, XY, fst, YC = make_into_area_streamlit(dft, selval_dict)
    plot_patches(list_of_tuplelists, selval_dict,dft, ft, L, H, XY,YC, fst, newfig = False,flag_annot = False, flag_sample_label_default = selval_dict["flag_sample_label_default"], medfont = selval_dict["medfont"], textht = selval_dict["textheight"])
    fig.set_size_inches(selval_dict['figwidth'], selval_dict['figheight'])

    fn = r"/mount/src/soilsmassbalance/data_sources/temp_flux_img.svg"
    fig.savefig(fn, format="svg",pad_inches = 0, transparent=True)
    st.image(fn, width = selval_dict["pixelwidth"])
    
    return fig


def deal_w_ustring(val):
    if isinstance(val, str):
        numstr = ufloat_fromstr(val)
    else:
        numstr = val
    return numstr

def make_into_area_streamlit(df,selval_dict):
    scale = selval_dict["boxscale"]
    shape_buffer = selval_dict["shape_buffer"]
    height = selval_dict['boxheight']
    flag_model = selval_dict['model_type']
    flag_model_style = selval_dict['model_shape']

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
    csum = 0
    fst = []
    Fbr_L = df[fmcols[0]].to_numpy()[0]   # BR value
    XC = [csum]


    for i, col in enumerate(fmcols[0:]):
        colval = df[col].to_numpy()[0]
        colval = deal_w_ustring(colval)
        if colval > 0:
            if flag_model_style == "Uniform height":
                htt = height
                L1 = colval/htt*scale   # maintains area == fluxes
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
   
        list_of_tuplelists.append([DL] + [UL] + [UR]+[DR] +[DL])
    return list_of_tuplelists, ft, fst, height, L, H, XC, fst, YC


def plot_carb_pct(df,selval_dict, collist = ['C_br', 'C_c', 'C_f', 'C_dust'],labellist = ['Bedrock Composition','Coarse Sediment Composition','Fine Sediment Composition','Dust Composition'],ft = ['F$_b$', 'F$_c$','F$_f$','F$_{dust}$'], ec = 'k', six = 0):
    # Plot interactive carbonate composition boxes. 
    for i, colname in enumerate(collist):
        labelname = labellist[i]
        # with lc:
        st.write(f"**{labelname}** ({ft[i]})  \n  \t   Percent Carbonate:")
        pct_carb = st.slider("Change Percent Carbonate:", min_value = 0., max_value = 100.0, value = float(df[colname].iloc[0]), format = "%0.1f", key = "slider_pct_carb" + colname + str(six), label_visibility = "collapsed")

        fig, ax = plt.subplots()

        # ax = axs[i]
        plt.sca(ax)
        Cv = pct_carb/100
   
        colors = ['grey', 'indianred']
        x = [0,0,Cv,Cv,0]
        x2 = [Cv,Cv,1,1,Cv]
        y = [0,1,1,0,0]
        pp = list(zip(x, y))
        pp2 = list(zip(x2, y))
        #### Soluble     hatch = ['x'],
        ax.add_patch(mpatches.Polygon(pp, ec = ec, fc = colors[0],  ls = '-', lw = .5))
        #### Insoluble
        ax.add_patch(mpatches.Polygon(pp2, ec = ec, fc = colors[1], ls = '-', lw = .5))

        ### LAbel
        # matplotlib._mathtext.SHRINK_FACTOR = 0.5
        plt.annotate(f"{ft[i]}", [0.5,1.05], va = 'bottom', fontsize = 10, ha = 'center')
        plt.annotate("Soluble\n{:0.1f}%".format(Cv*100), [0,1.05], va = 'bottom', fontsize = 8, ha = 'left')
        plt.annotate("Insoluble\n{:0.1f}%".format((1-Cv)*100), [.95,1.05], va = 'bottom', fontsize = 8, ha = 'left')
    # with rc2: 
        # for i in range(len(collist)):
        frame1 = ax #axs[i]
        plt.sca(frame1)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.4)

        # fig.set_facecolor('grey')
        frame1.axis('off')
        
        fig.set_size_inches(3,1)

        # fig.set_layout('constrained')
        fig.tight_layout()


        fn = r"/mount/src/soilsmassbalance/data_sources/temp_composition_img.svg"
        fig.savefig(fn, format="svg",pad_inches = 0, bbox_inches='tight',transparent=True)
        st.image(fn, width = 280)

        # buf = BytesIO()
        # fig.savefig(buf, format="png")
        # st.image(buf, width = 350)
        st.divider()
        df[colname] = pct_carb
        selval_dict[colname] = pct_carb
    df['DF'] = df['C_br']/(1-df['C_br'])   # DF is soluble/insoluble fractions of the bedrock... #Needed update here 9/28/25
    return df,selval_dict


def plot_patches(list_of_tuplelist, selval_dict,df, ft, L, H, XC, YC, fst,add_conc = 'auto', newfig = True, flag_annot = True, set_maxy = None, xoffset = 0, flag_sample_label_default = True, medfont = 12, textht = 3):
    flag_model_style = selval_dict['model_shape']
    height = selval_dict['boxheight']
    flag_model = selval_dict['model_type']
    if newfig:
        fig, ax = plt.subplots()
    else:
        ax = plt.gca()
        fig = plt.gcf()
    equals_locx = 0

    if flag_model_style == "squares":

        maxy = np.max(H)
        midy_patch = maxy/2
        # npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  npn[1]+ midy_patch*1.9 ) # Find x and y-midpoint
    else:
        maxy = 1 #H[0]   # bedrock height
        midy_patch = np.max(H)/2

        # npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  2) # Find x and y-midpoint
    maxx = XC[-1]
    suby = -0.5

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
    if isinstance(flag_sample_label_default, bool):    
        sample_label = df.sample_id.iloc[0] +"\n" + df.sample_region.iloc[0]
    else:
        sample_label = flag_sample_label_default+"\n(g/m$^2$/yr)"

    ## Make subscripts smaller??
    matplotlib._mathtext.SHRINK_FACTOR = 0.5

    mxx = []
    for i, points in enumerate(list_of_tuplelist):
        flag_along_baseline = False
        if flag_along_baseline: pass
        else:
            adjx = [points[p][0] + xoffset for p in np.arange(len(points))]
            y = [points[p][1] for p in np.arange(len(points))]
            npp = list(zip(adjx, y))
            #### ADD PATCH
            ax.add_patch(mpatches.Polygon(npp, ec = ec, fc = bxc[i], hatch = hch[i], ls = '-', lw = .5))

            midx = npp[0][0] + (npp[3][0]-npp[0][0])/2
            # npn = ( (npp[0][1]+npp[1][1])/2 ,  (npp[1][0] + npp[0][0])/2 )  # Find x and y-midpoint
            # npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2, midy_patch )  # Find x and y-midpoint
            
            if flag_model == 'simple':
                pct_denom = fst[0]  # just bedrock flux
            elif flag_model == 'carbbalance':
                pct_denom = fst[0]  # just bedrock flux
            else:
                pct_denom = fst[0] + fst[1] # bedrock + dust flux
            fst_as_pct = np.round(fst[i]/pct_denom*100, 0)

            # midpt of space between this patch and previous
            spacex = (npp[0][0] - (list_of_tuplelist[i-1][3][0] + xoffset))/2
            midspacex = npp[1][0]-spacex
            

            sll_offset = 0.4
            sample_label_loc = (npp[1][0]-spacex, textht+sll_offset)
            if flag_model_style == "squares":
                npn = (midx,  npn[1]+ midy_patch*1.9 ) # Find x and y-midpoint
                npsym = (midspacex,   npn[1]+ midy_patch*1.9)
            else:
                npn = (midx, textht) # Find x and y-midpoint
                npsym = (midspacex,textht)
            
            ## ANNOT: F_br etc
            lgfont = 18 
            plt.annotate(ft[i], npn, va = 'center', fontsize = medfont + 6, ha = 'center')
            ## ANNOT: AMOUNT (g/m2/yr)
            if i == 0:
                if isinstance(flag_sample_label_default, bool):
                    plt.annotate('\n {:0.1f}\n g/m$^2$/yr \n {:0.0f}%'.format(fst[i], fst_as_pct), npn, fontsize = medfont,  va = 'top', ha = 'center')
                else:
                    plt.annotate('\n {:0.1f}\n {:0.0f}%'.format(fst[i], fst_as_pct), npn, fontsize = medfont,  va = 'top', ha = 'center')
            else:
                plt.annotate('\n {:0.1f}\n {:0.0f}%'.format(fst[i], fst_as_pct), npn,fontsize = medfont, va = 'top', ha = 'center')

            # npn2 = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  (npp[0][1]+npp[1][1])/2 ) # Find x and y-midpoint
            # npn2 = (midx,  textht) # Find x midpoint and y below patch

            ## ANNOT: PCT
            # plt.annotate('{:0.0f}%'.format(fst_as_pct), npn2, va = 'center', ha = 'center', fontsize = medfont ) # , fontweight = "bold")
            # plt.annotate(f"LxH = Area\n{L[i]} x {H[i]} \n\t= {fst[i]}", (points[0][0], 0.1), va = "center", rotation = 20)
            # Add equation stuff to nearby box
            if i>0:
                if flag_model == 'simple':
                    if i == 1:
                        # Write sample name above equals sign: 
                        plt.annotate(sample_label, sample_label_loc, fontsize = lgfont, ha = "center") 
                    syms = [' ', '=', '+', '+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (midspacex, midy_patch),ha='center', va = 'center', fontsize = medfont)
                    # (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 )
                elif flag_model == 'carbbalance':
                    if i == 2:
                        # Write sample name above equals sign: 
                        plt.annotate(sample_label, sample_label_loc, fontsize = lgfont, ha = "center")  
    #                 print(i, points)
                    syms = [' ', '=', '+', '+', ' ']
                    sy = syms[i]
                    plt.annotate(sy,(midspacex, midy_patch),ha='center', va = 'center', fontsize = medfont)
                     # (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 )
                else:

                    # if i == 2:
                    #     # Write sample name above equals sign: 
                    #     plt.annotate(sample_label, sample_label_loc, fontsize = 13, ha = "center")
                    syms = [' ','+', '=', '+', '+','+', ' ']
                    sy = syms[i]
                    plt.annotate(sy,(midspacex, midy_patch),ha='center', va = 'center', fontsize = medfont)
                     # (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 )

                # Also write between F labels
                # if not flag_annot:
                    # plt.annotate(sy, npsym,ha='center', va = 'center', fontsize = medfont)

        mxx.append(adjx)
    maxx2 = np.max(np.array(mxx))
    frame1 = plt.gca()

    xl = maxx2+2
    yl = textht + sll_offset+0.1
    plt.xlim(-0.001, xl)
    plt.ylim(-0.01, yl )

    # add height annotation if uniform height.
    if flag_model_style != "squares":
    #     astyle = f']-, widthA={height}, lengthA=0.1'  # arrowprops=dict(arrowstyle=astyle, lw=1.0, color='k'),
        ax.annotate('height is {:0.1f} units'.format(height), xy=((xl-0.75), height/2), xytext=((xl-0.75), height/2), xycoords='data', fontsize=8, ha='center', va='center', rotation = 90)

    # fig.set_facecolor('red')
    frame1.axis('off')
    # fig.set_layout('constrained')
    fig.tight_layout()  # Why does this produce erros?
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

    # N is conc Be in at/g
    # p_re is density of soil g/cm3
    # z is soil depth (cm)
    # Inv is at/cm2 = at/g * g/cm3 * cm
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
# (x-x0) is dx here
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
    # Assumes that F_dust is ALL non carb and F_fines is ALL non carb
    F_dust = F_fines + (F_coarse - F_br)/ (1+ DF)
    return F_dust

def x_noncarb_fines(CaO_wt_pct_fraction_fines, Loss_on_ignition_fines):
    # Xf = LOI-CO2 mass percent using CaO_wt_pct_fraction_fines* CALCITESTOICHIOMETRY
    Calcite_wt_pct = CaO_wt_pct_fraction_fines *(100/56)  # Check stoichiometry
    CO2_from_calcite = Calcite_wt_pct * 44/100
    OM_fines = Loss_on_ignition_fines - CO2_from_calcite # Get CO2 loss that can't be attributed to Calcite
    X_fines = 1-calcite_wt_pct/100 - OM_fines/100    # just assume that is all OM
    return X_fines  

def f_noncarb_mass_balance_for_dust(F_fines, F_coarse, F_br, X_coarse,X_fines, X_br, X_dust):
    # X_ need to be fractions <1
    # F can be any unit.
    F_dust = (X_coarse*F_coarse + X_fines*F_fines -X_br*F_br)/X_dust
    return F_dust 

def f_f_from_br(dft):
    F_fines_from_br = dft['F_fines_boxmodel'] - dft['F_dust']
    return F_fines_from_br

def f_diss_from_other_F(F_fines, F_coarse, F_br, F_dust):
    # F_dissolved
    return F_br +F_dust - F_fines - F_coarse

def f_noncarb_mass_balance_for_diss(F_fines, F_coarse, F_br, X_coarse,X_fines, X_br, X_dust):
    # X_ need to be fractions <1
    # F can be any unit.
    F_diss = (X_coarse*F_coarse + X_fines*F_fines -X_br*F_br)/X_dust  - F_fines - F_coarse + F_br  
    return F_diss  

def f_dust_from_other_F(F_fines, F_coarse, F_br, F_diss):
    # F_dust
    return F_fines + F_coarse+ F_diss - F_br 


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

def D_graly(P,L, flag_P_is_mm_per_yr = True):
    # Graly 2010a
    # latitude = L
    # P = precip in cm/yr
    # flux in 10^4 at/cm3  but elsewhere in the paper it's written at/cm2/yr....
    if flag_P_is_mm_per_yr:
        P = P/10
    flux = P*(1.44/(1+np.exp((30.7-L)/4.36))+0.63)*10**4
    return flux


def Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six, expb_d, varunits_dict, index = 0, fmtfc = '%0.1f'):
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


    # st.write(expb_d[selcolkey])
    # st.write(varunits_dict[selcolkey])
    keystr = str(selcolkey) + "_radioval_"+ str(six)

    vvd =varvalues_dict[selcolkey]
    font_size = 20

    html_str = f"""
        <style>
        p.a {{
          font: bold {font_size}px Times New Roman;
        }}
        </style>
        <p class="a">{varnames_dict2[selcolkey]} </p>
        """
    rmd = f"**{varnames_dict2[selcolkey]}**"
    # st.markdown(html_str, unsafe_allow_html=True)
    st.write(rmd)

    val = st.radio( f"{expb_d[selcolkey]} ({varunits_dict[selcolkey]})", vvd, index = index,
        key = keystr, on_change=proc, args = (keystr,), horizontal = True)
    # st.write("3, ")
    # st.dataframe( dft["D"])
    if selcolkey == "D":
        st.write("Graly et al. (2010) provides an equation which scales delivery with a site's latitude and precipitation, which yields:")
        st.write("   $D_{AZ}$ = 5.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")
        st.write("   $D_{SP}$ = 9.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")
    elif selcolkey == 'z':
        with st.popover("How is the depth of the fine fraction of mobile regolith included in mass flux equations?"):
            bems = f"$^{{10}}$Be$_m$"
            st.write(f"The depth of the fine fraction of mobile regolith ($z$) is used to calculate the inventory of {bems}. We assume here that the concentration of {bems} ($N$) is constant with depth. Higher inventories ($I$) of  {bems} indicate longer residence times. ")
            st.latex(r"I = \int_z N \rho_{re} dz")
            st.write(f"Inventory is used to calculate the residence time (t) of the fine fraction: ")
            st.latex(r"t = \left(\frac{-1}{\lambda} \right) ln\left(1-\frac{ \lambda I_{Be}}{P_{Be}}\right)")
            st.write(f"where $\lambda$ is the radioactive decay constant for {bems} (5.1 x 10$^{{-7}}$ yr$^{{-1}}$), and $P_{{Be}}$ is the delivery rate of {bems} from the atmosphere (at/cm$^2$/yr).")
            st.write(f"Regolith depth is also used when calculating the mass flux ($F_f$): ")
            st.latex(r"F_f = z\rho_{re}/t")
            st.write(f"where z is soil depth, $\rho_{{re}}$ is bulk density of the regolith (\"fine\" fraction).  ")
            st.write(f"In short, z controls $F_f$ in a complex way, which can be seen by substituting the equations for I and t into the mass flux equation: ")
            st.latex(r"F_f = z\rho_{re}/\left[ \left(\frac{-1}{\lambda} \right) ln\left(1-\frac{\lambda z N \rho_{re}}{P_{Be}}\right) \right]")
            st.write(f"Simplified, with constants removed:")
            st.latex(r"F_f = \frac{z}{-ln(1-z)}")
            # st.latex(r"F_f = z/ C\left(\frac{-1}{\lambda} \right) ln\left(1-\frac{\lambda z N \rho_{re}}{P_{Be}}\right)")
    elif selcolkey == "coarse_mass":
        valarea = int(np.round(vars_dict['coarse_area'][0],0))
        st.write(f"Note: coarse mass is mass per collection area ({valarea} cm$^2$)")
    elif selcolkey == "N":
        st.write(f"Semi-Arid site measured value ($N_{{SP}}$): {selval_dict["N_SP"]:0.2e} at/g")
        st.write(f"Arid site measured value ($N_{{AZ}}$): {selval_dict["N_AZ"]:0.2e} at/g")

    st.divider()
    # st.write(" " )

    # Assign un-formatted choice to selval_dict "selected values dictionary"
    v2vdt = {vvd[ii]:vars_dict[selcolkey][ii] for ii in range(len(vvd))}

    selval_dict[selcolkey] = v2vdt[val]

    dft[selcolkey] = v2vdt[val]

    return dft, selval_dict

def partial_recalc(dft, selval_dict, ltl = 5.1e-7):
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
    v2 = dft['rt']# is this supposed to be in *yrs* -- yes bc D is yrs
    v3 = dft['p_re']
    # SD, res_t, ps)
    dft['F_fines_boxmodel'] =  flux_boxmodel(v1, v2, v3)

    v1 = dft['coarse_mass']
    v2 = dft['coarse_area']
    v3 = dft['max_coarse_residence_time']

    dft['F_coarse'] = f_coarse_flux(v1, v2, v3)

    v1 = dft['br_E_rate']
    v2 = dft['p_br']
    # st.write("before ", v1, v2, f_br_flux(v1, v2) )

    dft['F_br'] = f_br_flux(v1, v2)
    return dft, selval_dict

def simple_recalc(dft, selval_dict, ltl = 5.1e-7):
    dft, selval_dict = partial_recalc(dft, selval_dict, ltl = ltl)

    v1 = dft['F_fines_boxmodel']
    v2 = dft['F_coarse']
    v3 = dft['F_br']
    v4 = dft['DF']
    dft['F_dust'] =  f_mass_balance_for_dust(v1, v2, v3, v4)
    # st.write("3   ", dft[fmtcols].iloc[0])

    dft['F_fines_from_br'] = dft['F_fines_boxmodel'] - dft['F_dust']
    dft['F_dissolved'] = (dft['F_fines_boxmodel'] - dft['F_dust']) * dft['DF']

    dft  = modify_F_units(dft, to_m2_cols = ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust','F_dissolved', 'F_fines_from_br'])
    return dft, selval_dict

def modify_F_units(dft, to_m2_cols = ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust','F_dissolved']):
    #                 # Change fluxes to m2
    #                 # g/cm2/yr  * 100*100cm2/m2
    for c,cc in enumerate(to_m2_cols):
        dft[cc + '_g_m2_yr_val'] = dft[cc].apply(lambda x: x*10000).copy()
    dft['rt_ky'] = dft['rt'].copy() /1000 # ky
    return dft

def wrap_dual_eq_solve(CaO_wt_pct_fraction_fines, Loss_on_ignition_fines,F_fines, F_coarse, F_br, X_coarse,X_fines, X_br, X_dust ):
    X_fines = x_noncarb_fines(CaO_wt_pct_fraction_fines, Loss_on_ignition_fines)
    F_dust =  f_noncarb_mass_balance_for_dust(F_fines, F_coarse, F_br, X_coarse,X_fines, X_br, X_dust)

    F_diss =  f_diss_from_other_F(F_fines, F_coarse, F_br, F_dust)
    return X_fines, F_dust, F_diss

def display_massbalance_equations():
    tx = r'''\begin{equation}
        DF = F_{dis} / F_{fb}
        \end{equation}'''
    st.latex(tx)


def get_X_vals(dft, list_of_carbcols = ['C_br', 'C_c', 'C_f', 'C_dust']):
    # from percent carbs, returns fraction (of 1) of insoluble material.
    for i, col in enumerate(list_of_carbcols):
        dft["X_" + col[2:]] = (100 - dft[col])/100

    X_c = dft['X_c'].astype(float).values[0]
    X_f = dft['X_f'].astype(float).values[0]
    X_br = dft['X_br'].astype(float).values[0]
    X_dust = dft['X_dust'].astype(float).values[0]
    return dft, X_c, X_f, X_br, X_dust


def add_val_report(dft, user_option_keys, selval_dict):
    for col in user_option_keys:
        st.write(col, dft[col].astype(float).values[0])
    
    st.write("Mass balance")
    st.write(f"F_br+ F_dust = F_fines + F_coarse + F_diss")

    # st.write(dft['F_br_g_m2_yr_val'].values[0],dft['F_dust_g_m2_yr_val'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['F_coarse_g_m2_yr_val'].values[0], dft['F_dissolved_g_m2_yr_val'].values[0])
    st.write("{:0.2f} + {:0.2f} = {:0.2f} + {:0.2f} +{:0.2f}".format(dft['F_br_g_m2_yr_val'].values[0],dft['F_dust_g_m2_yr_val'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['F_coarse_g_m2_yr_val'].values[0], dft['F_dissolved_g_m2_yr_val'].values[0]))

    st.write("Non-carb mass balance")
    st.write("F_dissolved  = (X_f F_f + X_c F_c - X_br F_br)/X_dust - F_f - F_c + F_br")
    # st.write(dft['X_f'])
    # st.write(dft['F_dissolved_g_m2_yr_val'].values[0],dft['X_f'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['X_c'].values[0],dft['F_coarse_g_m2_yr_val'].values[0],dft['X_br'].values[0],dft['F_br_g_m2_yr_val'].values[0],dft['X_dust'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['F_coarse_g_m2_yr_val'].values[0],dft['F_br_g_m2_yr_val'].values[0])

    st.write("{:0.2f}  = ({:0.2f} {:0.2f} + {:0.2f} {:0.2f} - {:0.2f} {:0.2f})/{:0.2f} - {:0.2f} - {:0.2f} + {:0.2f}".format(dft['F_dissolved_g_m2_yr_val'].values[0],dft['X_f'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['X_c'].values[0],dft['F_coarse_g_m2_yr_val'].values[0],dft['X_br'].values[0],dft['F_br_g_m2_yr_val'].values[0],dft['X_dust'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['F_coarse_g_m2_yr_val'].values[0],dft['F_br_g_m2_yr_val'].values[0]))

    st.write("{:0.2f}  = ({:0.2f}+ {:0.2f}  - {:0.2f} )/{:0.2f} - {:0.2f} - {:0.2f} + {:0.2f}".format(dft['F_dissolved_g_m2_yr_val'].values[0],dft['X_f'].values[0]*dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['X_c'].values[0]*dft['F_coarse_g_m2_yr_val'].values[0],dft['X_br'].values[0]*dft['F_br_g_m2_yr_val'].values[0],dft['X_dust'].values[0],dft['F_fines_boxmodel_g_m2_yr_val'].values[0],dft['F_coarse_g_m2_yr_val'].values[0],dft['F_br_g_m2_yr_val'].values[0]))

    eq1 = "f_mass_balance_for_dust(F_fines, F_coarse, F_br, DF)\nF_dust = F_fines + (F_coarse - F_br)/ (1+ DF)"

    eq2 = "f_noncarb_mass_balance_for_dust(F_fines, F_coarse, F_br, X_coarse,X_fines, X_br, X_dust)\n    F_dust = (X_coarse*F_coarse + X_fines*F_fines -X_br*F_br)/X_dust"

    ffeq = "f_f_from_br(dft): F_fines_from_br = dft['F_fines_boxmodel'] - dft['F_dust']"

    fdis = "f_diss_from_other_F(F_fines, F_coarse, F_br, F_dust)\n  F_br +F_dust - F_fines - F_coarse"

    st.write('Ratio of bedrock that is mechanical wx: {:0.2f}'.format(float(dft['F_coarse_g_m2_yr_val'])/float(dft['F_br_g_m2_yr_val'])))
    fmcols = selval_dict['fmcols']

    ft = selval_dict['ft']
    ftexp = selval_dict['ftexp']

    for i, f in enumerate(fmcols):
        dft[ft[i]] = dft[f].copy()

    for i in range(len(ft)):
        st.write(f'''{ftexp[i]} Flux''')
        st.write(f"{ft[i]}:   {np.round(dft[ft[i]].to_numpy()[0], 1)} g/m$^2$/yr")
    # st.dataframe(dft[ ft])
