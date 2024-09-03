# MAin page in SoilsMassBalance
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
from io import BytesIO
# from py_functions.load_intro import *
# from py_functions.load_prep_initial_df import *
# from py_functions.load_wrap_plotting_funcs import *
# from py_functions.load_plotting_funcs import *       # map_cdict
# from py_functions.load_plotting_funcs_02_13 import *
# from uncertainties import ufloat_fromstr
st.set_page_config(layout="wide" )


fn = "https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/df_initialize.xlsx"
df = pd.read_excel(fn)
df = df[df['select_col'] != "zcol"].copy()  # not useful


fn = "https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/defaults_Tables.xlsx"
df_default =  pd.read_excel(fn)
# df = df_default.copy()

# toc = stoc()

# Title
st.title("Interactive Soil Mass Balance Models")
st.write("This work is based on the work done for the dissertation: ")

# g/m2/yr?


varnames_dict = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Soil Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction ($F_c$) Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface",
                "z": "Soil Depth (fines)"
                }
varnames_dict2 = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Soil Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms <sup>10<\sup>Be<sub>met<\sub> Delivered to Surface",
                "z": "Soil Depth (fines)"
                }
varunits_dict = {"Coarse_seds_subsurface":"%",
                "DF":"Solid products/Dissolved products",
                "p_re": "g/cm$^3$",
                "br_E_rate": "mm/ky",
                "coarse_mass": "kg",
                "max_coarse_residence_time":"kyr",
                "D":"Atoms/cm$^2$/yr",
                "z": "cm"
                }

                #  vals_arr = [ AZ_D_graly*0.5, AZ_D_graly,AZ_D_graly*1.5,
                #  SP_D_graly*0.5,SP_D_graly*1, SP_D_graly*1.5, SP_D_graly*4]
                #"D": ["0.5 $\cdot$ $D_A_Z$", "$D_A_Z$", "1.5$\cdot$ $D_A_Z$", "0.5 $\cdot$ $D_S_P$", "$D_S_P$", "1.5 $\cdot$ $D_S_P$", "4$\cdot$ $D_S_P$"],
                #
varvalues_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": ["Regional Default", "0.5x $D_{AZ}$", "$D_{AZ}$", "1.5x $D_{AZ}$", "0.5 x $D_{Sp}$", "$D_{Sp}$", "1.5x $D_{Sp}$", "4x $D_{Sp}$"],
                "DF":[2.5, 5, 7.5, 15, 22.5],
                "p_re": [0.1, 0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5, 50, 70],
                "coarse_mass": [.75, 1.5, 2.25],
                "max_coarse_residence_time":[2, 5.5, 11., 16.5],
                "z": ["Site Measurement", 5, 10, 20, 50]
                }

D = 1.8e6 # at/cm2/yr
p_re = 1.4 # g/cm3
lamba = 5.1e-7 # 1/yr
atom10Bemass = 1.6634e-23 #g/at
AZ_D_graly = D_graly(400, 31.2)
SP_D_graly = D_graly(510, 39.1) # used to be 450 mm/yr precip...
#lambda
ltl = 5.1e-7 # , 7.20e-7]  #1/yr lambda [Nishiizumi et al., 2007], Korschinek et al. (2010)

siu = df.sample_id.unique()
selcolu = list(varnames_dict.keys()) # df.select_col.unique()
vars_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": ['default', 0.5*AZ_D_graly, AZ_D_graly, 1.5*AZ_D_graly, 0.5*SP_D_graly, SP_D_graly, 1.5* SP_D_graly, 4*SP_D_graly],
                "DF":[2.5, 5, 7.5, 15, 22.5],
                "p_re": [0.1, 0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5, 50, 70],
                "coarse_mass": [.75e3, 1.5e3, 2.25e3],
                "max_coarse_residence_time":[2e3, 5.5e3, 11.0e3, 16.5e3],
                "z": ['default', 5, 10, 20, 50]
                }
def varvalsfmt(mt):   # functions to provide vals for 'model_type'
    varvalues_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": ["Regional Default", "0.5x $D_{AZ}$", "$D_{AZ}$", "1.5x $D_{AZ}$", "0.5 x $D_{Sp}$", "$D_{Sp}$", "1.5x $D_{Sp}$", "4x $D_{Sp}$"],
                "DF":[2.5, 5, 7.5, 15, 22.5],
                "p_re": [0.1, 0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5, 50, 70],
                "coarse_mass": [.75, 1.5, 2.25],
                "max_coarse_residence_time":[2, 5.5, 11., 16.5],
                "z": ["Site Measurement", 5, 10, 20, 50]
                }
    return varvalues_dict[mt]
selval_dict = {}

# Select Sample Name
# with lc:
default_ix = list(siu).index("NQT0")

# si = st.selectbox("Choose sample: ", siu, index = default_ix,
    # key = keystr, on_change=proc, args = (keystr,))
bc1, lc, rc= st.columns([0.2, 0.3, 0.5])

with lc:
    st.write("Choose samples: ")


    flag_choose_samples_checkbox = False
    si = []
    if flag_choose_samples_checkbox:
        c1, c2,c3, c4, c5 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
        colll = [c1, c2,c3, c4, c5, c1, c2,c3, c4, c5, c1, c2,c3, c4, c5, c1, c2,c3, c4, c5, c1, c2,c3, c4, c5]
        count = 0
        for ixsi, samp in enumerate(siu):
            tfsamp = False

            if samp == "NQT0": tfsamp = True
            if samp == "MT120": tfsamp = True
            keystr = "sample_id_selbox_" + samp
            with colll[count]:
                sitemp = st.checkbox(samp, value = tfsamp, key = keystr, on_change=proc, args = (keystr,))
                count +=1
            if sitemp:
                si.append(samp)
    else:
        keystr = "sample_id_selbox"

        si = st.multiselect(" ", siu, default = ["NQT0", "MT120"], key = keystr, on_change=proc, args = (keystr,))

selval_dict['sample_id'] = si
# dft = df[df['sample_id'] == si].copy()
# dft = df[df['sample_id'].isin(si)].copy()

# dfdsi = df_default[df_default['sample_id']== si].copy()
# default_cols = dfdsi.columns.to_list()

# default_dict = {c:dfdsi[c] for c in default_cols}
# Select model type (Simple mass balance  (solve for dissolution, no dust) + Compare with calcite mass balance
#       or with dust  (Dissolution constrained by calcite mass balance) )

with rc:
    keystr = "model_type_radio"
    model_type = st.radio("Model Type: ", ['simple', 'wdust'], format_func = mtfmt, key = keystr,
        on_change=proc, args = (keystr,))

    selval_dict['model_type'] = model_type

    st.write("Model A: Simple Mass Balance Model")
    st.latex(r"F_b = F_c + F_f + F_{dis}")
    st.write(" ")
    st.write("Model B: Mass Balance Including Dust")
    st.write("Dissolved flux is constrained by calcite mass balance.")
    st.latex(r"F_b + F_d = F_c + F_{f,b} + F_{dis} + F_d")


if model_type == 'simple':
    fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,
        'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
    ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
    ftexp = ['Bedrock', 'Coarse Sediment', 'Fine Sediment', 'Dissolved Material']

else:
    fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
         'F_coarse_g_m2_yr' ,
         'F_fines_from_br_g_m2_yr' ,
         'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
    ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']
    ftexp = ['Bedrock','Dust', 'Coarse Sediment', 'Fine Sediment (originating from bedrock)',
            'Dissolved Material', 'Dust (Fine sediment originating from dust)']
fmtcols = ['br_E_rate','F_br', 'F_br_g_m2_yr']

def sliderrange(start, step, num):
    return start + np.arange(num)*step
# Instructions & EXplanations:





if st.checkbox("Continue?"):
    # Scenario and values:
    baseline_dict = {}
    # units_dict = {}  #
    count = 0
    if len(si) == 3:
        coL, coM, coR = st.columns([0.33, 0.33, 0.33])
        colll = [ coL, coM, coR,  coL, coM, coR]
    elif len(si) == 2:
        coL, coR = st.columns([0.5,  0.5])
        colll = [ coL,  coR,  coL, coR, coL,  coR,  coL, coR]
    elif len(si) == 1:
        coL = st.container()
        colll = [ coL]

    count = 0
    for six, samp in enumerate(si):
        dft = df[df['sample_id']== samp].copy()
        # FIND DEFAULT ROW:
        # Get values from dft, overwrite later
        # dft = dft[dft['select_col'] == 'Coarse_seds_subsurface'].copy()
        # dft = dft[dft["select_col_val"]==0].copy()
        dft['Coarse_seds_subsurface'] = 0
        dft['z'] = dft['z_val'].copy()
        dft['p_re'] = dft['p_re_val'].copy()
        dft['p_br'] = dft['p_br_val'].copy()
        dft['N'] = dft['N_val'].copy()
        dft['coarse_mass'] = dft['coarse_mass_val'].copy()
        dft['coarse_area'] = dft['coarse_area_val'].copy()
        dft['br_E_rate'] = dft['br_E_rate_val'].copy()
        dft['Inv'] = dft['Inv_val'].copy()
        dft['D'] = dft['D_val'].copy()

        # st.write("1, ")
        # st.dataframe( dft["D"])
        troubleshoot = True

        with colll[count]:
            with st.expander(f"Sample {samp}", expanded = True):
                st.write("Changes to the input variables will be incorporated to the plots.")

                lc, rc = st.columns([0.5, 0.5])
                # colll = [lc,  lc, lc, lc, lc, lc,  rc, lc, lc, lc, lc, lc,  lc, lc, lc, lc, rc,lc,  rc, lc, rc, lc, rc]

                # for k, vlist in vars_dict.items():
                #     vld = []
                #     # vars_itemfmt_dict[k] = {}
                #     tempd = {}
                #     for j, sv in enumerate(vlist):
                #         valid_list = varvalues_dict[k]
                #         tempd[sv] = valid_list[j]
                #     vars_itemfmt_dict[sc]= tempd
                #     vld.append(tempd)
                expb = st.checkbox("More Explanation", key = "explaincheckbox_" + str(samp))
                # st.write("1   ", dft[fmtcols].iloc[0])
                with lc:
                    if expb: st.write("What if regolith contains coarse sediments? (Only coarse sediments were measured, this variable is likely NOT zero.)")
                    selcolkey = "Coarse_seds_subsurface"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six)

                    if expb:
                        # Add note defining DAz etc556495.6872
                        st.write("Meteoric $^{10}$Be delivery rates (D) are site-specific. Graly et al 2010 provides an equation, which yields: ")
                        st.write("$D_{AZ}$ = 5.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")
                        st.write("$D_{SP}$ = 9.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")

                    selcolkey = "D"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six)

                    if expb: st.write("The mass of coarse sediment measured on the surface.")
                    selcolkey = "coarse_mass"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six)

                    if expb: st.write("The maximum additional exposure time coarse sediments could endure while sharing the erosion history of the bedrock.")
                    selcolkey = "max_coarse_residence_time"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six, index =1)


                with rc:
                    if expb: st.write("What if the measured soil depth is an underestimate? Explore: ")
                    selcolkey = "z"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six)

                    if expb: st.write("Dust is assumed to have the same density as soil.")
                    selcolkey = "p_re"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six)

                    if expb: st.write("Bedrock erosion rate and flux of bedrock are directly related by the density of the material.")
                    selcolkey = "br_E_rate"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six, index =1)

                    if expb: st.write("Dissolution Factor is the mass ratio of dissolved bedrock and solid products of chemical erosion. DF is in the denominator, and so only increments below 7 seem important.")
                    selcolkey = "DF"
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six, index =2)

                # for fix, selcolkey in enumerate(selcolu):
                    # filtselcol = st.selectbox("Select Input Variable to Explore:",
                    #   [varnames_dict2[s] for s in selcolu], key = "select_filter_col_"+ samp)

                    # st.write("5, ")

                    # st.dataframe( dft["D"])

                #### Recalc
                # st.write('dft[D]',dft['D'].iloc[0])
                # st.write('type dft[D]',type(dft['D'].iloc[0]))
                # st.write('dft[Inv]',dft['Inv'].iloc[0], type(dft['Inv'].iloc[0]))
                # dff = write_defaults_to_df2(df)
                # dft, SD, N = set_up_df_for_flux_results3(dft,dff)  # calc inventory (depends on z)

                dft, selval_dict = simple_recalc(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six,fmtcols)
                # dftt = dft.copy()
                # st.write("2   ", dft[fmtcols].iloc[0])

                # st.write(' ')

                with st.popover(f"Plot dimension options"):

                    # Select box model shape:
                    keystr = "model_shape_radio"

                    model_shape = st.radio("Box shapes: ", ["Uniform height", "Squares", 1.,  5.], index = 1,
                        key = keystr, on_change=proc, args = (keystr,), horizontal = True)
                    selval_dict['model_shape'] = model_shape
                    # st.write(sliderrange(5, 2, 12))
                    hh = sliderrange(5, 2, 12)
                    keystr = "figwidth_radio" + str(samp)
                    selval_dict['figwidth'] = st.select_slider("Scale figure width: ", options = hh, value = 7,key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
                    keystr = "figheight_radio"+ str(samp)
                    selval_dict['figheight']  = st.select_slider("Scale figure height: ",  sliderrange(1, 1,7), value = 3,
                        key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
                    # height
                    keystr = "pixelwidth_radio"+ str(samp)

                    selval_dict["pixelwidth"] = st.select_slider("Scale width of plot in pixels: ",  sliderrange(500, 50, 12), value = 650,
                        key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
                     # Width in px of image produced...
                    keystr = "boxscale_radio"+ str(samp)

                    selval_dict["boxscale"] = st.select_slider("Scale boxes within plot: ",  sliderrange(0.8, 0.2, 12), value = 1,
                        key = keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width
                    keystr = "shape_buffer_radio"+ str(samp)

                    selval_dict["shape_buffer"] =st.select_slider("Scale space between boxes within plot: ", sliderrange(0.5, 0.25, 12),
                         value = 1, key =keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width

                # fcc = [f + str(dft[f].iloc[0]) for f in fmcols] + fmtcols
                # for f in fcc:
                #     st.write(f)
                # st.write(selval_dict)
                fig = wrap_flux_box_streamlit(dft, selval_dict)


                for i, f in enumerate(fmcols):
                    dft[ft[i]] = dft[f].copy()
                # dftt['Sample ID'] = dftt['sample_id']
                # st.write(dftt.columns.to_list())
                # st.write(dftt[ft])
                for i in range(len(ft)):
                    st.write(f'''{ftexp[i]} Flux''')
                    st.write(f"{ft[i]}:   {np.round(dft[ft[i]].to_numpy()[0], 1)} g/m$^2$/yr")
                # st.dataframe(dft[ ft])
                count +=1

with st.expander("Display Mass Balance Equations for F$_b$, F$_d$, and F$_f$"):
    display_massbalance_equations()
# # Add default values
st.dataframe(df_default)
df_defaultcols = df_default.columns.to_list()
# for i in range(len(df_default)):
    # st.write(f"{df_defaultcols[i]} {df_default[df_defaultcols[i]].to_}")
