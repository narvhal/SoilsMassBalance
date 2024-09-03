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
st.title("Interactive Soil Mass Balance Plots")
st.header("Flux boxes")

# g/m2/yr?


varnames_dict = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Soil Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction ($F_c$) Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface"
                }
varnames_dict2 = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Soil Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction ($F_c$) Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface"
                }
varunits_dict = {"Coarse_seds_subsurface":"(%)",
                "DF":"(Solid products/Dissolved products)",
                "p_re": "(g/cm$^3$)",
                "br_E_rate": "(mm/ky)",
                "coarse_mass": "(kg)",
                "max_coarse_residence_time":"(kyr)",
                "D":"Atoms/cm$^2$/yr"
                }

                #  vals_arr = [ AZ_D_graly*0.5, AZ_D_graly,AZ_D_graly*1.5,
                #  SP_D_graly*0.5,SP_D_graly*1, SP_D_graly*1.5, SP_D_graly*4]
                #"D": ["0.5 $\cdot$ $D_A_Z$", "$D_A_Z$", "1.5$\cdot$ $D_A_Z$", "0.5 $\cdot$ $D_S_P$", "$D_S_P$", "1.5 $\cdot$ $D_S_P$", "4$\cdot$ $D_S_P$"],
                #
varvalues_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": ["Regional Default", "0.5x $D_{AZ}$", "$D_{AZ}$", "1.5x $D_{AZ}$", "0.5 x $D_{Sp}$", "$D_{Sp}$", "1.5x $D_{Sp}$", "4x $D_{Sp}$"],
                "DF":[7.5, 15, 22.5],
                "p_re": [0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5],
                "coarse_mass": [.75, 1.5, 2.25],
                "max_coarse_residence_time":[5.5, 11., 16.5]
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
                "DF":[7.5, 15, 22.5],
                "p_re": [0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5],
                "coarse_mass": [.75e3, 1.5e3, 2.25e3],
                "max_coarse_residence_time":[5.5e3, 11.0e3, 16.5e3]
                }

selval_dict = {}

# Select Sample Name
# with lc:
default_ix = list(siu).index("NQT0")

# si = st.selectbox("Choose sample: ", siu, index = default_ix,
    # key = keystr, on_change=proc, args = (keystr,))
lc, mc, rc = st.columns([0.2, 0.6, 0.2])
with mc:
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

bc1, lc, rc, bc2 = st.columns([0.2, 0.3, 0.3, 0.2])
with lc:
    keystr = "model_type_radio"
    model_type = st.radio("Model Type: ", ['simple', 'wdust'], format_func = mtfmt, key = keystr,
        on_change=proc, args = (keystr,))

selval_dict['model_type'] = model_type

with rc:
# Select box model shape:
    keystr = "model_shape_radio"

    model_shape = st.radio("Box shapes: ", ["Uniform height", "Squares", 1.,  5.], index = 1,
        key = keystr, on_change=proc, args = (keystr,), horizontal = True)
selval_dict['model_shape'] = model_shape



def sliderrange(start, step, num):
    return start + np.arange(num)*step
# Instructions & EXplanations:
def varvalsfmt(mt):   # functions to provide vals for 'model_type'
    varvalues_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
        "D": ["Regional Default", r"0.5 x $D_{AZ}$", r"$D_{AZ}$", r"1.5x $D_{AZ}$", r"0.5x $D_{Sp}$", r"$D_{Sp}$", r"1.5x $D_{Sp}$", r"4x $D_{Sp}$"],
        "DF":[7.5, 15, 22.5],
        "p_re": [0.7, 1.4, 2.1],
        "br_E_rate": [7.5, 15, 22.5],
        "coarse_mass": [.75, 1.5, 2.25],
        "max_coarse_residence_time":[5.5, 11., 16.5]
        }
    return varvalues_dict[mt]

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
                st.text("Changes to the input variables will be incorporated to the plots.")

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

                # st.write(' ')

                selval_dict = {}
                for fix, selcolkey in enumerate(selcolu):
                    # filtselcol = st.selectbox("Select Input Variable to Explore:",
                    #   [varnames_dict2[s] for s in selcolu], key = "select_filter_col_"+ samp)
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

                    if selcolkey =="D":
                        # Add note defining DAz etc556495.6872
                        st.write("Meteoric $^{10}$Be delivery rates (D) are site-specific. Graly et al 2010 provides an equation, which yields: ")
                        st.write("$D_{AZ}$ = 5.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")
                        st.write("$D_{SP}$ = 9.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")

                        vvd = ['Regional Default'] + varvalues_dict[selcolkey]
                    else:
                        vvd = varvalues_dict[selcolkey]
                    # if not troubleshoot:

                    val = st.radio(f"{varnames_dict2[selcolkey]}", vvd,
                        key = keystr, on_change=proc, args = (keystr,), horizontal = True)
                    # st.write("3, ")
                    # st.dataframe( dft["D"])


                    v2vdt = {varvalues_dict[selcolkey][ii]:vars_dict[selcolkey][ii] for ii in range(len(varvalues_dict[selcolkey]))}
                    if selcolkey =="D":
                        v2vdt["Regional Default"] = dft["D"].iloc[0]
                    # st.write("v2vdt: ", v2vdt)
                    # st.write("val: ", val)
                    selval_dict[selcolkey] = v2vdt[val]
                    # st.write("4, ")
                    # st.dataframe( dft["D"])

                    dft[selcolkey] = v2vdt[val]
                    # st.write("5, ")

                    # st.dataframe( dft["D"])

                #### Recalc
                # st.write('dft[D]',dft['D'].iloc[0])
                # st.write('type dft[D]',type(dft['D'].iloc[0]))
                # st.write('dft[Inv]',dft['Inv'].iloc[0], type(dft['Inv'].iloc[0]))
                # dff = write_defaults_to_df2(df)
                # dft, SD, N = set_up_df_for_flux_results3(dft,dff)  # calc inventory (depends on z)
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

                # st.write("6, ", dft['Inv'] ,dft['D'])
                Inv = dft['Inv']
                dft['rt'] = (-1.0/ ltl) * log(1 - (ltl* dft['Inv']/ dft['D']))
                v1 = dft['z']
                v2 = dft['D']
                v3 = dft['N']
                v4 = dft['p_re']
                dft['E_fines'] =f_erate(v1, v2, v3, ltl, v4)

                v1 = dft['z']
                v2 = dft['rt']# is this supposed to be in *yrs* or ky?
                v3 = dft['p_re']
                dft['F_fines_boxmodel'] =  flux_boxmodel(v1, v2, v3)

                v1 = dft['coarse_mass']
                v2 = dft['coarse_area']
                v3 = dft['max_coarse_residence_time']

                dft['F_coarse'] = f_coarse_flux(v1, v2, v3)

                v1 = dft['br_E_rate']
                v2 = dft['p_br']
                dft['F_br'] = f_br_flux(v1, v2)

                v1 = dft['F_fines_boxmodel']
                v2 = dft['F_coarse']
                v3 = dft['F_br']
                v4 = dft['DF']
                dft['F_dust'] =  f_mass_balance_for_dust(v1, v2, v3, v4)


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

# # #                     DF = dft['DF']
# # #                     p_re = dft['p_re']

                to_m2_cols = [co for co in dft.columns.to_list() if co.startswith('F_')]
#                 # Change fluxes to m2
#                 # g/cm2/yr  * 100*100cm2/m2
#                 for c,cc in enumerate(to_m2_cols):
#                     dft[cc + '_g_m2_yr'] = dft[cc].apply(lambda x: x*10000).copy()

#                 dft['rt_ky'] = dft['rt'].copy() /1000 # ky

#                 # dftt = dft.copy()

#                 st.write(' ')

#                 with st.popover(f"Plot dimension options"):

#                     # st.write(sliderrange(5, 2, 12))
#                     hh = sliderrange(5, 2, 12)
#                     keystr = "figwidth_radio" + str(samp)
#                     selval_dict['figwidth'] = st.select_slider("Scale figure width: ", options = hh, value = 7,key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
#                     keystr = "figheight_radio"+ str(samp)
#                     selval_dict['figheight']  = st.select_slider("Scale figure height: ",  sliderrange(1, 1,7), value = 3,
#                         key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
#                     # height
#                     keystr = "pixelwidth_radio"+ str(samp)

#                     selval_dict["pixelwidth"] = st.select_slider("Scale width of plot in pixels: ",  sliderrange(500, 50, 12), value = 650,
#                         key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
#                      # Width in px of image produced...
#                     keystr = "boxscale_radio"+ str(samp)

#                     selval_dict["boxscale"] = st.select_slider("Scale boxes within plot: ",  sliderrange(0.8, 0.2, 12), value = 1,
#                         key = keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width
#                     keystr = "shape_buffer_radio"+ str(samp)

#                     selval_dict["shape_buffer"] =st.select_slider("Scale space between boxes within plot: ", sliderrange(0.5, 0.25, 12),
#                          value = 1, key =keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width

#                 # fig = wrap_flux_box_streamlit(dftt, selval_dict)


#                 if model_type == 'simple':
#                     fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,
#                         'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
#                     ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
#                     ftexp = ['Bedrock', 'Coarse Sediment', 'Fine Sediment', 'Dissolved Material']

#                 else:
#                     fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
#                          'F_coarse_g_m2_yr' ,
#                          'F_fines_from_br_g_m2_yr' ,
#                          'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
#                     ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']
#                     ftexp = ['Bedrock','Dust', 'Coarse Sediment', 'Fine Sediment (originating from bedrock)', 'Dissolved Material', 'Dust (Fine sediment originating from dust)']

#                 for i, f in enumerate(fmcols):
#                     dftt[ft[i]] = dftt[f].copy()
#                 # dftt['Sample ID'] = dftt['sample_id']
#                 # st.write(dftt.columns.to_list())
#                 # st.write(dftt[ft])
#                 for i in range(len(ft)):
#                     st.write(f'''{ftexp[i]} Flux''')
#                     st.write(f"{ft[i]}:   {np.round(dftt[ft[i]].to_numpy()[0], 1)} g/m$^2$/yr")
#                 # st.dataframe(dftt[ ft])
#                 count +=1


# # # Add default values
# st.dataframe(df_default)
# df_defaultcols = df_default.columns.to_list()
# for i in range(len(df_default)):
#     st.write(f"{df_defaultcols[i]} {df_default[df_defaultcols[i]].to_}")
