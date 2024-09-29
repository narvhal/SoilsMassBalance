import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
from io import BytesIO
import requests

# from py_functions.load_intro import *
# from py_functions.load_prep_initial_df import *
# from py_functions.load_wrap_plotting_funcs import *
# from py_functions.load_plotting_funcs import *       # map_cdict
# from py_functions.load_plotting_funcs_02_13 import *
# from uncertainties import ufloat_fromstr
st.set_page_config(layout="wide" )


troubleshoot = False
flag_gh = False 


## Add soil depth as variable.

if troubleshoot: pass
else:

    if flag_gh:

        fn = "https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/df_all_Tables_vars_baseline.xlsx"
        fn1 = "https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/defaults_Tables.xlsx"
    else:
        fn = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\df_all_Tables_vars_baseline.xlsx"
        fn1 = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\defaults_Tables.xlsx"
    df = pd.read_excel(fn)
    df = df[df['select_col'] != "zcol"].copy()  # not useful

    df_default =  pd.read_excel(fn1)


    # toc = stoc()

    # Title
    st.title("Interactive Soil Mass Balance Plots")
    st.header("Flux boxes")


    st.write("  ")
    st.write("This app is not yet done! I'll add clarification for how to use it soon. ")


    varnames_dict = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                    "DF":"Dissolution Factor",
                    "p_re": "Soil Density",
                    "br_E_rate": "Bedrock Erosion Rate",
                    "coarse_mass": "Coarse Fraction ($F_c$) Mass",
                    "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                    "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface"
                    }
    varnames_dict2 = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                    "DF":"Ratio of Soluble to Insoluble Mass in Bedrock",
                    "p_re": "Soil Density",
                    "br_E_rate": "Bedrock Erosion Rate",
                    "coarse_mass": "Coarse Fraction Mass",
                    "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                    "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface"
                    }
    varunits_dict = {"Coarse_seds_subsurface":"(%)",
                    "DF":"(Soluble/Insoluble)",
                    "p_re": "(g/cm$^3$)",
                    "br_E_rate": "(mm/ky)",
                    "coarse_mass": "(kg)",
                    "max_coarse_residence_time":"(kyr)",
                    "D":"Atoms/cm$^2$/yr"
                    }



    varvalues_dict = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                    "D": ["0.5x $D_{AZ}$", "$D_{AZ}$", "1.5x $D_{AZ}$", "0.5 x $D_{Sp}$", "$D_{Sp}$", "1.5x $D_{Sp}$", "4x $D_{Sp}$"],
                    "DF":[7.5, 15, 22.5],
                    "p_re": [0.7, 1.4, 2.1],
                    "br_E_rate": [7.5, 15, 22.5],
                    "coarse_mass": [.75, 1.5, 2.25],
                    "max_coarse_residence_time":[5.5, 11., 16.5]
                    }

    selval_dict = {}



    def sliderrange(start, step, num):
        return start + np.arange(num)*step
    with st.popover(f"Plot dimension options"):

        # st.write(sliderrange(5, 2, 12))
        hh = sliderrange(5, 2, 12)
        keystr = "figwidth_radio" 
        selval_dict['figwidth'] = st.select_slider("Scale figure width: ", options = hh, value = 7,key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
        keystr = "figheight_radio"
        selval_dict['figheight']  = st.select_slider("Scale figure height: ",  sliderrange(1, 1,7), value = 3,
            key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
        # height
        keystr = "pixelwidth_radio"

        selval_dict["pixelwidth"] = st.select_slider("Scale width of plot in pixels: ",  sliderrange(500, 50, 12), value = 650,
            key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
         # Width in px of image produced...
        keystr = "boxscale_radio"

        selval_dict["boxscale"] = st.select_slider("Scale boxes within plot: ",  sliderrange(0.8, 0.2, 12), value = 1,
            key = keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width
        keystr = "shape_buffer_radio"

        selval_dict["shape_buffer"] =st.select_slider("Scale space between boxes within plot: ", sliderrange(0.5, 0.25, 12),
             value = 1, key =keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width


    model_type = "wdust"
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
        ftexp = ['Bedrock','Dust', 'Coarse Sediment', 'Fine Sediment (originating from bedrock)', 'Dissolved Material', 'Dust (Fine sediment originating from dust)']

    selval_dict["fmcols"] = fmcols
    selval_dict["ft"] = ft
    selval_dict["ftexp"] = ftexp
    siu = df.sample_id.unique()
    selcolu = list(varnames_dict.keys()) # df.select_col.unique()
    vars_dict = {}
    # vars_itemfmt_dict = {}
    for i, sc in enumerate(selcolu):
        dft = df[df['select_col'] == sc].copy()
        vars_dict[sc]= dft.select_col_val.unique()


    plot_type = "boxflux"

    ## No default sample selection for Poster ---> Just show "Arid" and "semi-arid"


    # Select Sample Name
    si_to_poster = {"NQT0":"Semi-Arid", "MT120":"Arid"}

    lc, mc, rc = st.columns([0.2, 0.6, 0.2])
    with mc:
        # st.write("Choose samples: ")
        # keystr = "sample_id_selbox"
        # si = st.multiselect(" ", siu, default = ["NQT0", "MT120"], key = keystr, on_change=proc, args = (keystr,))["NQT0", "MT120"]
        si = ["NQT0", "MT120"]
        selval_dict['sample_id'] = si


        # Select model type (Simple mass balance  (solve for dissolution, no dust) + Compare with calcite mass balance
        #       or with dust  (Dissolution constrained by calcite mass balance) )

        # bc1, lc, rc, bc2 = st.columns([0.2, 0.3, 0.3, 0.2])
        # with lc:
            # keystr = "model_type_radio"
            # model_type = st.radio("Model Type: ", ['simple', 'wdust'],index = 1, format_func = mtfmt, key = keystr,
                # on_change=proc, args = (keystr,))
        model_type = "wdust"
        selval_dict['model_type'] = model_type

        # with rc:
        # Select box model shape:
            # keystr = "model_shape_radio"

            # model_shape = st.radio("Box shapes: ", ["Uniform height", "Squares", 1.,  5.], index = 1,
                # key = keystr, on_change=proc, args = (keystr,), horizontal = True)
        model_shape = "Uniform height"
        selval_dict['model_shape'] = model_shape



        # Instructions & EXplanations:


        # Scenario and values:
        baseline_dict = {}
        # units_dict = {}  #
        # if len(si) == 3:
        #     coL, coM, coR = st.columns([0.33, 0.33, 0.33])
        #     colll = [ coL, coM, coR,  coL, coM, coR]
        # elif len(si) == 2:
        #     coL, coR = st.columns([0.5,  0.5])
        #     colll = [ coL,  coR,  coL, coR, coL,  coR,  coL, coR]
        # elif len(si) == 1:
        coL = st.container()
        colll = [ coL]
        st.text("Changes to the input variables will be incorporated to the plots.")

        count = 0
        for six, samp in enumerate(si):
            dft = df[df['sample_id']== samp].copy()

            with coL:
                with st.expander(si_to_poster[samp], expanded = True):

                    plot_data_default = [True, False]

                    for pdd in plot_data_default:

                        if pdd:
                            dftt = dft.copy()
                        else:
                            lc, rc = st.columns([0.5, 0.5])

                            for k, vlist in vars_dict.items():
                                vld = []
                                # vars_itemfmt_dict[k] = {}
                                tempd = {}
                                for j, sv in enumerate(vlist):
                                    valid_list = varvalues_dict[k]
                                    tempd[sv] = valid_list[j]
                                # vars_itemfmt_dict[sc]= tempd
                                # vld.append(tempd)
                                def varvalsfmt(mt, dc = tempd):   # functions to provide vals for 'model_type'
                                    return dc[mt]
                            st.write(' ')
                            varname_units = [varnames_dict2[s] + " " + varunits_dict[s] for s in selcolu]
                            filtselcol = st.selectbox("Select Input Variable to Explore:",[varnames_dict2[s] for s in selcolu] , key = "select_filter_col_"+ samp)
                            vixfs = list(varnames_dict2.values()).index(filtselcol)
                            selcolkey = list(varnames_dict2.keys())[vixfs]
                            selcolunit = list(varunits_dict.values())[vixfs]

                            # with colll[count]:
                            # bc of the way I structured the df, there is no column for coarse seds subsurface, instead it is "select_col_val"
                            # st.write(k, list(vlist[:]))
                            # st.write(k in df_default.columns.to_list())
                            # st.write(df_default[k])   index = def_ix,
                            # vll = list(vlist[:])
                            # def_ix = vll.index(default_dict[k])   # Lots of weird errors here as I try to set the default value "value" for the radio button. ugh.
                            # if filtselcol in selcolu:
                            st.write(selcolunit)
                            if filtselcol == varnames_dict["D"]:
                                # Add note defining DAz etc556495.6872
                                st.write("Meteoric $^{10}$Be delivery rates (D) are site-specific. Graly et al 2010 provides an equation, which yields: ")
                                st.write("$D_{AZ}$ = 5.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")
                                st.write("$D_{SP}$ = 9.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")

                            keystr = str(selcolkey) + "_radioval_"+ str(six)
                            # st.write("filtselcol: ", filtselcol)
                            # st.write("selcolkey", selcolkey)
                            # st.write("vars_dict[selcolkey]",vars_dict[selcolkey])

                            # val = st.radio(f"{filtselcol}: ", vars_dict[selcolkey], format_func = varvalsfmt,
                            #     key = keystr, on_change=proc, args = (keystr,), horizontal = True)

                            # val = st.radio(f"{filtselcol}: ", vars_dict[selcolkey],
                            val = st.radio(" ", vars_dict[selcolkey],
                                key = keystr, on_change=proc, args = (keystr,), horizontal = True)
                            vix = list( vars_dict[selcolkey]).index(val)
                            # selval_dict[filtselcol] = val
                            # Filter df outside of func...
                            # Filter df
                            # if len(dft)>1:
                            # if filtselcol == "Coarse_seds_subsurface":
                            dftt = dft[dft["select_col"]==selcolkey].copy()
                            dftt = dftt[dftt["select_col_val"]==val].copy()
                            # else:
                                # dftt = dft[dft[selcolkey] == dft[selcolkey].unique()[vix]].copy()
                            # st.write("Length of df: ", len(dft))
                            # count+=1
                            # with mc:

                            # width = st.sidebar.slider("plot width", 1, 20, 3)
                            # height = st.sidebar.slider("plot height", 1, 14, 3)
                            st.write(' ')

                        fig = wrap_flux_box_streamlit(dftt, selval_dict)

                        # for i, f in enumerate(fmcols):
                        #     dftt[ft[i]] = dftt[f].copy()
                        # # dftt['Sample ID'] = dftt['sample_id']
                        # # st.write(dftt.columns.to_list())
                        # # st.write(dftt[ft])
                        # for i in range(len(ft)):
                        #     st.write(f'''{ftexp[i]} Flux''')
                        #     st.write(f"{ft[i]}:   {np.round(dftt[ft[i]].to_numpy()[0], 1)} g/m$^2$/yr")
                        # # st.dataframe(dftt[ ft])
                        count +=1


        # # Add default values
        # st.dataframe(df_default)


        # df_defaultcols = df_default.columns.to_list()
        # for i in range(len(df_default)):
        #     st.write(f"{df_defaultcols[i]} {df_default[df_defaultcols[i]].to_}")



        # filenametag = '_rows_D_dft_x4_cols_coarse_seds_subs_0_25'

        # savefig(filenametag,
        #             saveloc,
        #             [],
        #             [],
        #             (8.5, 5),
        #             w_legend=False,
        #             prefixtag='stacked_norm_vals')
