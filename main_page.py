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
# st.set_page_config(layout="wide" )


fn = "https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/df_all_Tables_vars_baseline.xlsx"
df = pd.read_excel(fn)
df = df[df['select_col'] != "zcol"].copy()  # not useful


fn = "https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/defaults_Tables.xlsx"
df_default =  pd.read_excel(fn)


# toc = stoc()

# Title
st.title("Interactive Soil Mass Balance Plots")
st.header("Flux boxes")

# g/m2/yr?


varnames_dict = {"Coarse_seds_subsurface":"Coarse Sediment \% in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Soil Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction ($F_c$) Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface"
                }
varunits_dict = {"Coarse_seds_subsurface":"(\%)",
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
                "D": ["0.5 $\cdot$ $D_{AZ}$", "$D_{AZ}$", "1.5$\cdot$ $D_{AZ}$", "0.5 $\cdot$ $D_{Sp}$", "$D_{Sp}$", "1.5 $\cdot$ $D_{Sp}$", "4$\cdot$ $D_{Sp}$"],
                "DF":[7.5, 15, 22.5],
                "p_re": [0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5],
                "coarse_mass": [.75, 1.5, 2.25],
                "max_coarse_residence_time":[5.5, 11., 16.5]
                }


siu = df.sample_id.unique()
selcolu = list(varnames_dict.keys()) # df.select_col.unique()
vars_dict = {}
# vars_itemfmt_dict = {}
for i, sc in enumerate(selcolu):
    dft = df[df['select_col'] == sc].copy()
    vars_dict[sc]= dft.select_col_val.unique()


plot_type = "boxflux" #"stackedbarfluxes"
# st.write(vars_itemfmt_dict)



if plot_type == "boxflux":
    selval_dict = {}

    # Select Sample Name
    # with lc:
    default_ix = list(siu).index("NQT0")

    # si = st.selectbox("Choose sample: ", siu, index = default_ix,
        # key = keystr, on_change=proc, args = (keystr,))
    st.write("Choose samples: ")
    si = []
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
    selval_dict['sample_id'] = si
    # dft = df[df['sample_id'] == si].copy()
    # dft = df[df['sample_id'].isin(si)].copy()

    # dfdsi = df_default[df_default['sample_id']== si].copy()
    # default_cols = dfdsi.columns.to_list()

    # default_dict = {c:dfdsi[c] for c in default_cols}
    # Select model type (Simple mass balance  (solve for dissolution, no dust) + Compare with calcite mass balance
    #       or with dust  (Dissolution constrained by calcite mass balance) )

    lc, rc = st.columns([0.6, 0.3])
    with lc:
        keystr = "model_type_radio"
        model_type = st.radio("Model Type: ", ['simple', 'wdust'], format_func = mtfmt, key = keystr,
            on_change=proc, args = (keystr,))

    selval_dict['model_type'] = model_type

    with rc:
    # Select box model shape:
        keystr = "model_shape_radio"

        model_shape = st.radio("Box shapes: ", ["Uniform height", "Squares", 1., 2., 5.],
            key = keystr, on_change=proc, args = (keystr,), horizontal = True)
    selval_dict['model_shape'] = model_shape

    if st.checkbox("Continue?"):
        # Scenario and values:
        baseline_dict = {}
        # units_dict = {}  #
        count = 0

        for six, samp in enumerate(si):
            dft = df[df['sample_id']== samp].copy()

            with st.expander(f"Sample {samp}"):
                st.text("Changes to the input variables will be incorporated to the plots.")

                lc, rc = st.columns([0.5, 0.5])
                colll = [lc,  lc, lc, lc, lc, lc,  rc, lc, lc, lc, lc, lc,  lc, lc, lc, lc, rc,lc,  rc, lc, rc, lc, rc]

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
                filtselcol = st.selectbox("Select Input Variable to Explore:", [str(varnames_dict[s]) for s in selcolu], key = "select_filter_col_"+ samp)
                # st.write(varnames_dict)
                # st.write(filtselcol)
                vixfs = list(varnames_dict.values()).index(filtselcol)
                selcolkey = list(varnames_dict.keys())[vixfs]
                    # with colll[count]:
                # bc of the way I structured the df, there is no column for coarse seds subsurface, instead it is "select_col_val"
                # st.write(k, list(vlist[:]))
                # st.write(k in df_default.columns.to_list())
                # st.write(df_default[k])   index = def_ix,
                # vll = list(vlist[:])
                # def_ix = vll.index(default_dict[k])   # Lots of weird errors here as I try to set the default value "value" for the radio button. ugh.
                # if filtselcol in selcolu:
                if filtselcol == varnames_dict["D"]:
                    # Add note defining DAz etc556495.6872
                    st.write("Meteoric $^{10}$Be delivery rates (D) are site-specific. Graly et al 2010 provides an equation, which yields: ")
                    st.write("$D_{AZ}$ = 5.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")
                    st.write("$D_{SP}$ = 9.6e5 at $^{10}$Be$_{met}$/cm$^2$/yr")

                keystr = str(selcolkey) + "_radioval_"+ str(six)
                st.write("filtselcol: ", filtselcol)
                st.write("selcolkey", selcolkey)
                st.write("vars_dict[selcolkey]",vars_dict[selcolkey])

                # val = st.radio(f"{filtselcol}: ", vars_dict[selcolkey], format_func = varvalsfmt,
                #     key = keystr, on_change=proc, args = (keystr,), horizontal = True)

                val = st.radio(f"{filtselcol}: ", vars_dict[selcolkey],
                    key = keystr, on_change=proc, args = (keystr,), horizontal = True)
                vix = list( vars_dict[selcolkey]).index(val)
                # selval_dict[filtselcol] = val
                # Filter df outside of func...
                # Filter df
                # if len(dft)>1:
                if filtselcol == "Coarse_seds_subsurface":
                    dftt = dft[dft["select_col"]==selcolkey].copy()
                    dftt = dftt[dftt["select_col_val"]==val].copy()
                else:
                    dftt = dft[dft[selcolkey] == dft[selcolkey].unique()[vix]].copy()
                # st.write("Length of df: ", len(dft))
                # count+=1
                # with mc:

                # width = st.sidebar.slider("plot width", 1, 20, 3)
                # height = st.sidebar.slider("plot height", 1, 14, 3)

                def sliderrange(start, step, num):
                    return start + np.arange(num)*step
                with st.popover(f"Plot dimension options"):

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

                fig = wrap_flux_box_streamlit(dft, selval_dict)


        if model_type == 'simple':
            fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,
                'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
            ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
        else:
            fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
                 'F_coarse_g_m2_yr' ,
                 'F_fines_from_br_g_m2_yr' ,
                 'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
            ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']

        # st.pyplot(fig, use_container_width = False)
            # st.pyplot(fig)
        for i, f in enumerate(fmcols):
            dft[ft[i]] = dft[f].copy()

        st.dataframe(dft[ft])

elif plot_type == "stackedbarfluxes":


    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    # K = 10, K = 10 & Dx4
    fns =  [ r'\vars_w_K_10\df_all_Tables.xlsx', r'\vars_w_both_K_10_Dx4\df_all_Tables.xlsx']
    # rows: D change (change table dir)
    # cols: coarse sediment % 0 , 25
    # samples in line:
    si = ['NQT0', 'NQCV2', 'MT120', 'MT130']
    fluxn_dust = ['F_br_normbr_val','F_coarse_normbr_val','F_fines_from_br_normbr_val', 'F_dust_normbr_val', 'F_dissolved_normbr_val', 'F_dissolved_simple_normbr_val']
    fluxn_fines_undifferentiated=['F_br_normbr_val','F_coarse_normbr_val', 'F_fines_boxmodel_normbr_val', 'F_dissolved_normbr_val', 'F_dissolved_simple_normbr_val']
    fluxn_all=['F_br_normbr_val','F_coarse_normbr_val', 'F_fines_boxmodel_normbr_val','F_fines_from_br_normbr_val', 'F_dust_normbr_val', 'F_dissolved_normbr_val', 'F_dissolved_simple_normbr_val']
    fluxcd = {'F_br_normbr_val': 'dimgrey', 'F_coarse_normbr_val': 'grey',
              'F_fines_from_br_normbr_val':'indianred',
              'F_fines_boxmodel_normbr_val':'rosybrown',
              'F_dust_normbr_val':'burlywood',
              'F_dissolved_normbr_val': 'lightcyan',
              'F_dissolved_simple_normbr_val':'steelblue'}
    dl = {'F_br_normbr_val': 'Bedrock',
          'F_coarse_normbr_val': 'Coarse Sediment',
              'F_fines_from_br_normbr_val':'Fine Sediment \nFrom Bedrock',
              'F_fines_boxmodel_normbr_val':'Fine Sediment',
              'F_dust_normbr_val':'Dust',
              'F_dissolved_normbr_val': 'Dissolved',
              'F_dissolved_simple_normbr_val':'Dissolved \n(Simple MB)'}
    for i, aax in enumerate([0,1]):
        dfall = pd.read_excel(saveloc +fns[i])
        dfall = dfall[dfall.sample_id.isin(si)].copy()
        for j, aaxx in enumerate([0,1]):
            # get df needed ---
            axt = ax[i,j]
            plt.sca(axt)

            dft = dfall[dfall.select_col == 'Coarse_seds_subsurface'].copy()
            dftt = dft[dft.select_col_val == dft.select_col_val.unique()[j]].copy()
            dftt['F_dissolved_simple_normbr_val'] = dftt['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_val']/dftt['F_br_val']
            dftt['F_br_normbr_val'] = dftt['F_br_val']/dftt['F_br_val']
            dftt.plot(ax = axt, x = 'sample_id',y = fluxn_dust,  kind = 'bar', stacked = True,  color=fluxcd, rot = 0)
            plt.xlabel('')
            print(dftt[['sample_id', 'D_val', 'F_dissolved_simple_normbr_val', 'select_col_val']])
            if j == 0:
                axt.get_legend().remove()
            else:
                handles, labels = axt.get_legend_handles_labels()
                labeld = [dl[d] for d in labels]
                axt.legend(handles=handles, labels=labeld, title = 'Fluxes Normalized to Bedrock Flux',loc = 2,
                           bbox_to_anchor = (1.02, 1), frameon = False)
    fig.set_size_inches(8.5, 5)
    # plt.annotate(prefix + strf + suffix, xy, xycoords = 'axes fraction', ha = ha, fontsize = 9)

    plt.tight_layout()


    filenametag = '_rows_D_dft_x4_cols_coarse_seds_subs_0_25'

    savefig(filenametag,
                saveloc,
                [],
                [],
                (8.5, 5),
                w_legend=False,
                prefixtag='stacked_norm_vals')
