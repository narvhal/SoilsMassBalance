# MAin page in SoilsMassBalance
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
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



st.text("Change the variables and see how the fluxes change!")

siu = df.sample_id.unique()
selcolu = df.select_col.unique()
vars_dict = {}
for i, sc in enumerate(selcolu):
    dft = df[df['select_col'] == sc].copy()
    vars_dict[sc]= dft.select_col_val.unique()


plot_type = "boxflux" #"stackedbarfluxes"



if plot_type == "boxflux":
    selval_dict = {}

    lc, mc, rc = st.columns([0.3, 0.3, 0.3])
    # Select Sample Name
    with lc:
        default_ix = list(siu).index("NQT0")
        si = st.selectbox("Choose sample: ", siu, index = default_ix, key = "sample_id_selbox")
    selval_dict['sample_id'] = si

    # dfdsi = df_default[df_default['sample_id']== si].copy()
    # default_cols = dfdsi.columns.to_list()

    # default_dict = {c:dfdsi[c] for c in default_cols}
    # Select model type (Simple mass balance  (solve for dissolution, no dust) + Compare with calcite mass balance
    #       or with dust  (Dissolution constrained by calcite mass balance) )
    with mc:
        model_type = st.radio("Model Type: ", ['simple', 'wdust'], format_func = mtfmt, key = "model_type_radio")

    selval_dict['model_type'] = model_type

    with rc:
    # Select box model shape:
        model_shape = st.radio("Box shapes: ", ["Uniform height", "Squares"], key = "model_shape_radio")
    selval_dict['model_shape'] = model_shape

    with mc:

        if st.checkbox("Continue?"):
            # Scenario and values:
            baseline_dict = {}
            # units_dict = {}  #
            dft = df.copy()
            for k, vlist in vars_dict.items():
                # bc of the way I structured the df, there is no column for coarse seds subsurface, instead it is "select_col_val"
                # st.write(k, list(vlist[:]))
                # st.write(k in df_default.columns.to_list())
                # st.write(df_default[k])   index = def_ix,
                # vll = list(vlist[:])
                # def_ix = vll.index(default_dict[k])   # Lots of weird errors here as I try to set the default value "value" for the radio button. ugh.
                val = st.radio(f"{k}: ", vlist,  key = str(k) + "_radioval")
                selval_dict[k] = val
                # Filter df outside of func...
                if k == "Coarse_seds_subsurface":
                    dft = dft[dft["select_col"]==k].copy()
                    dft = dft[dft["select_col_val"]==val].copy()
                else:
                    dft = dft[dft[k] == val].copy()
            st.write("Length of df: ", len(dft))
            fig = wrap_flux_box_streamlit(dft, selval_dict)

            st.pyplot(fig)


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
