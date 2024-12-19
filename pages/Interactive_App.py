import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
from io import BytesIO, StringIO
import requests
import base64
from streamlit_pdf_viewer import pdf_viewer
from streamlit_navigation_bar import st_navbar
    
# st.subheader("Modify Flux Inputs")
st.set_page_config( layout="wide" )

# Want to add organic matter as well. 

flag_gh = True
gh_branchname = "main"  # or SoilsMassBalance_Poster_appx or main
if flag_gh:
    # try:
    fn = r"https://github.com/narvhal/SoilsMassBalance/raw/refs/heads/"+gh_branchname+"/data_sources/df_initialize.xlsx"
    #https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/df_initialize.xlsx"
    # fn2 = r"https://github.com/narvhal/SoilsMassBalance/raw/refs/heads/"+gh_branchname+"/data_sources/defaults_Tables.xlsx"
    # https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/defaults_Tables.xlsx"
    fn3 = r"https://github.com/narvhal/SoilsMassBalance/raw/refs/heads/"+gh_branchname+"/data_sources/SGS_geochem.xlsx"
    # except: 
    #     fn = r"/mount/src/soilsmassbalance/data_sources/df_initialize.xlsx"
    #     # fn2 = r"/mount/src/soilsmassbalance/data_sources/defaults_Tables.xlsx"
    #     fn3 = r"/mount/src/soilsmassbalance/data_sources/SGS_geochem.xlsx"
    #     # https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/SGS_geochem.xlsx"
else:
    fn = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\df_initialize.xlsx"
    fn2 = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\defaults_Tables.xlsx"

    fn3 = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\SGS_geochem.xlsx"


# data_sources/df_initialize.xlsx
df = pd.read_excel(fn)
# df_default =  pd.read_excel(fn2)

df_chem =  pd.read_excel(fn3,sheet_name = 'Sheet2', skiprows = 1)

    
varnames_dict = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Fine Fraction of Regolith Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction ($F_c$) Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms $^{10}$Be$_{met}$ Delivered to Surface",
                "N": "Concentration of $^{10}$Be$_{met}$ in Fine Fraction",
                "z": "Fine Fraction of Regolith Depth",
                "C_br": "Bedrock Carbonate",
                "C_f": "Fine Fraction Carbonate",
                "C_c": "Coarse Fraction Carbonate",
                "C_dust": "Dust Fraction Carbonate"
                }
varnames_dict2 = {"Coarse_seds_subsurface":"Coarse Sediment % in subsurface",
                "DF":"Dissolution Factor",
                "p_re": "Fine Fraction of Regolith Density",
                "br_E_rate": "Bedrock Erosion Rate",
                "coarse_mass": "Coarse Fraction Mass",
                "max_coarse_residence_time":"Maximum Coarse Fraction Residence Time",
                "D":"Atoms <sup>10</sup>Be<sub>met</sub> Delivered to Surface",
                "N": "Concentration of <sup>10</sup>Be<sub>met</sub> in Fine Fraction",
                "z": "Fine Fraction of Regolith Depth",
                "C_br": "Percent carbonate in bedrock",
                "C_f": "Percent carbonate in fine sediments",
                "C_c": "Percent carbonate in coarse sediments",
                "C_dust": "Percent carbonate in dust that is carbonate"
                }
varnames_dict2 = varnames_dict
varunits_dict = {"Coarse_seds_subsurface":"%",
                "DF":"Soluble/Insoluble",
                "p_re": "g/cm$^3$",
                "br_E_rate": "mm/ky",
                "coarse_mass": "kg",
                "max_coarse_residence_time":"kyr",
                "D":"atoms/cm$^2$/yr",
                "N": "atoms/g",
                "z": "cm",
                "C_br": "%",
                "C_f": "%",
                "C_c": "%",
                "C_dust": "%"
                }
                #  vals_arr = [ AZ_D_graly*0.5, AZ_D_graly,AZ_D_graly*1.5,
                #  SP_D_graly*0.5,SP_D_graly*1, SP_D_graly*1.5, SP_D_graly*4]
                #"D": ["0.5 $\cdot$ $D_A_Z$", "$D_A_Z$", "1.5$\cdot$ $D_A_Z$", "0.5 $\cdot$ $D_S_P$", "$D_S_P$", "1.5 $\cdot$ $D_S_P$", "4$\cdot$ $D_S_P$"],
                #
# HAS to match vars_dict_orig
varvalues_dict_orig= {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": [ "0.5x $D_{AZ}$", "$D_{AZ}$", "1.5x $D_{AZ}$", "0.5 x $D_{Sp}$", "$D_{Sp}$", "1.5x $D_{Sp}$", "4x $D_{Sp}$"],
                "DF":[2.5, 5, 7.5, 15, 22.5],
                "p_re": [0.1, 0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5, 50, 70],
                "coarse_area": [1, 100, 1000],
                "coarse_mass": [.75, 1.5, 2.25,3],
                "max_coarse_residence_time":[2, 5.5, 11., 16.5, 20],
                "N": ["0.5x $N_{AZ}$", "$N_{AZ}$","1.5x $N_{AZ}$",  "0.5 x $N_{SP}$", "$N_{SP}$", "1.5x $N_{SP}$", "4x $N_{SP}$"],
                "z": [ 5, 10, 20, 50],
                "C_br": [50, 90, 100],
                "C_f": [0, 5, 10, 15, 20, 25, 30, 90],
                "C_c": [10, 50, 90, 100],
                "C_dust": [1, 10, 20, 30, 40, 50, 90]
                }

D = 1.8e6 # at/cm2/yr
p_re = 1.4 # g/cm3
lamba = 5.1e-7 # 1/yr
atom10Bemass = 1.6634e-23 #g/at
AZ_D_graly = D_graly(400, 31.2)
SP_D_graly = D_graly(510, 39.1) # used to be 450 mm/yr precip...



#lambda
ltl = 5.1e-7 # , 7.20e-7]  #1/yr lambda [Nishiizumi et al., 2007], Korschinek et al. (2010)

#
siu_dict = {"Semi-Arid": "NQT0",  "Arid":"MT120"}
selcolu = list(varnames_dict.keys()) # df.select_col.unique()
si_id = list(varnames_dict.values()) # df.select_col.unique()

dft = df[df['sample_id']== "NQT0"].copy()
N_SP = dft['N_val'].astype(float).values[0]
dft = df[df['sample_id']== "MT120"].copy()
N_AZ = dft['N_val'].astype(float).values[0]


vars_dict_orig = {"Coarse_seds_subsurface":[0, 25, 50, 75],
                "D": [ 0.5*AZ_D_graly, AZ_D_graly, 1.5*AZ_D_graly, 0.5*SP_D_graly, SP_D_graly, 1.5* SP_D_graly, 4*SP_D_graly],
                "DF":[2.5, 5, 7.5, 15, 22.5],
                "p_re": [0.1, 0.7, 1.4, 2.1],
                "br_E_rate": [7.5, 15, 22.5, 50, 70],
                "coarse_mass": [.75e3, 1.5e3, 2.25e3,3e3],
                "coarse_area": [1, 100, 1000],
                "max_coarse_residence_time":[2e3, 5.5e3, 11.0e3, 16.5e3,20e3],
                "N": [ 0.5*N_AZ, N_AZ, 1.5*N_AZ, 0.5*N_SP, N_SP, 1.5* N_SP, 4*N_SP],
                "z": [5, 10, 20, 50, 100],
                "C_br": [50, 90, 100],
                "C_f": [0, 5, 10, 15, 20, 25, 30, 90],
                "C_c": [10, 50, 90, 100],
                "C_dust": [1, 10, 20, 30, 40, 50, 90]}

selval_dict = {}
selval_dict["AZ_D_graly"] = AZ_D_graly
selval_dict["SP_D_graly"] = SP_D_graly
selval_dict["N_SP"] = N_SP
selval_dict["N_AZ"] = N_AZ

# Disable model selection for now, until I have a better wy of displaying the simple model violating the carbonate mass balance. 
model_type = 'wdust'
selval_dict['model_type'] = model_type

# Select Sample Site 
default_site = list(siu_dict.keys())[0]

# st.write("Changes to the input variables will be immediately reflected in the mass balance flux plot below.")
keystr = "sample_id_selbox" + "0"
lc, rc, ec, tc = st.columns([0.2, 0.15,0.15, 0.5])
with lc:
    st.write("Choose sample sites: ")
with rc:
    si0 = st.checkbox( list(siu_dict.keys())[0], value = True, key = keystr, on_change=proc, args = (keystr,))
keystr = "sample_id_selbox" + "1"
with ec:
    si1 = st.checkbox(list(siu_dict.keys())[1], value = False, key = keystr , on_change=proc, args = (keystr,))
si = []
if si0: 
    si = [list(siu_dict.keys())[0]]
    if si1: 
        si = [list(siu_dict.keys())[0]] + [list(siu_dict.keys())[1]]
elif si1: 
    si =  [list(siu_dict.keys())[1]]
else: 
    si = []
    st.write("Select Site to show model!")
# if si1: si = si + [list(siu_dict.keys())[1]]

list_of_sample_id = [siu_dict[ss] for ss in si]
list_of_sample_climates = [ss for ss in si]
selval_dict['sample_id'] = list_of_sample_id

if model_type == 'simple':
    fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
    ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
    ftexp = ['Bedrock', 'Coarse Sediment', 'Fine Sediment', 'Dissolved Material (Mass Balance)']
elif model_type == 'wdust':
    fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
         'F_coarse_g_m2_yr' ,
         'F_fines_from_br_g_m2_yr',
         'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
    ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']
    ftexp = ['Bedrock','Dust', 'Coarse Sediment', 'Fine Sediment (originating from bedrock)','Dissolved Material', 'Dust (Fine sediment originating from dust)']
# st.write(model_type)
selval_dict["fmcols"] = fmcols
selval_dict["ft"] = ft
selval_dict["ftexp"] = ftexp
fmtcols = ['br_E_rate','F_br', 'F_br_g_m2_yr']
selval_dict["fmtcols"] = fmtcols
selval_dict["flag_sample_label_default"] = True


# Instructions & EXplanations:
count = 0

for six, samp in enumerate(list_of_sample_id):
    dft = df[df['sample_id']== samp].copy()
    dft['Coarse_seds_subsurface'] = 0
    dft['z'] = dft['z_val'].astype(float).values[0]
    dft['p_re'] = dft['p_re_val'].astype(float).values[0]
    dft['p_br'] = dft['p_br_val'].astype(float).values[0]
    dft['N'] = dft['N_val'].astype(float).values[0]
    dft['coarse_mass'] = dft['coarse_mass_val'].astype(float).values[0]
    dft['coarse_area'] = dft['coarse_area_val'].astype(float).values[0]
    dft['br_E_rate'] = dft['br_E_rate_val'].astype(float).values[0]
    dft['Inv'] = dft['Inv_val'].astype(float).values[0]
    dft['D'] = dft['D_val'].astype(float).values[0]
    dft['DF'] = dft['DF_val'].astype(float).values[0]
    dft['max_coarse_residence_time'] = dft['max_coarse_residence_time_val'].astype(float).values[0]

    # have to solve for default Carb BR based on DF: ratio of SOLUBLE/INSOLUBLE products
    # So % Insoluble = INSOLUBLE /(Soluble + Insoluble) = 1/(DF + 1)
    df_chemt = df_chem[df_chem['Sample_Name'] == dft['geochem_br_sample_names__sheet2'].values[0]].copy()
    Xbr = 1/(dft['DF'].astype(float).values[0] + 1) #df_chemt['X_fines'].astype(float).values[0]  # This column is in pct
    # st.write('Pct noncarb in br: {:0.2f}'.format(Xbr*100))
    df_chemt = df_chem[df_chem['Sample_Name'] == dft['geochem_fines_sample_names__sheet2'].values[0]].copy()
    Xf = 1  # df_chemt['X_fines'].astype(float).values[0]
    if Xbr<0:
        Xbr = .01
    if Xf<0:
        Xf = .01

    Xdust = 1

    dft_def = dft.copy()

    vars_dict_def = {"Coarse_seds_subsurface":dft_def['Coarse_seds_subsurface'].astype(float).values[0],
            "D": dft_def['D'].astype(float).values[0],
            "DF":dft_def['DF'].astype(float).values[0],
            "p_re": dft_def['p_re'].astype(float).values[0],
            "br_E_rate": dft_def['br_E_rate'].astype(float).values[0],
            "coarse_mass":  dft_def['coarse_mass'].astype(float).values[0],
            "coarse_area":  dft_def['coarse_area'].astype(float).values[0],
            "max_coarse_residence_time":dft_def['max_coarse_residence_time'].astype(float).values[0],
            "N": dft_def['N'].astype(float).values[0],
            "z": dft_def['z'].astype(float).values[0],
            "C_br": (100-Xbr*100),
            "C_c": (100-Xbr*100),
            "C_f": (100-Xf*100),
            "C_dust": (100 - Xdust*100)
            }
    # Appearance of default vars
    varsfmttd_dict_def = {"Coarse_seds_subsurface":f'{vars_dict_def["Coarse_seds_subsurface"]:0.0f}',
            "D":f'{vars_dict_def["D"]:0.2e}',
            "DF":f'{vars_dict_def["DF"]:0.1f}',
            "p_re": f'{vars_dict_def["p_re"]:0.1f}',
            "br_E_rate": f'{vars_dict_def["br_E_rate"]:0.1f}',
            "coarse_mass":  f'{vars_dict_def["coarse_mass"]/1e3:0.0f}',
            "coarse_area":  f'{vars_dict_def["coarse_area"]:0.0f}',
            "max_coarse_residence_time":f'{vars_dict_def["max_coarse_residence_time"]/1e3:0.1f}',
            "N":f'{vars_dict_def["N"]:0.2e}',
            "z": f'{vars_dict_def["z"]:0.1f}',
            "C_br": f'{vars_dict_def["C_br"]:0.1f}',
            "C_c": f'{vars_dict_def["C_c"]:0.1f}',
            "C_f": f'{vars_dict_def["C_f"]:0.1f}',
            "C_dust": f'{vars_dict_def["C_dust"]:0.1f}'
            }
    selval_dict_def = {}
    kki = ["fmtcols", "ft", "ftexp", 'fmcols', 'flag_sample_label_default', 'model_type']
    for k in kki:
        selval_dict_def[k] = selval_dict[k]

    # Calculate and plot default values: 
    varvalues_dict_def = varvalues_dict_orig
    vars_dict_def2 = vars_dict_orig
    for k in list(vars_dict_def.keys()):
        selval_dict_def[k] = vars_dict_def[k]
        varvalues_dict_def[k] = [f"Default: {varsfmttd_dict_def[k]}" ] + varvalues_dict_def[k]
        vars_dict_def2[k] = [vars_dict_def[k] ] + vars_dict_def2[k] 
        dft_def[k] = vars_dict_def[k]
        dft[k] = vars_dict_def[k]
    selval_dict_def['model_type'] = model_type
    # simple recalc inputs: 
    dfti, selval_dict_def = simple_recalc(dft_def, selval_dict_def)

    # Provide plot dimension defaults early
    kk = {'model_shape':"Uniform height", 'figwidth':9, 'figheight':3, 'pixelwidth':525, 'boxscale':1, 'shape_buffer':2, 'boxheight':2, 'medfont':12, 'textheight':3.25}

    for ki, key in enumerate(list(kk.keys())):
        selval_dict[key] = kk[key]
        selval_dict_def[key] = kk[key]

    varvalues_dict = varvalues_dict_def
    vars_dict = vars_dict_def2

    if len(si) >1:
        coL = st.container()
        coR = st.container()
        # coL, coR = st.columns([0.5,  0.5])
        colll = [ coL,  coR,  coL, coR, coL,  coR,  coL, coR]
    elif len(si) == 1:
        coL = st.container()
        colll = [ coL]

    left_vars, right_results = st.columns([0.4, 0.6])
    count = 0
    with left_vars:
        with st.container(height = 600, border = True):
            #  \textcolor{#808080}   Fb
            #  \textcolor{#deb887}   Fd
            #  \textcolor{#bc8f8f}   Fc
            #  \textcolor{#cd5c5c}   Ffbr
            #  \textcolor{#e0ffff}   Fdis
            # \textcolor{#deb887}{F_d} 

            txt_exp1 = f'''<font color="deb887">Modify Bedrock Inputs</font>'''
            exp_br = st.expander(txt_exp1) #, key = "expand_carb_comp_br"+str(count))
            exp_c = st.expander(f"**Modify Coarse Sediment Inputs**")#, key = "expand_carb_comp_c"+str(count))
            exp_f = st.expander(f"**Modify Dust and Fine Sediment Inputs**") #, key = "expand_carb_comp_f"+str(count))
            with exp_br:
                dft,selval_dict = plot_carb_pct(dft,selval_dict, collist = ['C_br'],labellist = ['Bedrock Composition'],ft = ['F$_b$'], ec = 'k', six = six)

            with exp_c:
                dft,selval_dict = plot_carb_pct(dft,selval_dict, collist = [ 'C_c'],labellist = ['Coarse Sediment Composition'],ft = [ 'F$_c$'], ec = 'k', six = six)
            with exp_f:
                dft,selval_dict = plot_carb_pct(dft,selval_dict, collist = [ 'C_f', 'C_dust'],labellist = ['Fine Sediment Composition', 'Dust Composition'],ft = ['F$_f$','F$_{dust}$'], ec = 'k',six = six)
                
            expb_d = {"Coarse_seds_subsurface": "What if regolith contains coarse sediments? (Only coarse sediments were measured, this variable is likely NOT zero.)",
                "D":"Meteoric $^{10}$Be delivery rates (D) are site-specific.",
                "coarse_mass": "The mass of coarse sediment measured on the surface.",
                "max_coarse_residence_time": "The maximum additional exposure time coarse sediments could endure while sharing the erosion history of the bedrock.",
                "N": "Measured concentration of $^{10}$Be$_{m}$ in fine fraction.",
                "z": "Measured depth of the fine fraction of mobile regolith. ",
                "p_re": "Dust is assumed to have the same density as the fine fraction of mobile regolith.",
                "br_E_rate": "Bedrock erosion rate and flux of bedrock are directly related by the density of the material.",
                "C_br": "Percent of bedrock that is soluble (e.g. CaCO3).",
                "C_c": "Percent of coarse sediments that are soluble (e.g. CaCO3).",
                "C_f": "Percent of fine sediments that are soluble (e.g. CaCO3).",
                "C_dust": "Percent of dust that is soluble (e.g. CaCO3)."
            }
            fmtfc = ['.2e', '.0f', '.0f', '.1f', '.1f', '.1f']
            user_option_keys = sorted(list(set(list(expb_d.keys())) - set(['Coarse_seds_subsurface', 'C_br', 'C_c', 'C_f', 'C_dust'])))
            selval_dict['Coarse_seds_subsurface'] = selval_dict_def['Coarse_seds_subsurface']
            # Move D and N option to last in line
            user_option_keys = user_option_keys[2:] + user_option_keys[0:2]
            len_user_optionk = len(user_option_keys)
            for sck, selcolkey in enumerate(user_option_keys):
                if sck < 1:
                    sck_col = exp_br
                elif sck <(np.floor(len_user_optionk/2)):
                    sck_col = exp_c  
                else : 
                    sck_col = exp_f   
                
                with sck_col:
                    dft, selval_dict = Make_Var_Radio(dft, selcolkey, selval_dict, varvalues_dict, varnames_dict2, vars_dict, six, expb_d, varunits_dict)

            if six == 0:
                with st.expander("**See plot dimension options**"): #,  key='exp_show_plot_dimensions'):
                    # Select box model shape:
                    keystr = "model_shape_radio_" + str(samp)
                    model_shape = st.radio("Box shapes: ", ["Uniform height", "Squares", 1.,  5.], index = 0, key = keystr, on_change=proc, args = (keystr,), horizontal = True)
                    selval_dict['model_shape'] = model_shape

                    # start, step, number
                    hh = sliderrange(5, 1, 10)
                    keystr = "figwidth_radio" + str(samp)
                    selval_dict['figwidth'] = st.select_slider("Scale figure width: ", options = hh, value = kk['figwidth'],key = keystr, on_change = proc, args = (keystr,))

                    keystr = "figheight_radio"+ str(samp)
                    selval_dict['figheight']  = st.select_slider("Scale figure height: ",  sliderrange(1, 1,7), value = kk['figheight'],
                        key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width
                    # height
                    keystr = "pixelwidth_radio"+ str(samp)
                    selval_dict["pixelwidth"] = st.select_slider("Scale width of plot in pixels: ",  sliderrange(200, 25, 25), value = kk['pixelwidth'],
                        key = keystr, on_change = proc, args = (keystr,))#, horizontal = True) # width

                     # Width in px of image produced...
                    keystr = "boxscale_radio"+ str(samp)
                    selval_dict["boxscale"] = st.select_slider("Scale boxes within plot: ",  np.arange(20)/5 + 0.2, value = kk['boxscale'],
                        key = keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width

                    keystr = "shape_buffer_radio"+ str(samp)
                    selval_dict["shape_buffer"] =st.select_slider("Scale space between boxes within plot: ", sliderrange(0.5, 0.25, 20), value = kk['shape_buffer'], key =keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width

                    keystr = "boxheight_radio"+ str(samp)
                    selval_dict['boxheight'] = st.select_slider("Scale box height: ", sliderrange(0.25, 0.25, 16),value = kk['boxheight'], key =keystr, on_change = proc, args = (keystr,))

                    keystr = "medfont_radio"+ str(samp)
                    selval_dict["medfont"]=st.select_slider("Scale label font size: ", sliderrange(6, 1, 15), value = kk['medfont'], key =keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width

                    keystr = "textheight_radio"+ str(samp)
                    selval_dict["textheight"]=st.select_slider("text height: ", sliderrange(0, 0.25, 28),value = kk['textheight'], key =keystr, on_change = proc, args = (keystr,)) #, horizontal = True) # width
                        
                    kki = ['model_shape', 'figwidth', 'figheight', 'pixelwidth', 'boxscale', 'shape_buffer', 'boxheight', 'medfont', 'textheight']
                    for k in kki:
                        selval_dict_def[k] = selval_dict[k]

            #### Recalc
            ##############################
            # Whether this is plotting default values or modified depends whether dataframe dft has been modified. 
            dft, selval_dict = partial_recalc(dft, selval_dict)

            st.write(f"Coarse Sediment Residence Time: {np.round(dft['max_coarse_residence_time'].iloc[0]/1000, 2)} ky")
            st.write(f"Fine Sediment Residence Time: {np.round(dft['rt'].iloc[0]/1000, 2)} ky")

            key = "st_check_CoarseResTime" + str(six)
            st_check_CoarseResTime =  st.checkbox(f"Set coarse sediment residence time to equal fine sediment residence time: {np.round(dft['rt'].iloc[0]/1000, 2)} ky", key = key, on_change = proc, args = (key,))
            if st_check_CoarseResTime:
                # Snippet taken from partial recalc func
                v1 = dft['coarse_mass']
                v2 = dft['coarse_area']
                v3 = dft['rt']   # This is the fine sediment fraction
                dft['F_coarse'] = f_coarse_flux(v1, v2, v3)

            # Load modified values from dft, convert to g/cm2/yr
            F_fines =dft['F_fines_boxmodel'].astype(float).values[0]
            F_coarse = dft['F_coarse'].astype(float).values[0]
            F_br = dft['F_br'].astype(float).values[0]

            dft, X_c, X_f, X_br, X_dust = get_X_vals(dft)
            # st.write(dft['X_f'])

            # try getting F_diss first, maybe solves issue??
            F_diss = f_noncarb_mass_balance_for_diss(F_fines, F_coarse, F_br, X_c,X_f, X_br, X_dust)

            F_dust = f_dust_from_other_F(F_fines, F_coarse, F_br, F_diss)

            # F_dust =  f_noncarb_mass_balance_for_dust(F_fines, F_coarse, F_br,  X_c, X_f, X_br, X_dust)
            # F_diss =  f_diss_from_other_F(F_fines, F_coarse, F_br, F_dust)

            dft['F_dust'] = F_dust 
            dft['F_dissolved'] = F_diss
            dft['F_fines_from_br'] =f_f_from_br(dft)

            dft = modify_F_units(dft, to_m2_cols = ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust','F_dissolved', 'F_fines_from_br'])

    with right_results:
        with st.container(border = False): # , key = "left_vars"+str(count)):
            st.write(f"**{list_of_sample_climates[six]} Site**")
            lct, rct = st.columns([0.3, 0.7])
            with lct:
                st.markdown("*Modified Input*")

            fig = wrap_flux_box_streamlit(dft, selval_dict)

            # st.write(f"F$_f$ (alternative method) = D/N x 10000 = {np.round(dft['D'].iloc[0]/dft['N'].iloc[0]*10000,1)} g/m$^2$/yr")

            buf = BytesIO()
            fig.savefig(buf, format="png")
            # lct, rct= st.columns([0.3, 0.7])
            # st.image(buf, width = selval_dict["pixelwidth"])
            with rct:
                st.download_button(label ="Download Image",
                    data=buf,
                    file_name="Modified_Mass_Balance_Fluxes.png",
                    mime="image/png", key = "Download_modified_" + str(six))

            # if st.checkbox("View Mass Balance with Default Variables", value = True, key ="chkbx_Download_default_" + str(six)):
            lct, rct = st.columns([0.3, 0.7])
            with lct:
                st.write(f"*Default inputs*")
            fig_def = wrap_flux_box_streamlit(dfti, selval_dict_def)

            buf = BytesIO()
            fig_def.savefig(buf, format="png")
            with rct:
                st.download_button(label ="Download Image",
                    data=buf,
                    file_name="Default_Mass_Balance_Fluxes.png",
                    mime="image/png",  key = "Download_default_" + str(six))
                        
            ###############
            # need to pretty-fy this report
            # if st.checkbox("See Default and Modified Input Values and Fluxes",key = "see_default_inputs" + str(samp)):
            #     lc, rc = st.columns([0.5, 0.5])
            #     with lc:
            #         st.write(f"**Default values**")
            #         dfti['X_c'] = X_c
            #         dfti['X_f'] = X_f
            #         dfti['X_br'] = X_br  
            #         dfti['X_dust'] = X_dust
            #         add_val_report(dfti, user_option_keys,selval_dict)
            #     with rc:
            #         st.write(f"**Modified Values**")
            #         add_val_report(dft,user_option_keys, selval_dict)
            
            count +=1


run_mkdn_style()
