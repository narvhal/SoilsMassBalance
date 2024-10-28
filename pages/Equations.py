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


cc, rc = st.columns([0.61,0.39])
with cc:
    st.subheader(f"**Mass Balance Equations**")

    st.write(f"Mass Balance Including dust, constrained by insoluble material mass balance")
    st.latex(r"F_b + {\color{olive}F_d} = {\color{grey}F_c} + {\color{red}F_{f}} + {\color{teal}F_{dis}}")
    st.write(f"where **$F_b$** is bedrock mass flux, **$F_d$** is dust mass flux, **$F_c$** is coarse fraction of sediment mass flux, **$F_{{dis}}$** is dissolved material mass flux, and **$F_f$** is the entire fine fraction of sediment. ")
    st.write(f"Note that:  ")
    st.latex( r"{\color{red}F_f} = {\color{purple}F_{f,b}} + {\color{olive}F_d}")
    st.write(f"where **$F_{{f,b}}$** is insoluble material derived from dissolved bedrock. The technique used here quantifies $F_f$ directly, and we calculate $F_{{f,b}}$ and $F_d$ using other constraints.")
    st.write(f"Also consider an expression representing the conservation of insoluble (non-carbonate) mass. ")
    st.latex(r"X_b F_b + {\color{olive}X_d F_d} = {\color{grey}X_c F_c} + {\color{red}X_f F_f} + {\color{teal}X_{dis} F_{dis}}")
    st.write(f" where X represents the fraction of mass flux that is insoluble. Note that **$X_{{dis}}$** is 0, by definition (all dissolved material is soluble), so that term goes to 0. For each other component, the fraction of insoluble material can be determined by bulk geochemistry. ")
    st.write(f"By solving both the mass balance and the insoluble fraction mass balance for $F_d$, and setting them equal to each other, we arrive at an expression for $F_{{dis}}$. Then, $F_d$ can be found using the mass balance equation.")
    st.write(f"Dissolved flux: ")
    st.latex(r"{\color{teal}F_{dis}} = ({\color{grey}X_{c} F_{c}} + {\color{red} X_{f} F_{f}} - X_{b} F_{b})/{\color{olive}X_{d}}  - {\color{red}F_{f}} - {\color{grey}F_{c}} + F_{b} ")

with rc:

    diag_name = r"/mount/src/soilsmassbalance/data_sources/Figure_Fluxes_Concepts_from_INKSCAPE_GSAPoster2.png"
    st.image(diag_name, output_format = 'PNG',width = 400, caption="Conceptual diagram")



st.markdown("""
    <style>
    
           /* Remove blank space at top and bottom */ 
           .block-container {
               padding-top: 2.5rem;
               padding-bottom: 0rem;
            }
           
            /* moves menu up a little */
            [data-testid="stSidebarNavItems"] {
            padding-top: 3rem;
            }
    </style>
    """, unsafe_allow_html=True)

#[data-testid="stSidebarNavItems"] { data-testid="stSidebarNavItems"