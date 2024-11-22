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

#  \textcolor{#808080}   Fb
#  \textcolor{#deb887}   Fd
#  \textcolor{#bc8f8f}   Fc
#  \textcolor{#cd5c5c}   Ffbr
#  \textcolor{#e0ffff}   Fdis



cc, rc = st.columns([0.61,0.39])
with cc:
    st.subheader(f"**Conservation of Mass**")

    st.write(f"Mass Balance Including dust, constrained by insoluble material mass balance")
    st.latex(r"\LARGE F_b + \textcolor{#deb887}{F_d} = \textcolor{#bc8f8f}{F_c} + \textcolor{#cd5c5c}{F_f} + {\color{teal}F_{dis}}")
    st.write(f"where **$F_b$** is bedrock mass flux, **$F_d$** is dust mass flux, **$F_c$** is coarse fraction of sediment mass flux, **$F_{{dis}}$** is dissolved material mass flux, and **$F_f$** is the entire fine fraction of sediment. ")
    st.write(f"Note that:  ")
    st.latex( r"\LARGE \textcolor{#cd5c5c}{F_f} = {\color{purple}F_{f,b}} + \textcolor{#deb887}{F_d} ")
    st.write(f"where **$F_{{f,b}}$** is insoluble material derived from dissolved bedrock. The technique used here quantifies $F_f$ directly, and we calculate $F_{{f,b}}$ and $F_d$ using other constraints.")
    st.subheader(f"**Conservation of Non-carbonate Mineral Mass**")

    st.write(f"We can also expect conservation of insoluble (non-carbonate) mass. ")
    st.latex(r"\LARGE X_b F_b + \textcolor{#deb887}{X_d F_d} = \textcolor{#bc8f8f}{X_c F_c} + \textcolor{#cd5c5c}{X_f F_f} + {\color{teal}X_{dis} F_{dis}}")
    st.write(f" where X represents the fraction of mass flux that is insoluble. Note that **$X_{{dis}}$** is 0, by definition (all dissolved material is soluble), so that term goes to 0. For each other component, the fraction of insoluble material can be determined by bulk geochemistry. ")
    st.write(f"By solving both the mass balance and the insoluble fraction mass balance for $F_d$, and setting them equal to each other, we arrive at an expression for $F_{{dis}}$. Then, $F_d$ can be found using the mass balance equation.")
    st.write(f"Dissolved flux: ")
    st.latex(r"\LARGE {\color{teal}F_{dis}} = (\textcolor{#bc8f8f}{X_{c} F_{c}} + \textcolor{#cd5c5c}{ X_{f} F_{f}} - X_{b} F_{b})/\textcolor{#deb887}{X_{d}}  - \textcolor{#cd5c5c}{F_{f}} - \textcolor{#bc8f8f}{F_{c}} + F_{b} ")

with rc:
    st.write(" ")
    st.write(" ")

    diag_name = r"/mount/src/soilsmassbalance/data_sources/Figure_Fluxes_Concepts_from_INKSCAPE_GSAPoster2.svg"
    st.image(diag_name,width = 400, caption="Conceptual diagram")


run_mkdn_style()

