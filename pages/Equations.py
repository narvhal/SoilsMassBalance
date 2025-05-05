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

cc, rc = st.columns([0.65,0.35])
with cc:
    with st.container(height=650):
        st.header(f"**Mass Balance Research Summary**")

        # with st.expander(f"**Introduction**"):
        st.subheader(f"**Introduction**")
        st.write(f"Carbonate rocks can erode mechanically and through dissolution. In humid environments, carbonate rocks form landforms unique to chemical erosion. However, the extent to which chemical erosion acts in semi-arid and arid environments is poorly understood. Significant chemical erosion may reduce sediment flux, decoupling it from bedrock erosion. Another important factor in arid environments, windblown dust may enhance sediment flux as it accumulates. This study quantifies the mass fluxes of hilltop regolith to test whether chemical erosion and dust accumulation are necessary to reconcile the mass balance. ")

        st.subheader(f"**Methods**")
        st.write(f"A mass balance quantifies each component into and out of the volume of interest; here, a unit-volume patch on a hilltop.  In this work we consider the fluxes that compose regolith, the layer of mobile sediment on hillslopes. Flux, F (g/m2/yr), represents the amount of material moving across the boundaries of the unit-volume hilltop patch of regolith.")
        st.write(f"Mechanical weathering of bedrock produces coarse (>2 mm) lithic clasts.  Windblown dust, accumulating in the regolith, adds to the fine sediment fraction ($F_{{fines}}$ = $F_{{dust}}$ + $F_{{fines, bedrock}}$).")

        st.subheader(f"**Conservation of Mass Equations**")

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


        st.subheader(f"**Sample Sites**")
        st.write(f"Two carbonate hillslopes were chosen: one in arid south-eastern Arizona and one in semi-arid zone south-eastern Spain.")
        # with st.expander(f"**Conclusions**"):
        st.subheader(f"**Conclusions**")
        st.write(f"For both arid and semi-arid sites, the total mass flux is composed of more than 60% dissolved material. Even in arid and semi-arid regions, carbonate rocks may be subject to significant chemical erosion. Future work could identify whether additional chemical erosion occurs during clast transport, or if mineral precipitation inhibits transport out of the hillslope system (McFadden, 2013). ")
        st.write(f"The proportion of dissolved flux at the arid site is less than at the other site.  Supports a direct correlation between chemical erosion and climate. More work quantifying chemical erosion in carbonate soils across aridity gradients would help test this hypothesis. ")
        st.write(f"Dust flux was greater at the arid site. Besides climate, different local and regional dust sources, as well as variations in topography, might affect local dust accumulation (McClintock et al., 2015).  ")

        st.write(" ")

        url = r"/mount/src/soilsmassbalance/data_sources/GSA_2024_poster_NMiller_fontsfixed.pdf"
        # pdf_viewer(url, width = 900)

        st.image(r"/mount/src/soilsmassbalance/data_sources/GSA_2024_poster_NMiller3.png")
        
        with open(url, "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label ="Download GSA 2024 Poster",
                            data=PDFbyte,
                            file_name="NMiller_GSA_2024.pdf",
                            mime='application/octet-stream')    

with rc:

    st.write(" ")

    diag_name = r"/mount/src/soilsmassbalance/data_sources/Figure_Fluxes_Concepts_from_INKSCAPE_GSAPoster2.svg"
    st.image(diag_name,width = 300, caption="Conceptual diagram. The mass input of material from bedrock and dust is balanced by the outflux of dissolved material, and mechanically eroded material. ")

run_mkdn_style()

