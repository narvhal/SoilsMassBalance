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

def show_overview():
    # if tabb == "Overview":
    # Title
    
    st.subheader("Interactive Carbonate Regolith Mass Balance Model")
    # st.write("This work is based on the work done for the dissertation: ")


    st.write("This is an interactive supplement to my Geological Society of America poster. ")
    st.write("Explore the effects of input variables on the dust and dissolved material fluxes. ")
    st.write("Questions/Comments? Get in touch! nari.v.miller   a   t   gmail.com")


    # lc, rc = st.columns([0.3, 0.7])
    # with lc:

    # with rc:

    lc, rc = st.columns([0.5, 0.5])
    with rc:
        url = r"/mount/src/soilsmassbalance/data_sources/GSA_2024_poster_NMiller_fontsfixed.pdf"
        pdf_viewer(url, width = 900)
        with open(url, "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label ="Download GSA 2024 Poster",
                            data=PDFbyte,
                            file_name="NMiller_GSA_2024.pdf",
                            mime='application/octet-stream')    


    with lc:
        st.subheader(f"**Summary of Poster**")

        # with st.expander(f"**Introduction**"):
        st.write(f"**Introduction**")
        st.write(f"Carbonate rocks can erode mechanically and through dissolution. In humid environments, carbonate rocks form landforms unique to chemical erosion. However, the extent to which chemical erosion acts in semi-arid and arid environments is poorly understood. Significant chemical erosion may reduce sediment flux, decoupling it from bedrock erosion. Another important factor in arid environments, windblown dust may enhance sediment flux as it accumulates. This study quantifies the mass fluxes of hilltop regolith to test whether chemical erosion and dust accumulation are necessary to reconcile the mass balance. ")

        st.write(f"**Methods**")
        st.write(f"A mass balance quantifies each component into and out of the volume of interest; here, a unit-volume patch on a hilltop.  In this work we consider the fluxes that compose regolith, the layer of mobile sediment on hillslopes. Flux, F (g/m2/yr), represents the amount of material moving across the boundaries of the unit-volume hilltop patch of regolith.")
        st.write(f"Mechanical weathering of bedrock produces coarse (>2 mm) lithic clasts.  Windblown dust, accumulating in the regolith, adds to the fine sediment fraction ($F_{{fines}}$ = $F_{{dust}}$ + $F_{{fines, bedrock}}$).")

        st.write(f"**Sample Sites**")
        st.write(f"Two carbonate hillslopes were chosen: one in arid south-eastern Arizona and one in semi-arid zone south-eastern Spain.")
        # with st.expander(f"**Conclusions**"):
        st.write(f"**Conclusions**")
        st.write(f"For both arid and semi-arid sites, the total mass flux is composed of more than 60% dissolved material. Even in arid and semi-arid regions, carbonate rocks may be subject to significant chemical erosion. Future work could identify whether additional chemical erosion occurs during clast transport, or if mineral precipitation inhibits transport out of the hillslope system (McFadden, 2013). ")
        st.write(f"The proportion of dissolved flux at the arid site is less than at the other site.  Supports a direct correlation between chemical erosion and climate. More work quantifying chemical erosion in carbonate soils across aridity gradients would help test this hypothesis. ")
        st.write(f"Dust flux was greater at the arid site. Besides climate, different local and regional dust sources, as well as variations in topography, might affect local dust accumulation (McClintock et al., 2015).  ")


