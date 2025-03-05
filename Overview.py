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

st.set_page_config( layout="wide" )

st.header("Nari Miller, Ph.D.")
st.subheader("Geomorphologist")

# st.header("Interactive Carbonate Regolith Mass Balance Model")
lc, rc = st.columns([0.8, 0.2])
with lc: 

    st.write(f"I defended my dissertation in geomorphology in July 2024. I am looking for a job that makes use of the varied skills I've developed in graduate school. I enjoy the creative aspects of science: combining data in new ways, addressing questions using different data visualizations, and discussions with others. I'm motivated by work that involves a component of service, such as science education or hazards research. ")

with rc:
    st.image(r"/mount/src/soilsmassbalance/data_sources/website_face.jpg")

url = r"/mount/src/soilsmassbalance/data_sources/Miller_N_CV_2024.pdf"

with open(url, "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label ="Download CV",
                    data=PDFbyte,
                    file_name="Miller_N_CV_2024.pdf",
                    mime='application/octet-stream')    

st.subheader(f"**Recent Work**")

st.write(f"I studied arid and semi-arid landscapes composed of carbonate rocks, which erode mechanically and chemically via dissolution. Chemical erosion increases with precipitation, but the magnitude and controls on this process are still being worked out. There are dramatic and unique karst features in high-precipitation zones, but it is unclear whether dissolution continues to shape carbonate landscapes at lower ranges of precipitation. My current research stems from an interest in how the total mass of sediment from carbonate hillslopes has been reduced by chemical erosion and augmented by windblown dust. ")

st.write(f"I test whether carbonate landscapes in arid and semi-arid regions are shaped by chemical erosion. A first-order effect of chemical erosion is the reduction of sediment produced on the landscape that requires transport by gravity-driven forces. I compared new and previously published erosion rates to topographic metrics to assess whether there was a signal of lower sediment flux in regions with higher precipitation. I planned and executed research to quantify a mass balance of hillslope sediments, which allowed me to independently quantify chemical erosion. ")

    st.write(f"Most models of sediment production and transport ignore chemical erosion and dust accumulation. However, I found the mass balances for both sites – semi-arid Spain and more arid Arizona – indicate 1) that more than half of the eroded material (>60% of total mass flux) may be dissolved, which would significantly change the character of sediment deposits in the catchment, and 2) dust accumulation (~10%) is necessary to balance the mass fluxes. Since graduating, I constructed an interactive webapp to show how these results change with different inputs (this site). I have presented this work at the Geological Society of America conference in Anaheim, and am preparing it for publication. ")

## Tell me more ! expansion. 
# st.write(f"Much of my graduate work was supported by a grant awarded to an interdisciplinary group of geomorphologists, archeologists, and fire scientists. I did fieldwork in southeast Spain alongside a group of archeologists from ASU and University of Valencia, who were primarily studying the charcoal and pollen in layered sedimentary deposits. Carbonate rocks like limestone can chemically erode (i.e. dissolve) when exposed to rainwater; caves and cenotes are examples of landforms shaped by dissolution. ")

st.write(f"I also constructed a mass balance model which solves for dissolved material and dust influx, which I implemented at an arid and a semi-arid site.")

st.subheader(f"The rest of this website")

st.write("This is an interactive supplement to my Geological Society of America poster. ")
st.write("Explore the effects of input variables on the dust and dissolved material fluxes. ")
st.write("Questions/Comments? Get in touch! nari.v.miller@gmail.com")

lc, rc = st.columns([0.5, 0.5])

with lc:
    st.subheader(f"**Research Summary**")

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


with rc:
    url = r"/mount/src/soilsmassbalance/data_sources/GSA_2024_poster_NMiller_fontsfixed.pdf"
    # pdf_viewer(url, width = 900)

    st.image(r"/mount/src/soilsmassbalance/data_sources/GSA_2024_poster_NMiller3.png")
    
    with open(url, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label ="Download GSA 2024 Poster",
                        data=PDFbyte,
                        file_name="NMiller_GSA_2024.pdf",
                        mime='application/octet-stream')    


run_mkdn_style()
