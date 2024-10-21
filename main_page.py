# MAin page in SoilsMassBalance
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
from io import BytesIO



st.set_page_config(layout="wide")
# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`

st.subheader("Nari Miller")

lc, rc = st.columns([0.75, 0.25])
with lc:
    # st.write("I like investigating open-ended problems, integrating disparate data sources, and making meaningful contributions. I am looking for a job that meets one or more of those . ")
    # st.write("This website is under construction!")
    st.write("I'm looking for a temporary job or research position, while I write a few fellowship applications to establish what is next. ")
    st.write("Feel free to contact me if you have questions or know of an interesting opportunity.")
    st.write("A paper is in the works, but for now, my 2024 GSA poster presents my work concisely.")
    # st.write("I am looking for a job that meets one or more of those . ")
st.write("Dissertation (July 2024) available here: https://keep.lib.asu.edu/items/195269")
st.write("nari.v.miller@gmail.com")

st.write(" ")
st.write("GSA 2024 Poster (to zoom in, you need to download it by right-clicking --> save as):")
# def displayPDF(file):
#     # Opening file from file path
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display =  f"""<embed
#     class="pdfobject"
#     type="application/pdf"
#     title="Embedded PDF"
#     src="data:application/pdf;base64,{base64_pdf}"
#     style="overflow: auto; width: 100%; height: 100%;">"""

#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)


flag_gh = True  #  st.checkbox("Flag_gh ? ", key = "flag_gh_key", value = True)

if flag_gh:


    fn = r"https://github.com/narvhal/SoilsMassBalance/blob/main/data_sources/GSA_2024_poster_NMiller3.png?raw=true"
          #https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/GSA_2024_poster_NMiller3.png"
          # "https://github.com/narvhal/SoilsMassBalance/blob/main/data_sources/GSA_2024_poster_NMiller3.pdf
else:
    fn = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\GSA_2024_poster_NMiller3.pdf"


#displayPDF(fn)#
st.image(fn)

