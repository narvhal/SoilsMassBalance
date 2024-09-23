import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
from io import BytesIO

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
    fn = r"https://github.com/narvhal/SoilsMassBalance/raw/main/data_sources/GSA_2024_poster_NMiller3.pdf"
          # "https://github.com/narvhal/SoilsMassBalance/blob/main/data_sources/GSA_2024_poster_NMiller3.pdf
else:
    fn = r"C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance\data_sources\GSA_2024_poster_NMiller3.pdf"


#displayPDF(fn)#
st.image(fn)
