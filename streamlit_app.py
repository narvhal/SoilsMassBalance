import os

import streamlit as st
from streamlit_navigation_bar import st_navbar

import pages as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_functions.make_flux_boxes import *
from io import BytesIO, StringIO
import requests
import base64
from streamlit_pdf_viewer import pdf_viewer
from streamlit_navigation_bar import st_navbar

st.set_page_config( layout="wide", initial_sidebar_state="collapsed")

pages = ["Overview",  "Mass Balance Equations", "Interactive App"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "data_sources", "F_logo.svg")
# urls = {"GitHub": ""}
# styles = {
#     "nav": {
#         "background-color": "grey",
#         "justify-content": "left",
#     },
#     "img": {
#         "padding-right": "14px",
#     },
#     "span": {
#         "color": "white",
#         "padding": "14px",
#     },
#     "active": {
#         "background-color": "white",
#         "color": "var(--text-color)",
#         "font-weight": "normal",
#         "padding": "14px",
#     }
# }
options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(
    pages,
    options=options,
    logo_path=logo_path,
)



#     styles=styles,
    # 
    # urls=urls,

functions = {
    pages[0]: pg.show_overview,
    pages[1]: pg.show_equations,
    pages[2]: pg.show_app,
}

go_to = functions.get(page)
if go_to:
    go_to()

if page == "Home":
    functions.get(pages[0])


