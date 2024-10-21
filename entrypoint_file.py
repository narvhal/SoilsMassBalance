
import streamlit as st
pages = {
    " ": [
        st.Page("main_page.py", title="About Me")
    ],
    "Web Apps": [
        st.Page("pages/page_app_w_recalc.py", title="Interactive Mass Balance Model"),
    ],
}

pg = st.navigation(pages)
pg.run()