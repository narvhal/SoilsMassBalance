import streamlit as st


def show_equations():
    lc, cc, rc = st.columns([0.1, 0.8, 0.1])
    with cc:
        st.subheader(f"**Mass Balance Equations**")

        st.write(f"Mass Balance Including dust, constrained by insoluble material mass balance")
        st.latex(r"F_b + {\color{olive}F_d} = {\color{grey}F_c} + {\color{red}F_{f}} + {\color{teal}F_{dis}}")
        st.write(f"where **$F_b$** is bedrock mass flux, **$F_d$** is dust mass flux, **$F_c$** is coarse fraction of sediment mass flux, **$F_{{dis}}$** is dissolved material mass flux, and **$F_f$** is the entire fine fraction of sediment. ")
        st.write(f"Note that:  ")
        st.latex( r"{\color{red}F_f} = {\color{purple}F_{f,b}} + {\color{olive}F_d}")
        st.write("where **$F_{{f,b}}$** is insoluble material derived from dissolved bedrock. The technique used here quantifies $F_f$ directly, and we calculate $F_{{f,b}}$ and $F_d$ using other constraints.")
        st.write(r"Also consider an expression representing the conservation of insoluble (non-carbonate) mass. ")
        st.latex(r"X_b F_b + {\color{olive}X_d F_d} = {\color{grey}X_c F_c} + {\color{red}X_f F_f} + {\color{teal}X_{dis} F_{dis}}")
        st.write(f" where X represents the fraction of mass flux that is insoluble. Note that **$X_{{dis}}$** is 0, by definition (all dissolved material is soluble), so that term goes to 0. For each other component, the fraction of insoluble material can be determined by bulk geochemistry. ")
        st.write(f"By solving both the mass balance and the insoluble fraction mass balance for $F_d$, and setting them equal to each other, we arrive at an expression for $F_{{dis}}$. Then, $F_d$ can be found using the mass balance equation.")
        st.write(f"Dissolved flux: ")
        st.latex(r"{\color{teal}F_{dis}} = ({\color{grey}X_{c} F_{c}} + {\color{red} X_{f} F_{f}} - X_{b} F_{b})/{\color{olive}X_{d}}  - {\color{red}F_{f}} - {\color{grey}F_{c}} + F_{b} ")