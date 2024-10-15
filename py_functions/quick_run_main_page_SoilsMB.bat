@ECHO OFF
call "C:\ProgramData\miniconda3\Scripts\activate.bat" "C:\ProgramData\miniconda3"
call conda activate py3streamlit
cd /d C:\Users\nariv\OneDrive\JupyterN\streamlit_local\SoilsMassBalance
python -m streamlit run main_page.py  
PAUSE
