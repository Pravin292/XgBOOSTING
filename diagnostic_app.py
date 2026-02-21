import streamlit as st
import sys
import os
import traceback

# MUST BE FIRST
st.set_page_config(page_title="Project Diagnostic", layout="wide")

st.title("🛠 System Diagnostic Mode")

try:
    st.write("### 📂 Filesystem Check")
    files = os.listdir('.')
    st.write(f"Root files: {files}")
    
    st.write("### 📦 Dependency Modules")
    import numpy
    import pandas
    import joblib
    import sklearn
    st.success("Core ML libraries loaded successfully.")
    
    st.write("### 🧩 Local Utils Check")
    if os.path.exists("utils.py"):
        try:
            import utils
            st.success("utils.py detected and importable.")
        except Exception as e:
            st.error(f"Failed to import utils.py: {e}")
            st.code(traceback.format_exc())
    else:
        st.error("utils.py is MISSING from root folder.")

    st.write("### 🤖 Model Artifacts")
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    st.write(f"Found models: {pkl_files}")
    
    if len(pkl_files) == 0:
        st.warning("No .pkl models found. You may need to run model_training.py locally.")

except Exception as global_e:
    st.error(f"Global Boot Failure: {global_e}")
    st.code(traceback.format_exc())

st.divider()
st.info("If all checks above are green, the issue is likely within the specific app UI logic.")
