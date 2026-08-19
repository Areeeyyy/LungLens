import streamlit as st

def page_home(go_to_func):
    st.markdown(
        """
        <div class="hero">
            <h1>Detect Pneumonia with<br>CNN Approach Based<br>on X-Ray Images</h1>
            <p>Upload a chest X-ray image and let our Convolutional Neural Network
            analyze it for signs of pneumonia, fast, simple, and educational.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns([3, 1.2, 1.2, 3])
    with col2:
        if st.button("Upload yours", key="hero_upload", use_container_width=True):
            go_to_func("upload")
    with col3:
        if st.button("How It Works?", key="hero_hiw", use_container_width=True):
            go_to_func("how_it_works")

    st.markdown(
        """
        <div class="disclaimer" style="max-width:520px; margin:2rem auto;">
            <strong>⚠️ Disclaimer:</strong> This tool is intended for educational and research
            purposes only. It is <strong>not</strong> a substitute for professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )
