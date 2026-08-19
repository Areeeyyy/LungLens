import os
from pathlib import Path
import streamlit as st

from pages.home import page_home
from pages.upload import page_upload
from pages.how_it_works import page_how_it_works

# ============================================================
# Configuration
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Page config MUST be the first Streamlit call
st.set_page_config(
    page_title="LungLens, Pneumonia Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load CSS stylesheet
css_path = Path(__file__).parent / "styles" / "main.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================================
# Navigation state
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page: str):
    st.session_state.page = page


def render_navbar():
    cols = st.columns([6, 1, 1, 1.5])
    with cols[0]:
        st.markdown(
            '<p style="font-size:1.5rem; font-weight:900; letter-spacing:-0.5px; '
            'margin:0; padding:0.5rem 0; color:var(--text-primary);">LungLens</p>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button("Home", key="nav_home", use_container_width=True):
            go_to("home")
    with cols[2]:
        if st.button("Upload", key="nav_upload", use_container_width=True):
            go_to("upload")
    with cols[3]:
        if st.button("How It Works?", key="nav_hiw", use_container_width=True):
            go_to("how_it_works")
    st.markdown(
        '<hr style="margin:0 0 2rem; border:none; border-top:1.5px solid var(--border);">',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <p>
                <strong>LungLens</strong> · Built for educational &amp; research purposes<br>
                Convolutional Neural Network · Chest X-Ray Classification
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    render_navbar()

    page = st.session_state.page
    if page == "home":
        page_home(go_to)
    elif page == "upload":
        page_upload()
    elif page == "how_it_works":
        page_how_it_works()

    render_footer()


if __name__ == "__main__":
    main()
