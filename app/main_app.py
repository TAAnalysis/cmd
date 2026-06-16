"""
Campus Orientation Assistant - Streamlit Application
Supports three input modes:
  1. Image upload (CLIP + FAISS retrieval)
  2. Voice recording / audio file upload (Whisper transcription)
  3. Typed text query (DistilBERT intent classification + KB entity matching)

Output panel displays the matched location's name, description,
opening hours, events, and a map reference.

Run (inside Docker):
    streamlit run app/main_app.py
"""

import os
import sys
import tempfile

import streamlit as st
from PIL import Image

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fusion import CampusAssistantFusion  # noqa: E402


st.set_page_config(
    page_title="Campus Orientation Assistant",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_fusion_pipeline():
    return CampusAssistantFusion()


def display_response(result):
    """Render the output panel for a fusion result dict."""
    if not result.get("found", False):
        st.warning(result.get("message", "No matching location found."))
        if result.get("transcript"):
            st.caption(f"Transcribed query: \"{result['transcript']}\"")
        if result.get("intent"):
            st.caption(f"Detected intent: {result['intent']} "
                       f"(confidence: {result.get('intent_confidence', 0):.2f})")
        return

    st.success(result["message"])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(result["name"])
        st.markdown(f"**Category:** {result['category'].title()}")
        st.markdown(f"**Description:** {result['description']}")
        st.markdown(f"**Opening Hours:** {result['opening_hours']}")
        st.markdown(f"**Map Reference:** {result['map_reference']}")

        if result.get("events"):
            st.markdown("**Upcoming Events:**")
            for event in result["events"]:
                st.markdown(f"- **{event['name']}** — {event['datetime']} "
                             f"({event['details']})")
        else:
            st.markdown("**Upcoming Events:** None scheduled")

    with col2:
        gps = result.get("gps_coordinates")
        if gps:
            st.markdown("**GPS Coordinates**")
            st.map(data=[{"lat": gps["lat"], "lon": gps["lng"]}], zoom=15)

    # Diagnostics expander
    with st.expander("Show diagnostics (modality evidence)"):
        if result.get("transcript"):
            st.write(f"**Transcribed query:** {result['transcript']}")
        if result.get("intent"):
            st.write(f"**Detected intent:** {result['intent']} "
                     f"(confidence: {result.get('intent_confidence', 0):.2f})")
        if result.get("source"):
            st.write(f"**Resolution source:** {result['source']}")
        if result.get("modality_evidence"):
            st.json(result["modality_evidence"])


def main():
    st.title("🎓 Campus Orientation Assistant")
    st.caption("Liberty University Edition — find buildings, check hours, and discover events "
               "using an image, your voice, or a typed question.")

    with st.spinner("Loading models (CLIP, DistilBERT, Whisper)... this may take a minute on first run."):
        fusion = load_fusion_pipeline()

    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎙️ Voice Query", "⌨️ Text Query"])

    # ------------------------------------------------------------
    # Tab 1: Image upload
    # ------------------------------------------------------------
    with tab1:
        st.subheader("Upload a photo of a campus building")
        uploaded_image = st.file_uploader(
            "Choose an image", type=["png", "jpg", "jpeg"], key="image_upload"
        )
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            col_img, col_result = st.columns([1, 2])
            with col_img:
                st.image(image, caption="Uploaded image", use_column_width=True)
            with col_result:
                with st.spinner("Identifying location..."):
                    result = fusion.answer(image=image)
                display_response(result)

    # ------------------------------------------------------------
    # Tab 2: Voice query
    # ------------------------------------------------------------
    with tab2:
        st.subheader("Upload a voice query (MP3/WAV)")
        uploaded_audio = st.file_uploader(
            "Choose an audio file", type=["mp3", "wav", "m4a"], key="audio_upload"
        )
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            # Save to a temp file for Whisper
            suffix = os.path.splitext(uploaded_audio.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_audio.read())
                tmp_path = tmp.name

            with st.spinner("Transcribing and processing your query..."):
                result = fusion.answer(audio_path=tmp_path)

            os.unlink(tmp_path)

            if result.get("transcript"):
                st.info(f"**Transcribed:** \"{result['transcript']}\"")

            display_response(result)

    # ------------------------------------------------------------
    # Tab 3: Text query
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Type your question")
        text_query = st.text_input(
            "e.g. \"Where is the Jerry Falwell Library?\" or "
            "\"Is the cafeteria open on Sundays?\"",
            key="text_query",
        )
        if st.button("Ask", key="text_submit") and text_query.strip():
            with st.spinner("Processing your query..."):
                result = fusion.answer(text=text_query)
            display_response(result)

    # ------------------------------------------------------------
    # Sidebar: about / info
    # ------------------------------------------------------------
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This assistant combines:\n"
            "- **CLIP + FAISS** for image-based building recognition\n"
            "- **Whisper** for speech-to-text transcription\n"
            "- **Fine-tuned DistilBERT** for intent classification\n"
            "- A structured **knowledge base** of 15 campus locations\n"
        )
        st.markdown("---")
        st.markdown("**Example queries:**")
        st.markdown(
            "- Where is the Jerry Falwell Library?\n"
            "- Is the LaHaye Recreation Center open on weekends?\n"
            "- What events are happening at the Montview Student Union today?\n"
        )


if __name__ == "__main__":
    main()
