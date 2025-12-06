import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/rag/query"


def layout():
    st.set_page_config(
        page_title="The YouTuber RAG",
        layout="wide",
    )

    st.title("The YouTuber RAG")

    st.markdown(
        """
        **This dashboard lets you ask questions based on a subset of transcribed YouTube videos.**
        """
    )

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    with col_left:
        st.subheader("Ask the YouTuber!")

        question = st.text_input(label="Input a question here:")

        send_clicked = st.button("Send")

        if send_clicked:
            if question.strip() == "":
                st.warning("Please enter a question before sending.")
            else:
                st.session_state.last_question = question
                st.session_state.last_response = None

                with st.spinner("Contacting The YouTuber RAG..."):
                    try:
                        response = requests.post(
                            API_URL,
                            json={"prompt": question},
                            timeout=60,
                        )
                    except requests.RequestException as e:
                        st.error(f"Could not contact API: {e}")
                    else:
                        if not response.ok:
                            try:
                                data = response.json()
                                detail = data.get("detail", response.text)
                            except Exception:
                                detail = response.text

                            st.error(f"API error ({response.status_code}): {detail}")
                        else:
                            st.session_state.last_response = response.json()

        if st.session_state.last_question and st.session_state.last_response:
            data = st.session_state.last_response

            st.markdown("### Question")
            st.markdown(st.session_state.last_question)

            st.markdown("### Answer")
            st.markdown(data.get("answer", "No answer returned from backend."))
        elif st.session_state.last_question and not st.session_state.last_response:
            st.markdown("### Question")
            st.markdown(st.session_state.last_question)

    with col_right:
        st.subheader("Sources (transcribed videos)")

        if st.session_state.get("last_response") is None:
            st.info("When you get an answer, the used sources will be shown here.")
        else:
            data = st.session_state.last_response
            sources = data.get("sources") or []

            if not sources:
                st.write("No sources were recorded for this answer.")
            else:
                for i, src in enumerate(sources, start=1):
                    video_id = src.get("video_id", "unknown")
                    title = src.get("title", "No title")
                    score = src.get("score", None)

                    st.markdown(f"{i}. {title}")
                    st.write(f"Video ID: `{video_id}`")
                    if score is not None:
                        st.write(f"Score: `{score}`")
                    st.markdown("---")


if __name__ == "__main__":
    layout()