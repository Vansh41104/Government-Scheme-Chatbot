import streamlit as st





def main():
    st.set_page_config(page_title="Government Scheme Chatbot", page_icon="🤖", layout="wide")

    st.header("Government Scheme Chatbot")
    st.text_input("Hello I am your personalised bot which will tell you about the government schemes which you would like to know about")
    with st.sidebar:
        st.subheader("About")
        st.file_uploader("Upload file")
        st.button("Submit")

    


if __name__ == '__main__':
    main()