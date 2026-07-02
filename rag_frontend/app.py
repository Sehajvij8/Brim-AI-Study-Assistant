import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag_backend.pdf_loader import pdf_to_text
from rag_backend.Chunking import chunker
from rag_backend.vector_store import store_embeddings
from rag_backend.rag_pipeline import rag_pipeline
from rag_backend.Embeddings import generate_embeddings

# Page config settings
st.set_page_config(
    page_title = "Brim",
    page_icon = "🤖",
    layout = "wide"
)

#Session state settings
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Settings
with st.sidebar:
    st.title("🤖 Brim")
    st.markdown("---")

#Uploaded files settings
    uploaded_files = st.file_uploader("📄 Upload your PDF(s)",
                                      type = ["PDF"],
                                      accept_multiple_files= True)
    
    process_button = st.button("Process your PDFs")
    if process_button:
        if not uploaded_files:
            st.warning("⚠️ Please upload your one pdf atleast.")
        else:
            st.success("PDFs uploaded successfully.")
            with st.spinner("📚 Processing PDFs...."):
                for uploaded_file in uploaded_files:

                    #Temporary file create
                    with tempfile.NamedTemporaryFile(
                        delete = False,
                        suffix=".pdf"
                        ) as temp_file:
                        
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name
                        
                   

                    #Create chunks
                    chunks = chunker(temp_path)

                    #Create embeddings
                    embeddings = generate_embeddings(chunks)

                    # Store embeddings in Qdrant
                    store_embeddings(embeddings , chunks)

                    #Delete temporary file
                    os.remove(temp_path)


                #Check temporary path
                st.success(f"{len(uploaded_files)} PDFs processed successfully.")

# HTML page settings        
    st.markdown("---")
    
    st.markdown(
    """
    <div style="
        background-color:#FFFFFF;
        padding:15px;
        border-radius:12px;
        color:black;
    ">
    
    <h3 style="
        font-size:35px;
        margin-top:0;
        margin-bottom:15px;
    ">
        🤖 <b><I><U>Brim Features Support</b></I></U>
    </h3>

    ✅ Multiple PDF Support<br><br>

    ✅ Semantic Search<br><br>

    ✅ Groq LLM<br><br>

    ✅ RAG Powered

    </div>
    """,
    unsafe_allow_html=True
)

# Main title
st.title("🤖 Brim")
st.caption("Your Intelligent AI Study Assistant")
st.divider()


#Old chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Question Selection
question = st.text_input(
    "💬 Ask anything from your uploaded documents",
    placeholder="Example : What is Artificial Intelligence?"
)

col1 , col2 = st.columns([1,5])
with col1:
    ask_button = st.button(
        "Ask Brim 💬",
        use_container_width=True
        )


st.divider()
st.subheader("🤖Answer")

answer_placeholder = st.empty()
if ask_button:
    if question.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔Brim is thinking...."):
            #Saving user message
            st.session_state.messages.append({"role":"user" , "content":question})
            result = rag_pipeline(question , st.session_state.messages)
            if result:
                answer_placeholder.success(result["answer"])
                st.session_state.messages.append({"role":"assistant" , "content": result["answer"]})
            
                with st.expander("📄 Retrieved_context"):

                    for i , chunk in enumerate(result["chunks"] , start = 1):
                        st.markdown(f"###Chunk{i}")
                        st.write(f"**Source:**{chunk['source']}")
                        st.write(chunk["text"])
                        st.divider()
            else:
                answer_placeholder.error("❌ Failed to generate answer.")

        
            
