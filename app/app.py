import sys
import os
from pathlib import Path

# Get the absolute path to the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  
sys.path.insert(0, str(project_root))

# Now import your modules
import streamlit as st

# Import your custom modules using absolute path
from app.src.utils.rag import RAGPipeline
from app.config import TOP_K, DATA_DIR
from app.src.monitoring.feedback import FeedbackLogger

st.set_page_config(
    page_title="RAG Chatbot", 
    page_icon="💬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💬 RAG Chatbot with Document Intelligence")
st.caption("Ask questions about your documents and get AI-powered answers with source citations")

@st.cache_resource
def load_rag():
    return RAGPipeline()

@st.cache_resource
def load_feedback_logger():
    return FeedbackLogger()

rag = load_rag()
feedback_logger = load_feedback_logger()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model settings
    top_k = st.slider("Number of document chunks to retrieve", 1, 10, TOP_K, 1)
    
    st.markdown("---")
    st.header("📊 Feedback")
    
    # Feedback section
    if st.session_state.get('last_response'):
        st.subheader("Rate the last response")
        rating = st.slider("Rating (1-5 stars)", 1, 5, 3, key="rating_slider")
        feedback_text = st.text_area("Additional feedback", key="feedback_text")
        
        if st.button("Submit Feedback", key="submit_feedback"):
            rag.log_feedback(
                question=st.session_state.last_question,
                answer=st.session_state.last_response['answer'],
                sources=st.session_state.last_response['sources'],
                rating=rating,
                user_feedback=feedback_text
            )
            st.success("✅ Feedback submitted! Thank you.")
            # Clear the feedback fields
            st.session_state.rating_slider = 3
            st.session_state.feedback_text = ""
    
    st.markdown("---")
    st.header("📁 Document Management")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        data_dir = DATA_DIR
        os.makedirs(data_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success(f"📄 Saved {len(uploaded_files)} files. Run ingestion to process them.")
    
    # URL input
    st.subheader("Add URLs")
    url_input = st.text_input("Enter URL to add to urls.txt")
    if st.button("Add URL") and url_input:
        urls_path = os.path.join(DATA_DIR, 'urls.txt')
        with open(urls_path, 'a') as f:
            f.write(f"{url_input}\n")
        st.success(f"✅ URL added to {urls_path}")
    
    # Stats
    st.markdown("---")
    st.header("📈 Statistics")
    
    try:
        stats = feedback_logger.get_feedback_stats()
        st.metric("Total Feedback", stats['total_feedback'])
        st.metric("Average Rating", f"{stats['average_rating']}/5")
        st.metric("Text Feedback", stats['text_feedback_count'])
    except:
        st.info("No feedback statistics available yet")

# Main chat interface
st.header("💭 Chat with your documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask anything about your documents...")
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                response = rag.answer(user_input, top_k=top_k)
                
                # Display answer
                st.markdown(response['answer'])
                
                # Display sources
                with st.expander("📚 View Sources"):
                    st.text(response['sources_text'])
                
                # Store for feedback
                st.session_state.last_response = response
                st.session_state.last_question = user_input
                
                # Add to chat history
                chat_response = f"{response['answer']}\n\n**Sources:**\n{response['sources_text']}"
                st.session_state.chat_history.append({"role": "assistant", "content": chat_response})
                
            except Exception as e:
                error_msg = f"❌ Error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
### 🚀 How to use:
1. **Upload documents** in the sidebar (PDF/TXT)
2. **Add URLs** to scrape web content
3. **Ask questions** about your documents
4. **Rate responses** to help improve the system

### 🔄 To process new documents:
Run the ingestion script: `python -m app.src.ingestion.ingest`
""")
