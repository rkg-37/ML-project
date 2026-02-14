from langchain_ollama import ChatOllama  , OllamaEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.callbacks.base import BaseCallbackHandler
import pandas as pd
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter
import os
import tempfile


ollama_endpoint = "http://127.0.0.1:11434"
ollama_model = "llama3.2:latest"
embedding_model = "granite-embedding:278m"
ollama = ChatOllama(model=ollama_model, base_url=ollama_endpoint)

st.set_page_config(page_title="RAG App", page_icon=":robot_face:", layout="wide")
st.title("RAG App with LangChain and Ollama")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    """Configure the retriever by loading documents, splitting them, and creating a Chroma vector store.
    Returns:
        Chroma: The configured Chroma vector store retriever.
    """
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        tempfile_path = os.path.join(temp_dir.name, file.name)

        with open(tempfile_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyMuPDFLoader(tempfile_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    embedding = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(split_docs, embedding=embedding, collection_name="rag_collection")
    return vectorstore.as_retriever()


class StreamHandler(BaseCallbackHandler):
    """Custom callback handler to stream responses to Streamlit."""
    
    def __init__(self,container , initial_text):
        self.container = container  # Container to hold the chat messages
        self.text = initial_text  # Initial text to display while streaming


    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Append new tokens to the chat history in real-time."""
        self.text += token 
        self.container.markdown(self.text)
        
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
if not uploaded_files:
    st.info("Please upload PDF files to continue.")
    st.stop()
    
retriever = configure_retriever(uploaded_files)

llm = ChatOllama(model=ollama_model, base_url=ollama_endpoint,callbacks=[StreamHandler(st.empty(), "")])

qa_template = """You are a helpful assistant that answers questions based on the following retrieved documents:
                {context}

                Question: {question}
                """
                
qa_prompt = ChatPromptTemplate.from_template(qa_template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_ragchain = (
    {
        "context" : itemgetter("question")
        |
        retriever
        |
        format_docs,
        "question" : itemgetter("question")
    }
        |
        qa_prompt
        |
        llm
)
    
streamlit_msg_history  = StreamlitChatMessageHistory(key  = "langchain_messages")

if (len(streamlit_msg_history.messages) == 0):
    streamlit_msg_history.add_ai_message("Please ask any question")
    
for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)
    
class postmessagehandler(BaseCallbackHandler):
    def __init__(self, msg):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []
        
    def on_retriever_end(self, documents, **kwargs) -> None:
        """Store retrieved documents for later use."""
        source_ids = []
        for doc in documents:
            metadata = {
                "source": doc.metadata['source'],
                "page": doc.metadata["page"],
                "content": doc.page_content[:100]
            }
            idx = (metadata["source"] ,metadata["page"])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, * , run_id , parent_run_id ,**kwargs) -> None:
        if len(self.sources):
            st.markdown("__sources__"+"\n")
            sources_df = pd.DataFrame(self.sources[:3])
            st.dataframe(sources_df)


if user_prompt:=st.chat_input():
    st.chat_message("human").write(user_prompt)
    with st.chat_message('ai'):
        stream_handler = StreamHandler(st.empty(), "")
        source_container = st.write("")
        pm_handler = postmessagehandler(source_container)
        config = {"callbacks":[stream_handler, pm_handler]}
        qa_ragchain.invoke({"question": user_prompt}, config=config)
