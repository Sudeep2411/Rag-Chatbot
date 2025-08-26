from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from typing import List, Dict, Optional
import os
from app.config import PERSIST_DIR, EMBEDDING_MODEL
from app.src.utils.logger import get_logger

logger = get_logger("LangChainWrapper")

class LangChainRAG:
    def __init__(self, persist_dir: str = PERSIST_DIR, embedding_model: str = EMBEDDING_MODEL):
        self.persist_dir = persist_dir
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model,
            collection_name="langchain_rag"
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Define prompt template
        self.prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Use the provided context to answer the user's question.
- Cite sources with [source] where source is the origin (filename, URL title, or page).
- If the answer is not in the context, say you don't know based on the provided documents.
- Be concise and factual.

Question: {question}

Context: {context}

Instructions:
- Use only the context to answer.
- Add citations like [{{source}}] inline where relevant.
- If context is insufficient, say so and suggest where to look next.

Answer:
""")
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | StrOutputParser()
        )
        
        logger.info("LangChain RAG wrapper initialized")
    
    def query(self, question: str):
        """Query the RAG system using LangChain"""
        try:
            logger.info(f"LangChain query: {question[:50]}...")
            result = self.rag_chain.invoke(question)
            logger.debug(f"LangChain response: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error in LangChain query: {e}")
            return f"Error processing query: {str(e)}"
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store using LangChain"""
        try:
            langchain_docs = []
            for doc in documents:
                metadata = doc.get('metadata', {})
                langchain_docs.append(
                    Document(
                        page_content=doc['content'],
                        metadata=metadata
                    )
                )
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(langchain_docs)
            logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
            
            # Add to vectorstore
            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()
            
            # Update retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            logger.info(f"Added {len(splits)} document chunks to vector store")
            return len(splits)
            
        except Exception as e:
            logger.error(f"Error adding documents with LangChain: {e}")
            return 0
    
    def get_document_count(self):
        """Get the number of documents in the vector store"""
        try:
            collection = self.vectorstore._collection
            if collection:
                return collection.count()
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def similarity_search(self, query: str, k: int = 4):
        """Perform similarity search and return results"""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', 0)  # Some vector stores include score
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
