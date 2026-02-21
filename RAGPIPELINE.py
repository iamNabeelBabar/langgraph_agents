import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


class RAGPipeline:
    
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        self.pc = Pinecone(api_key=pinecone_api_key)
        
    def ingest(self, pdf_path: str, index_name: str, chunk_size: int = 800, chunk_overlap: int = 80) -> str:
        docs = self._load_pdf(pdf_path)
        chunked_docs = self._split_and_clean(docs, chunk_size, chunk_overlap)
        message = self._create_vectorstore(chunked_docs, pdf_path, index_name)
        return message
    
    def _load_pdf(self, pdf_path: str) -> list:
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def _split_and_clean(self, docs: list, chunk_size: int, chunk_overlap: int) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        chunked_docs = splitter.split_documents(docs)
        processed_docs = []
        
        for doc in chunked_docs:
            text = doc.page_content
            text = re.sub(r"\n+", " ", text)
            text = re.sub(r"\d+\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            
            cleaned_doc = Document(
                page_content=text,
                metadata={
                    "source": doc.metadata["source"],
                    "page_no": doc.metadata["page"] + 1
                }
            )
            processed_docs.append(cleaned_doc)
            
        return processed_docs
    
    def _create_vectorstore(self, docs: list, pdf_path: str, index_name: str) -> str:
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        file_name = pdf_path.split("/")[-1]
        first_two_words = " ".join(file_name.split()[:2])
        namespace = first_two_words.lower().replace(" ", "_")
        
        embed_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
            api_key=self.openai_api_key
        )
        
        PineconeVectorStore.from_documents(
            docs,
            embed_model,
            index_name=index_name,
            namespace=namespace
        )
        
        index = self.pc.Index(index_name)
        stats = index.describe_index_stats()
        vector_count = stats['total_vector_count']
        
        return f"Vectorstore successfully created! {index_name} has {vector_count} vectors"


# Usage
pipeline = RAGPipeline(
    openai_api_key="your_openai_key",
    pinecone_api_key="your_pinecone_key"
)

result = pipeline.ingest(
    pdf_path="document.pdf",
    index_name="my-index",
    chunk_size=800,
    chunk_overlap=80
)

print(result)