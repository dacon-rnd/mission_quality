import os

import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
LLM_MODEL = 'gpt-3.5-turbo-0125'

#PDF 문서에서 텍스트를 추출
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#지정된 조건에 따라 주어진 텍스트를 더 작은 덩어리로 분할
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(text)
    return chunks

#주어진 텍스트 청크에 대한 임베딩을 생성하고 FAISS를 사용하여 벡터 저장소를 생성
def get_vectorstore(text_chunks):
    # embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
    return vectorstore

#주어진 벡터 저장소로 대화 체인을 초기화
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)  #ConversationBufferWindowMemory에 이전 대화 저장
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name=LLM_MODEL),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
        memory=memory
    ) #ConversationalRetrievalChain을 통해 langchain 챗봇에 쿼리 전송
    return conversation_chain

# ========== ① 문서로드 ========== #
# PDF 파일 로드
def remove_pdf_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def pdf_loader(file_name):
    loader = PyPDFLoader(file_name)
    document = loader.load()
    remove_pdf_file(file_name)
    return document

# ========== ② 문서분할 ========== #

# 스플리터 지정
def split_document(document):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",  # 분할기준
        chunk_size=3000,   # 사이즈
        chunk_overlap=500, # 중첩 사이즈
    )

    # 분할 실행
    split_docs = text_splitter.split_documents(document)
    # 총 분할된 도큐먼트 수
    print(f'총 분할된 도큐먼트 수: {len(split_docs)}')

    return split_docs

def save_pdf_file(uploaded_file):
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
    return temp_file

user_upload = st.file_uploader("파일을 업로드해주세요~")

if user_upload is not None:
    if st.button("Upload"):
        with st.spinner("처리중.."):
            pdf_file_name = save_pdf_file(user_upload)
        with st.spinner('Document를 읽어들이는 중입니다...'):
            document = pdf_loader(pdf_file_name)
            split_docs = split_document(document)
            # PDF 텍스트 저장을 위해 FAISS 벡터 저장소 만들기
            vectorstore = get_vectorstore(split_docs)
            # 대화 체인 만들기
            st.session_state.conversation = get_conversation_chain(vectorstore)

if user_query := st.chat_input("질문을 입력해주세요~"):
    # 대화 체인을 사용하여 사용자의 메시지를 처리
    if 'conversation' in st.session_state:
        result = st.session_state.conversation({
            "question": user_query,
            "chat_history": st.session_state.get('chat_history', [])
        })
        response = result["answer"]
    else:
        response = "먼저 문서를 업로드해주세요~."
    with st.chat_message("assistant"):
        st.write(response)