import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()
LLM_MODEL = 'gpt-3.5-turbo-0125'

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

def create_llm(model_name):
    return ChatOpenAI(
        temperature=0,
        model_name=model_name
    )

# ========== ③ Map 단계 ========== #

# Map 단계에서 처리할 프롬프트 정의
# 분할된 문서에 적용할 프롬프트 내용을 기입합니다.
# 여기서 {pages} 변수에는 분할된 문서가 차례대로 대입되니다.
def create_map_chain(llm):
    map_template = """당신은 공공데이터를 관리하는 감독관입니다.
    다음 평가내용을 참고하여 평가내용을 요약해줘
    평가내용 : {pages}
    답변:"""

    # Map 프롬프트 완성
    map_prompt = PromptTemplate.from_template(map_template)

    # Map에서 수행할 LLMChain 정의
    return LLMChain(llm=llm, prompt=map_prompt)

# ========== ④ Reduce 단계 ========== #

# Reduce 단계에서 처리할 프롬프트 정의
def create_reduce_chain(llm):
    reduce_template = """당신은 공공데이터의 관리하는 감독관입니다.
    다음 bullet summary를 종합하여 좋은 평가를 받기 위해 살펴봐야 하는 내용을 요약해줘 : {doc_summaries}
    답변:"""

    # Reduce 프롬프트 완성
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Reduce에서 수행할 LLMChain 정의
    return LLMChain(llm=llm, prompt=reduce_prompt)

# 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
def create_combine_chain(reduce_chain):
    return StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
    )

# Map 문서를 통합하고 순차적으로 Reduce합니다.
def create_reduce_documents_chain(combine_documents_chain):
    return ReduceDocumentsChain(
        # 호출되는 최종 체인입니다.
        combine_documents_chain=combine_documents_chain,
        # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
        collapse_documents_chain=combine_documents_chain,
        # 문서를 그룹화할 때의 토큰 최대 개수입니다.
        token_max=4000,
    )

# ========== ⑤ Map-Reduce 통합단계 ========== #

# 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
def create_map_reduce_chain(map_chain, reduce_documents_chain):
    return MapReduceDocumentsChain(
        # Map 체인
        llm_chain=map_chain,
        # Reduce 체인
        reduce_documents_chain=reduce_documents_chain,
        # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
        document_variable_name="pages",
        # 출력에서 매핑 단계의 결과를 반환합니다.
        return_intermediate_steps=False,
    )

# ========== ⑥ 실행 결과 ========== #

# Map-Reduce 체인 실행
# 입력: 분할된 도큐먼트(②의 결과물)
def run_chain(map_reduce_chain, split_docs):
    result = map_reduce_chain.invoke(split_docs)
    # 요약결과 출력
    print(result)
    return result


def save_pdf_file(uploaded_file):
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
    return temp_file


def main():
    result = None
    st.title("PDF 요약하기")
    st.divider()
    uploaded_file = st.file_uploader('PDF 파일을 업로드 해주세요.', type='pdf')

    if uploaded_file is not None:
        with st.spinner('PDF File 업로드 중입니다...'):
            pdf_file_name = save_pdf_file(uploaded_file)
        with st.spinner('Document를 읽어들이는 중입니다...'):
            document = pdf_loader(pdf_file_name)
            split_docs = split_document(document)
        with st.spinner('문서를 요약중입니다.'):
            llm = create_llm(LLM_MODEL)
            map_chain = create_map_chain(llm)
            reduce_chain = create_reduce_chain(llm)
            combine_documents_chain = create_combine_chain(reduce_chain)
            reduce_documents_chain = create_reduce_documents_chain(combine_documents_chain)
            map_reduce_chain = create_map_reduce_chain(map_chain, reduce_documents_chain)
            result = run_chain(map_reduce_chain, split_docs)
        if result is not None:
            st.markdown(result['output_text'])

if __name__=='__main__':
    main()