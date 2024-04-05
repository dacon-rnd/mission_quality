from dotenv import load_dotenv
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()
# ========== ① 문서로드 ========== #

# PDF 파일 로드
loader = PyPDFLoader("assets/2023년_공공데이터_제공_운영실태_평가_편람.pdf")
document = loader.load()
document[0].page_content[:200]

# ========== ② 문서분할 ========== #

# 스플리터 지정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n\n",  # 분할기준
    chunk_size=3000,   # 사이즈
    chunk_overlap=500, # 중첩 사이즈
)

# 분할 실행
split_docs = text_splitter.split_documents(document)
# 총 분할된 도큐먼트 수
print(f'총 분할된 도큐먼트 수: {len(split_docs)}')

# ========== ③ Map 단계 ========== #

# Map 단계에서 처리할 프롬프트 정의
# 분할된 문서에 적용할 프롬프트 내용을 기입합니다.
# 여기서 {pages} 변수에는 분할된 문서가 차례대로 대입되니다.
map_template = """다음은 문서 중 일부 내용입니다
{pages}
이 문서 목록을 기반으로 평가개요, 평가지표 세부내용, 평가 유의사항으로 요약해 주세요.
답변:"""

# Map 프롬프트 완성
map_prompt = PromptTemplate.from_template(map_template)

# Map에서 수행할 LLMChain 정의
llm = ChatOpenAI(temperature=0,
                 model_name='gpt-3.5-turbo-0125')
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# ========== ④ Reduce 단계 ========== #

# Reduce 단계에서 처리할 프롬프트 정의
reduce_template = """다음은 요약의 집합입니다:
{doc_summaries}
이것들을 바탕으로 통합된 요약을 만들어 주세요.
답변:"""

# Reduce 프롬프트 완성
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Reduce에서 수행할 LLMChain 정의
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
)

# Map 문서를 통합하고 순차적으로 Reduce합니다.
reduce_documents_chain = ReduceDocumentsChain(
    # 호출되는 최종 체인입니다.
    combine_documents_chain=combine_documents_chain,
    # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
    collapse_documents_chain=combine_documents_chain,
    # 문서를 그룹화할 때의 토큰 최대 개수입니다.
    token_max=4000,
)

# ========== ⑤ Map-Reduce 통합단계 ========== #

# 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
map_reduce_chain = MapReduceDocumentsChain(
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
result = map_reduce_chain.run(split_docs)
# 요약결과 출력
print(result)