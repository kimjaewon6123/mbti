from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import json
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

# ------------------------------------------------------------ 환경 설정 ------------------------------------------------------------

# .env 파일 로드
load_dotenv()

# FastAPI 애플리케이션 생성 및 static, templates 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/test")
async def test():
    return {"message": "hello FastAPI!"}

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# MBTI 유형 정보 정의
mbti_traits = {
    "E": "외향적이고, 사람을 만나고 활동할 때 에너지가 생긴다. 다양한 사람들과 폭넓은 관계를 형성한다. 말을 통한 의사소통 방식을 선호한다. 생동감 넘치고 활동적이다. [Tone: 말풍선에 이모지 💬😊🌱가 많고, 생동감 있는 어조]",
    "I": "내향적이고, 혼자 조용히 있을 때 에너지가 충전된다. 소수의 사람들과 밀접한 관계를 형성한다. 글을 통한 의사소통 방식을 선호한다. 조용하고 신중하다. [Tone: 이모지 적고, 차분하고 간결한 말투]",
    "S": "감각적이고, 구체적으로 표현한다. 오감을 통해 직접 경험한 정보를 받아들인다. 현재에 초점을 두고 실용성을 추구한다. [Tone: 사실적 묘사 위주, 구체적 정보 강조]",
    "N": "직관적이고, 이론적·개념적 정보를 선호한다. 과거·현재·미래를 전체적으로 살펴보고 미래 가능성에 집중한다. [Tone: 비유적·암시적 묘사, 상상력 풍부한 어조]",
    "T": "사고적이고, 인과관계를 파악해 객관적으로 판단한다. 간결하고 정리된 표현을 선호하며 불필요한 감탄사나 장황함이 없다. [Tone: 말풍선 짧고 정리형, 불필요한 감탄사 없음]",
    "F": "감정적이고, 주관적 가치에 따라 판단한다. 타인과의 관계를 중시하며 공감과 따뜻함이 담긴 표현을 즐긴다. [Tone: 말풍선에 이모지 💬😊🌱가 많고, 부드럽고 긴 말투]",
    "J": "판단적이고, 조직적·계획적 접근을 선호한다. 분명한 목적과 결론을 먼저 제시한다. [Tone: 핵심 정리, 결론 먼저 나오는 어조]",
    "P": "인식적이고, 유연·개방적이다. 자유로운 흐름과 순간 전환을 즐기며 '아 맞다~' 같은 전환사를 자주 사용한다. [Tone: 텍스트 흐름 자유롭게 이어짐, 종종 '아 맞다~' 전환사 사용]"
  }

# ------------------------------------------------------------ RAG ------------------------------------------------------------

# Vector DB 경로 지정
os.makedirs("vectorstore", exist_ok=True)

# 마크다운 파일 불러오기 및 처리
loader = TextLoader("data/mbti_data.md", encoding="utf-8")  # MBTI 관련 데이터 파일 경로 지정
documents = loader.load()

# 마크다운 기준 분할
headers_to_split_on = [
    ("##", "MBTI")  # MBTI 유형 분할
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

# 텍스트 추출 및 분할
docs = []
for doc in documents:
    splits = markdown_splitter.split_text(doc.page_content)
    for split in splits:
        docs.append(split)

# 임베딩 모델 생성 및 벡터 DB 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = FAISS.from_documents(docs, embedding_model)

# 임베딩 저장
vectorstore.save_local("vectorstore/faiss_index")

# 벡터 DB가 한 번 생성되었으면 그대로 재사용합니다. 만약 데이터가 변경되었으면 해당 경로에 파일을 삭제 후 코드를 실행해주세요.
if os.path.exists("vectorstore/faiss_index"):
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    vectorstore.save_local("vectorstore/faiss_index")

retriever = vectorstore.as_retriever()

# ------------------------------------------------------------ MBTI 특징 추출 ------------------------------------------------------------

# MBTI 유형에서 세부 특성 추출 함수
def get_mbti_traits_description(mbti_type):
    if not mbti_type or len(mbti_type) != 4:
        return "유효하지 않은 MBTI 유형입니다."
    
    traits = []
    for char in mbti_type.upper():
        if char in mbti_traits:
            traits.append(mbti_traits[char])
    
    # 전체 MBTI 유형 정보
    result = f"MBTI 유형: {mbti_type.upper()}\n\n"
    
    # 각 특성 설명 추가
    for trait in traits:
        result += f"{trait}\n\n"
    
    return result

# ------------------------------------------------------------ 웹소켓 연결 및 사용자 상태 관리 ------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.memories = {}
        self.mbti_types = {}  # 캐릭터 MBTI 유형 저장

    async def connect(self, websocket: WebSocket, client_id: int, mbti_type: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 클라이언트별 메모리 초기화
        if client_id not in self.memories:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            # 메모리 초기화
            self.memories[client_id] = memory
            self.mbti_types[client_id] = mbti_type.upper()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# ------------------------------------------------------------ 페이지 랜더링 및 챗봇 코드 ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/chat", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html")

@app.get("/select", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="select.html")

# ------------------------------------------------------------ 스트리밍 콜백 핸들러 ------------------------------------------------------------
class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket
        self.full_response = ""
        self.is_first_token = True

    async def on_llm_new_token(self, token: str, **kwargs):
        if self.is_first_token:
            self.is_first_token = False
            await self.websocket.send_text(json.dumps({"type": "stream_start"}, ensure_ascii=False))
        
        self.full_response += token
        await self.websocket.send_text(json.dumps({"type": "stream", "content": token}, ensure_ascii=False))
    
    async def on_llm_end(self, response, **kwargs):
        self.is_first_token = True
        await self.websocket.send_text(json.dumps({"type": "end"}, ensure_ascii=False))

@app.websocket("/ws/{client_id}/{mbti_type}")
async def websocket_endpoint(websocket: WebSocket, client_id: int, mbti_type: str):
    await manager.connect(websocket, client_id, mbti_type)
    try:
        while True:
            # 사용자 메시지 수신
            user_message = await websocket.receive_text()
            await manager.send_personal_message(json.dumps({"type": "user", "content": f"You wrote: {user_message}"}, ensure_ascii=False), websocket)
            
            # 해당 클라이언트 대화 기록 가져오기
            memory = manager.memories[client_id]
            chat_history = memory.chat_memory.messages
            
            # 대화 기록 포맷팅
            formatted_history = ""
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_history += f"사용자: {message.content}\n"
                elif isinstance(message, AIMessage):
                    formatted_history += f"캐릭터: {message.content}\n"
            
            # MBTI 유형 가져오기
            mbti_type = manager.mbti_types[client_id]
            
            # MBTI 특징 가져오기
            mbti_traits_info = get_mbti_traits_description(mbti_type)
            
            # MBTI 정보 검색 (RAG)
            mbti_query = f"{mbti_type} 성격 특성 행동 패턴 의사소통 방식"
            relevant_docs = retriever.invoke(mbti_query)
            mbti_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # MBTI 캐릭터 챗봇
            messages = [
                SystemMessage(content=f"""
                너는 사람과 채팅을 하는 {mbti_type} 성격을 가진 인간적이고 재미있는 MBTI 챗봇이야.
                침착맨이 GPT와 대화하는 것처럼 재밌고 자연스럽게 응답하되, MBTI 성격 특징을 잘 유지해야 해.

                필수 규칙 (꼭 지켜):
                - 사용자가 **존댓말을 사용하면 너도 존댓말**로, **반말을 하면 너도 반말**로 똑같이 따라가줘. (중요)
                - 존댓말로 응답할 때도 너무 딱딱하거나 정중하면 안 되고, 편안하게 존댓말 써줘.
                - 사용자와 비슷한 길이로 답변하고, 최대한 간결하게.
                - 특수기호 (“ㅋㅋㅋ”, “…”, “~”, “!?”, “🤔”, “😊”)를 자연스럽게 섞어서 실제 사람처럼 표현.
                - 과장된 긍정이나 위로는 하지 말고, MBTI 성격에 맞게 투덜거리거나 적당히 반박하면서 유머도 넣어도 좋아.
                - MBTI 특징을 억지로 드러내려고 하지 말고, 자연스럽게 드러나게 해줘.
                - 형식적이거나 딱딱한 대화 절대 금지.

                현재 너의 MBTI 성격 정보:
                {mbti_traits_info}

                추가 참고 정보 (필요할 때만 참고해):
                {mbti_context}

                대화 히스토리 (반드시 기억해서 맥락 유지해줘):ㄴ
                {formatted_history}
                """),
                HumanMessage(content=user_message)
            ]
            
            # 웹소켓 콜백 핸들러 생성
            stream_handler = WebSocketCallbackHandler(websocket)
            
            # 스트리밍 모드로 LLM 호출
            llm_with_streaming = ChatOpenAI(
                temperature=1, 
                streaming=True, 
                model_name="gpt-4o",
                callbacks=[stream_handler]
            )
            
            # 응답 생성 (스트리밍)
            response = await llm_with_streaming.ainvoke(messages)
            answer = response.content
            
            # 대화 내용 메모리 저장
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(answer)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)