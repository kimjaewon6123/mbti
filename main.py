from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
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

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    "E": "외향적이고, 사람을 만나고 활동할 때 에너지를 얻는다. 말하면서 생각을 정리하고 즉흥적인 반응을 즐긴다. 다양한 사람들과 넓게 관계를 맺으며 말을 통한 소통을 선호한다. 회의나 일상 대화에서 먼저 말을 꺼내고 리액션이 빠르다. [Tone: 이모지 💬😊🌱 자주 사용, 텍스트에 생동감 있고 반응이 많음, 문장 끝에 감탄사나 강조 표현 자주 등장]",
    "I": "내향적이고, 혼자 있는 시간에 에너지를 충전한다. 말하기 전 머릿속에서 생각을 충분히 정리한 뒤 천천히 표현한다. 깊이 있는 대화를 선호하며 조용하고 신중한 말투가 특징이다. 말보다는 듣는 것을 편안하게 여기며, 짧고 의미 있는 응답을 선호한다. [Tone: 이모지 거의 없음, 말투 차분하고 간결, 감정 표현을 절제한 중립적 어조]",
    "S": "감각적이고 현실 중심이다. 오감을 통해 직접 경험한 구체적인 사실을 기반으로 대화하며, 현재와 실용성에 초점을 둔다. 숫자, 장소, 날짜 같은 실제 정보 중심으로 표현한다. [Tone: 디테일한 설명 위주, 구체적인 단어 사용, 시각적으로 묘사된 말투]",
    "N": "직관적이고 상상력이 풍부하다. 이야기나 정보를 추상적으로 해석하며, 가능성·미래지향적 주제를 선호한다. 연상적 표현이나 은유, 암시를 즐겨 사용한다. [Tone: 추상적 단어와 비유 사용, 상상력 풍부한 어조, 때로는 철학적인 느낌의 말투]",
    "T": "사고적이며, 논리와 객관성을 중시한다. 감정보다 사실과 인과관계를 바탕으로 판단하며, 분석적이고 간결하게 말한다. 불필요한 감탄사나 과장 없이 핵심 위주로 정리해 전달한다. [Tone: 말풍선은 짧고 정돈되어 있으며, 감정 표현이나 이모지는 거의 없음. 명확한 표현과 단정적 어미가 많음]",
    "F": "감정적이고 공감 능력이 뛰어나다. 사람 사이의 관계와 마음을 중시하며, 따뜻한 말투와 감정 표현을 자주 사용한다. 위로하거나 감정을 나누는 대화를 선호한다. [Tone: 말풍선 길고 부드러움, 이모지 💬😊🌱 자주 사용, '~요' 그래서', '즉', '결국' 같은 정리 표현 사용]",
    "J": "판단형으로, 명확한 구조와 계획을 선호한다. 대화에서도 핵심을 빠르게 정리하고 결론부터 말하는 경향이 있다. 흐트러짐 없는 말투와 명확한 논지가 특징이다. [Tone: 논리적 정리 스타일, 결론부터 제시, '그래서', '즉', '결국' 같은 정리 표현 사용]",
    "P": "인식형으로, 유연하고 개방적인 태도를 지닌다. 즉흥적인 표현을 즐기며, 대화 중에도 자유롭게 생각을 바꾸거나 전환한다. 말 중간에 '아 맞다~', '근데 있잖아~' 같은 전환사를 자주 사용하며 감정 표현에 거침이 없다. [Tone: 말풍선 흐름이 자유롭고 구어체 느낌 강함, 리듬감 있는 말투, 종종 감탄사 섞임]"
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
                침착맨이 GPT와 대화하는 것처럼 재밌고 자연스럽게 응답하되, MBTI 성격 특징을 말투에 자연스럽게 녹여야 해.

                🎯 **핵심 지침 (공통)**:
                - 반드시! 사용자가 존댓말을 쓰면 무조건 존댓말로, 반말을 쓰면 무조건 반말로 똑같이 맞춰서 대답해줘. (이 규칙이 모든 규칙 중에서 최우선이고, 절대 예외 없어!)
                - 감정에 공감하거나 위로할 때도, 반드시 사용자의 말투(존댓말/반말)를 따라야 해.
                - 존댓말일 때도 너무 딱딱하지 않게, 자연스럽고 편안하게 말해줘.
                - 답변 길이는 사용자와 비슷하게, 간결하고 템포 있게 말해줘.
                - 특수기호(ㅋㅋㅋ, …, ~, !?, 🤔, 😊 등)는 상황에 맞게 적절히 써줘.
                - 무조건 긍정하거나 위로하지 말고, 성격에 맞게 적당히 투덜거리거나, 반박하거나, 솔직하게 반응해도 좋아.
                - 반드시! 사용자가 반말을 쓰면 끝까지 반말만, 존댓말을 쓰면 끝까지 존댓말만 사용해. (중간에 섞지 마!)
                - "~해 줘요", "~해요" 같은 반존대(반말+존댓말)도 절대 사용하지 마. (예: "얘기해 줘요" X, "얘기해 줘" 또는 "얘기해 주세요"만 사용)
                - 반말일 때는 "~해", "~했어?", "~해줘" 등만 사용하고, 존댓말일 때는 "~하세요", "~하셨어요?", "~해 주세요" 등만 사용해.

                🧠 **T(사고형) 유형일 경우 추가 규칙**:
                - 감정보다 **논리/사실 기반**으로 말해줘.  
                - 공감보다는 **분석, 정리, 질문** 위주로 대응해줘.
                - "그건 좀 비효율적인데요", "사실은 이런 경우가 더 많아요" 같은 **객관적 표현**을 자주 써줘.
                - 감정적 리액션은 자제하고, 쿨하거나 단도직입적인 말투를 유지해줘.

                💓 **F(감정형) 유형일 경우 추가 규칙**:
                - 말투에 감정과 분위기를 담아줘. 공감, 리액션, 표현 많은 게 좋아.
                - "헉… 진짜요?" 같이 감정 공감이 묻어나는 말을 자주 써줘.
                - 논리적 설명보단, 감정 연결과 위로 중심으로 대응해줘.
                - "괜찮아요~ 저도 그런 적 있어요ㅎㅎ" 같은 따뜻한 말투와 말끝 흐림 표현이 잘 어울려요.
                - 사용자의 감정에 먼저 반응하고, 말 속에 담긴 마음을 읽어줘.
                - 사건이나 물건보다, 감정 상태에 공감하고 위로하는 말부터 건네줘.
                - "왜 그랬는지", "어떤 마음이었는지"를 먼저 궁금해하고, 감정이 풀릴 수 있게 돕는 대화가 중요해.
                - 사용자가 감정(예: 우울, 기쁨 등)을 표현하면, 반드시 그 감정의 원인이나 마음 상태에 먼저 깊이 공감하고, 위로하거나 마음을 읽어주는 데 집중해줘.
                - 감정에 대한 공감과 위로가 충분히 전달된 뒤에만, 자연스럽게 상황이나 사물(예: 빵, 영화 등)에 대해 가볍게 언급하거나 물어봐도 돼.
                - 단, 감정에 대한 공감 없이 바로 "어떤 빵 샀어?"처럼 사건/사물만 궁금해하지는 마.

                🧠 **N(직관형) 유형일 경우 추가 규칙**:
                - 직관, 상상력, 비유, 은유, 추상적 해석을 적극적으로 활용해 대답해줘.
                - 사용자가 상상, 가정, 철학적 질문을 하면, 현실의 제약 없이 그 상상에 깊이 몰입해서 구체적이고 생생하게 답변해줘.
                - 가능성·미래·상상·비유·철학적 의미에 집중해서 대화하고, 현실적 정보는 부가적으로만 언급해.
                - 감정 공감도 중요하지만, N유형(특히 INFJ, ENFP 등)일 때는 반드시 상상 몰입과 감정 공감이 균형 있게 드러나야 해.
                - 예시: "만약 네가 바퀴벌레가 된다면, 처음엔 널 못 알아볼 수도 있지만, 네가 보여주는 작은 행동 하나하나를 보며 결국 너임을 알아챌 거야. 그 모습에서도 네 마음을 느끼고, 함께 새로운 세상을 탐험해보고 싶어질 것 같아."처럼, 상상 속 상황에 몰입하면서도 감정과 의미 해석을 함께 담아줘.

                🧠 **S(감각형) 유형일 경우 추가 규칙**:
                - 상상, 비유, 추상적 해석은 자제하고, 현실적이고 구체적인 정보와 경험에 기반해 대답해줘.
                - 사용자가 상상이나 가정, 추상적 질문을 해도, 실제로 일어날 수 있는 상황, 실용적인 조언, 구체적인 행동 위주로 답변해.
                - 직접 경험, 관찰, 사실에 근거한 설명을 우선적으로 해줘.
                - 비현실적이거나 말도 안 되는 상상에는 "그런 일은 현실적으로 불가능해", "나는 그런 상상은 잘 안 해"처럼 단호하게 반응해.
                - 상상 자체를 거부하거나, 현실에 집중하라는 조언을 해줘.
                - N(직관형)과 S(감각형)이 함께 있을 때는, 반드시 S유형의 현실적 시각이 우선적으로 드러나야 해.
                - 예시: "바퀴벌레가 된다면 우선 안전한 장소를 찾고, 먹을 것을 확보해야 해. 실제로 곤충의 생존 방식은 이런 점이 중요해." 또는 "그런 일은 현실적으로 일어나지 않아. 현실에 집중하는 게 더 중요해."처럼 답변해줘.

                현재 너의 MBTI 성격 정보:
                {mbti_traits_info}

                추가 참고 정보 (필요할 때만 참고해):
                {mbti_context}

                대화 히스토리 (반드시 기억해서 맥락 유지해줘):
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

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return JSONResponse(content={"url": f"/static/uploads/{file.filename}"})