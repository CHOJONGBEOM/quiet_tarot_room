import random
import time
import streamlit as st
import base64
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path
from PIL import Image

# ============================================================
# 페이지 및 기본 설정
# ============================================================
st.set_page_config(
    page_title="A Quiet Symbolic Readig Room",
    page_icon="🔮",
    layout="centered",
)

# Document 폴더 자동 생성
if not os.path.exists("Document"):
    os.makedirs("Document")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ============================================================
# 커스텀 CSS (All-White & Clean Blue 테마)
# ============================================================
st.markdown(
    """
<style>
    .stApp { background-color: #ffffff; }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f2f6;
    }

    .user-box {
        background-color: #0066cc; 
        color: white; 
        padding: 15px;
        border-radius: 20px 20px 5px 20px; 
        margin: 10px 0 10px 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 15px;
    }
    .ai-box {
        background-color: #f8f9fa; 
        color: #1a1a1a; 
        padding: 15px;
        border-radius: 20px 20px 20px 5px; 
        margin: 10px 20% 10px 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        font-size: 15px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #0066cc;
        background-color: white;
        color: #0066cc;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0066cc;
        color: white;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-color: #e9ecef !important;
    }
    
    .search-result {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    .source-link {
        color: #0066cc;
        font-size: 0.9em;
    }
    
    .mode-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .mode-rag {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .mode-web {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .mode-llm {
        background-color: #fff3e0;
        color: #e65100;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


# ============================================================
# RAG: 인덱싱 함수
# ============================================================
def perform_indexing():
    with st.spinner("Document 폴더 내 문서를 인덱싱 중입니다..."):
        try:
            loader = PyPDFDirectoryLoader("Document/")
            documents = loader.load()
            if not documents:
                st.warning("Document 폴더에 PDF 파일이 없습니다.")
                return
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
            vectorstore = FAISS.from_documents(splits, embeddings)

            return vectorstore
        except Exception as e:
            st.error(f"인덱싱 중 오류 발생: {e}")

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    logo_b64 = get_base64_image("Symbol_logo.png")
    if logo_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" width="100%">',
            unsafe_allow_html=True,
        )
    else:
        st.title("🔮 Symbol_Whisper")

    st.divider()
    #지식 데이터베이스 섹션

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if st.session_state.vector_store is None:
        with st.spinner("📚 룰북을 불러오는 중입니다..."):
            st.session_state.vector_store = perform_indexing()

    st.caption("기록이 준비되었습니다.")


    st.divider()
    st.markdown("© 2025 A Quiet Symbolic Reading Room")

# ============================================================
# 메인 화면
# ==============# ============================================================

# =============================
# 경로
# =============================
BASE_DIR = Path(__file__).parent
CARD_BACK_PATH = BASE_DIR / "cards" / "back.png"
CARD_FRONT_DIR = BASE_DIR / "cards" / "front"

def format_card_name(card_id: str) -> str:
    """
    card_51_two_of_swords -> Two of Swords
    """
    # card_숫자_ 제거
    name = card_id.split("_", 2)[-1]

    # 언더스코어 → 공백
    name = name.replace("_", " ")

    # 보기 좋게 Title Case
    return name.title()


# ============================================================
# 🔮 LLM 해석용 카드 컨텍스트 생성
# ============================================================
def build_card_context():
    main = st.session_state.first_card
    supports = st.session_state.support_cards

    text = f"""
[MAIN CARD]
- Name: {format_card_name(main['card']['id'])}
- Orientation: {main['orientation']}
"""

    for i, c in enumerate(supports, start=1):
        text += f"""
[SUPPORT CARD {i}]
- Name: {format_card_name(c['card']['id'])}
- Orientation: {c['orientation']}
"""

    return text.strip()

# ============================================================
# 룰북에서 해석 가져오기
# ============================================================
def retrieve_tarot_rules(card_context: str, k: int = 6):
    vs = st.session_state.vector_store
    retriever = vs.as_retriever(search_kwargs={"k": k})

    # LangChain 버전 호환
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(card_context)
    else:
        docs = retriever.get_relevant_documents(card_context)

    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# LLM해석함수
# ============================================================
def generate_tarot_reading():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.6,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    card_context = build_card_context()
    rule_context = retrieve_tarot_rules(card_context)

    system_prompt = """
    You are a professional tarot reader.
    You strictly follow the provided tarot rulebook.
    You do not invent meanings outside the rulebook.
    Your tone is calm, symbolic, and reflective.

    IMPORTANT:
    - All final interpretations MUST be written in Korean.
    - Do NOT output English sentences.
    - You may internally use English rulebook content, but the user-facing response must be Korean.
    """

    user_prompt = f"""
[User Question]
{st.session_state.user_question}

[Selected Cards]
{card_context}

[Tarot Rulebook Excerpts]
{rule_context}

Please provide:
1. A holistic tarot interpretation
2. How the main card defines the core theme
3. How the three support cards develop the situation
4. Gentle and practical advice for the user
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content



# =============================
# 유틸: 앞면 카드 목록 로드
# =============================
def load_card_deck(front_dir: Path):
    # card_00_fool.png 같은 파일 전부 읽기
    files = sorted([p for p in front_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if len(files) == 0:
        raise RuntimeError(f"앞면 카드 이미지가 없습니다: {front_dir}")

    deck = []
    for p in files:
        deck.append({
            "id": p.stem,          # 예: card_00_fool
            "path": p,
        })
    return deck

CARD_DECK = load_card_deck(CARD_FRONT_DIR)





# ============================================================
# 상태 초기화
# ============================================================

def init_state():
    if "phase" not in st.session_state:
        st.session_state.phase = "question"

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    
    if "draft_question" not in st.session_state:
        st.session_state.draft_question = ""

    if "card_order" not in st.session_state:
        st.session_state.card_order = random.sample(range(9), 9)

    if "first_card" not in st.session_state:
        st.session_state.first_card = None

    if "support_cards" not in st.session_state:
        st.session_state.support_cards = []




init_state()
# ============================================================
# 헤더
# ============================================================

st.markdown(
    "<h2 style='color: #0066cc;'>카드를 뽑고, 미래를 마주하세요</h2>",
    unsafe_allow_html=True
)
st.caption("🕯️ 질문을 품고, 하나의 카드를 선택합니다.🕯️")


# ============================================================
# Phase 1 — 질문
# ============================================================

QUESTION_POOLS = {
    "관계": [
        "이 사람과의 관계는 앞으로 어떻게 변할까요?",
        "지금 이 관계를 계속 이어가는 게 맞을까요?",
        "이 관계에서 내가 놓치고 있는 게 있을까요?",
        "상대방은 나를 어떻게 바라보고 있을까요?",
        "지금 이 거리감은 어떤 의미일까요?"
    ],
    "일/진로": [
        "지금 선택한 진로는 나에게 맞는 길일까요?",
        "지금 이 일을 계속해도 괜찮을까요?",
        "변화를 선택하면 어떤 결과가 올까요?",
        "지금의 고민은 성장의 신호일까요?",
        "내가 두려워하는 건 실패일까요, 변화일까요?"
    ],
    "선택": [
        "이 선택을 하면 무엇을 얻게 될까요?",
        "지금 망설이는 이유는 무엇일까요?",
        "선택하지 않은 길은 어떤 의미일까요?",
        "지금 결정을 미뤄도 괜찮을까요?",
        "이 선택은 나를 어디로 데려갈까요?"
    ],
    "나 자신": [
        "지금의 나는 어떤 상태에 있을까요?",
        "내가 나를 너무 몰아붙이고 있는 걸까요?",
        "지금 필요한 건 노력일까요, 휴식일까요?",
        "나는 무엇을 두려워하고 있을까요?",
        "지금의 불안은 어디서 온 걸까요?"
    ],
    "그냥 궁금함": [
        "지금 이 시기의 흐름은 어떤 의미일까요?",
        "지금 나에게 필요한 태도는 무엇일까요?",
        "지금의 혼란은 어떤 변화를 예고할까요?",
        "이 시기를 어떻게 받아들이는 게 좋을까요?",
        "지금 멈춰 서도 괜찮을까요?"
    ]
}



# ============================================================
# Phase 1 — 질문 생성 & 확정
# ============================================================

if st.session_state.phase == "question":

    st.markdown("### 어떤 미래가 궁금하신가요?")

    cols = st.columns(len(QUESTION_POOLS))
    for i, (topic, pool) in enumerate(QUESTION_POOLS.items()):
        if cols[i].button(topic):
            st.session_state.draft_question = random.choice(pool)
            st.rerun()

    st.caption("버튼을 다시 누르면 다른 질문이 나타납니다.")

    st.markdown(
        "<div style='margin-top: 350px;'></div>",
        unsafe_allow_html=True
    )


    question_text = st.text_input(
        "",
        value=st.session_state.draft_question,
        placeholder="주제를 선택하거나 궁금한 점을 입력하세요"
    )

    if st.button("카드를 뽑으러 간다"):
        if question_text and question_text.strip():
            st.session_state.user_question = question_text.strip()
            st.session_state.phase = "first_select"
            st.rerun()

# ============================================================
# Phase 2 — 첫 번째 카드 (1장)
# ============================================================
elif st.session_state.phase == "first_select":

    st.markdown("### 첫 번째 카드 — 핵심 흐름")

    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            with cols[col]:                          
                
                st.image(CARD_BACK_PATH, use_container_width=True)
                if st.button("선택", key=f"first_{idx}"):

                    card = random.choice(CARD_DECK)
                    orientation = random.choice(["upright", "reversed"])

                    st.session_state.first_card = {
                        "card": card,
                        "orientation": orientation,
                    }

                    st.session_state.phase = "first_reveal"
                    st.rerun()

# ============================================================
# Phase 3 — 첫 카드 공개
# ============================================================
elif st.session_state.phase == "first_reveal":

    st.markdown("### 선택된 카드")

    card = st.session_state.first_card
    img = Image.open(card["card"]["path"])
    if card["orientation"] == "reversed":
        img = img.rotate(180, expand=True)

    st.image(img, width=300)
    display_name = format_card_name(card["card"]["id"])
    direction = "역방향" if card["orientation"] == "reversed" else "정방향"

    st.caption(f"{display_name} · {direction}")

    if st.button("다음 카드를 뽑는다"):
        st.session_state.card_order = random.sample(range(9), 9)
        st.session_state.phase = "second_select"
        st.rerun()

# ============================================================
# Phase 4 — 보조 카드 (3장)
# ============================================================

    

elif st.session_state.phase == "second_select":
    CARD_SIZE = 100  # ⭐ 여기만 조절하면 전체 크기 바뀜
    st.markdown("### 세 장의 카드 — 흐름의 전개")
    st.caption(f"선택됨: {len(st.session_state.support_cards)} / 3")

    selected_map = {c["slot"]: c for c in st.session_state.support_cards}
    selected_slots = set(selected_map.keys())

    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            idx = row * 3 + col

            with cols[col]:

                # ✅ 이미 선택된 카드 → 즉시 앞면
                if idx in selected_map:
                    card = selected_map[idx]
                    img = Image.open(card["card"]["path"])
                    if card["orientation"] == "reversed":
                        img = img.rotate(180, expand=True)
                    st.image(img, width=CARD_SIZE)
                    display_name = format_card_name(card["card"]["id"])
                    direction = "역방향" if card["orientation"] == "reversed" else "정방향"
                    st.caption(f"{display_name} · {direction}")

                # ⛔ 아직 선택 안 된 카드
                else:
                    st.image(CARD_BACK_PATH, width=CARD_SIZE)

                    # 🔒 3장 미만일 때만 선택 가능
                    if len(selected_slots) < 3:
                        if st.button("선택", key=f"support_{idx}"):

                            used_ids = {st.session_state.first_card["card"]["id"]}
                            used_ids |= {
                                c["card"]["id"]
                                for c in st.session_state.support_cards
                            }

                            deck = [
                                c for c in CARD_DECK
                                if c["id"] not in used_ids
                            ]

                            card = random.choice(deck)
                            orientation = random.choice(["upright", "reversed"])

                            st.session_state.support_cards.append({
                                "slot": idx,
                                "card": card,
                                "orientation": orientation
                            })

                            st.rerun()

    # 🔮 해석 버튼 (정확히 3장일 때만)
    if len(selected_slots) == 3:
        st.divider()
        if st.button("🔮 이 카드들로 해석하기"):
            st.session_state.phase = "interpret"
            st.rerun()



# ============================================================
# Phase 5 — 최종 해석
# ============================================================
elif st.session_state.phase == "interpret":

    st.markdown("### 네 장의 카드가 말하는 이야기")
    st.caption(f"질문: {st.session_state.user_question}")

    main_card = st.session_state.first_card
    support_cards = st.session_state.support_cards

    # =========================
    # 🔮 핵심 카드 (상단 중앙)
    # =========================
    st.markdown("#### 핵심")

    left, center, right = st.columns([1, 2, 1])
    with center:
        img = Image.open(main_card["card"]["path"])
        if main_card["orientation"] == "reversed":
            img = img.rotate(180, expand=True)

        st.image(img, width=280)
        display_name = format_card_name(main_card["card"]["id"])
        direction = "역방향" if main_card["orientation"] == "reversed" else "정방향"

        st.caption(f"{display_name} · {direction}")

    st.divider()

    # =========================
    # 🃏 전개 카드 3장 (하단 가로)
    # =========================
    st.markdown("#### 흐름의 전개")

    cols = st.columns(3)
    labels = ["전개 1", "전개 2", "결론"]

    for col, label, card in zip(cols, labels, support_cards):
        with col:
            img = Image.open(card["card"]["path"])
            if card["orientation"] == "reversed":
                img = img.rotate(180, expand=True)

            st.image(img, width=180)  # 🔹 메인보다 작게
            display_name = format_card_name(card["card"]["id"])
            direction = "역방향" if card["orientation"] == "reversed" else "정방향"
            st.caption(f"{label}\n"f"{display_name} · {direction}")


    st.divider()

    st.markdown("### 🔮 타로 해석")

    if "reading_result" not in st.session_state:
        with st.spinner("징표를 해석하는 중입니다..."):
            st.session_state.reading_result = generate_tarot_reading()

    st.markdown(
        f"""
        <div class="ai-box">
        {st.session_state.reading_result}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    if st.button("🔄 다시 시작"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
