# ============================================================================
# 🌐 Streamlit 网页版入口（部署到 Streamlit Cloud 用这个文件）
# ============================================================================
# 【这个文件做什么？】
#   给 langGraphDome.py 里的"三步搜索助手"套一个浏览器聊天界面：
#     - 左边/下方输入框：像 ChatGPT 一样提问
#     - 助手回答以"打字机"方式逐字流式显示
#     - 同一个浏览器标签页内自动保留多轮对话上下文
#
# 【本地运行】
#   在本目录下执行：streamlit run streamlit_app.py
#
# 【密钥从哪里来？】
#   - 本地运行：读取项目根目录的 .env（langGraphDome 内部会自动加载）
#   - 云端部署：在 Streamlit Cloud 网页的 Secrets 里填写，下面代码会把
#     st.secrets 的内容转成环境变量，供 langGraphDome 读取
# ============================================================================

# os：读取/设置环境变量；uuid：给每个浏览器会话生成独立的会话 ID
import os
import uuid

# streamlit：网页 UI 框架，惯例别名为 st
import streamlit as st

# ----------------------------------------------------------------------------
# 第 0 步：把云端 Secrets 注入环境变量（必须在 import langGraphDome 之前执行）
# ----------------------------------------------------------------------------
# st.secrets 是 Streamlit 提供的安全密钥仓库；本地没有配置时这一步会抛异常，
# 用 try/except 忽略，改走 .env 文件
try:
    for _key, _value in st.secrets.items():
        # setdefault：不覆盖本地 .env 已有的值
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

# 从现有文件直接复用"图"的构建函数和消息类，无需重写 Agent 逻辑
from langGraphDome import create_search_assistant  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402


# ----------------------------------------------------------------------------
# 第 1 步：流式 token 清洗器（逻辑移植自 langGraphDome.py 的 clean_delta）
# ----------------------------------------------------------------------------
class TokenCleaner:
    """清洗模型流式输出：过滤特殊控制 token 和 ```think 思考块。

    为什么要做成类？流式是一块一块到达的，思考块可能横跨多个 chunk，
    需要用 self.in_think_block 记住"当前是否在思考块里"。
    """

    def __init__(self):
        # 标记当前是否处于 ```think ... ``` 内部
        self.in_think_block = False

    def clean(self, delta: str) -> str:
        result = []
        i = 0
        while i < len(delta):
            # 过滤盒子标记
            if delta.startswith("<|begin_of_box|>", i):
                i += len("<|begin_of_box|>")
                continue
            if delta.startswith("<|end_of_box|>", i):
                i += len("<|end_of_box|>")
                continue
            # 遇到思考块开头，进入"跳过模式"
            if not self.in_think_block and delta.startswith("```think", i):
                self.in_think_block = True
                i += len("```think")
                continue
            # 思考块内部：跳到块尾 ```；本 chunk 没有结尾就整块丢弃
            if self.in_think_block:
                end_idx = delta.find("```", i)
                if end_idx != -1:
                    self.in_think_block = False
                    i = end_idx + 3
                else:
                    i = len(delta)
                continue
            # 普通内容，保留
            result.append(delta[i])
            i += 1
        return "".join(result)


# ----------------------------------------------------------------------------
# 第 2 步：把 LangGraph 的事件流转成"只吐答案文字"的异步生成器
# ----------------------------------------------------------------------------
async def answer_stream(app, prompt: str, thread_id: str):
    """调用图并逐字 yield 最终答案节点（answer）的 token。

    st.write_stream 会不断接收 yield 的文本并实时渲染到页面上。
    """
    config = {"configurable": {"thread_id": thread_id}}
    cleaner = TokenCleaner()

    async for ev in app.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
        version="v2",
    ):
        # 只关心 answer 节点的 LLM 流式 token
        node = ev.get("metadata", {}).get("langgraph_node")
        if node == "answer" and ev["event"] == "on_chat_model_stream":
            delta = ev["data"]["chunk"].content
            if delta:
                cleaned = cleaner.clean(delta)
                if cleaned:
                    yield cleaned


# ----------------------------------------------------------------------------
# 第 3 步：页面初始化（整个文件会从上到下执行，缓存只构建一次图）
# ----------------------------------------------------------------------------

st.set_page_config(page_title="智能搜索助手", page_icon="🤖")
st.title("🤖 智能搜索助手")
st.caption("理解问题 → Tavily 联网搜索 → 生成答案（LangGraph 驱动）")

# 密钥检查：缺少时直接在页面给出提示，避免运行时报错看不懂
if not os.getenv("LLM_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error(
        "缺少 API 密钥：请确认本地 .env（或云端 Settings → Secrets）中已配置 "
        "**LLM_API_KEY** 和 **TAVILY_API_KEY**。"
    )

# cache_resource：图对象只在首次访问时创建一次，之后所有对话复用
@st.cache_resource
def get_app():
    return create_search_assistant()


app = get_app()

# 每个浏览器会话分配独立 thread_id，对话记忆互不干扰
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4()}"

# 页面消息历史：用于刷新聊天记录的展示
if "messages" not in st.session_state:
    st.session_state.messages = []

# 先把历史消息渲染出来
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------------------------------------------------------
# 第 4 步：聊天输入与流式回复
# ----------------------------------------------------------------------------
prompt = st.chat_input("请输入你的问题，例如：今天有什么科技新闻？")

if prompt:
    # 1) 展示并保存用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) 流式展示助手回复（spinner 期间图在执行 理解→搜索→生成 三个节点）
    with st.chat_message("assistant"):
        with st.spinner("正在思考并联网搜索…"):
            # write_stream 支持异步生成器；返回值是拼完整的最终文本
            answer = st.write_stream(
                answer_stream(app, prompt, st.session_state.thread_id)
            )

    # 3) 保存助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
