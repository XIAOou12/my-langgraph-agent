# ============================================================================
# 📚 LangGraph 三步问答助手（带详细注释版）
# ============================================================================
# 【这个文件做什么？】
#   实现一个"理解问题 → 联网搜索 → 生成答案"的三步工作流对话机器人。
#   用户输入问题后，AI 会：
#     1. understand 节点：用 LLM 理解问题、提炼搜索关键词
#     2. search 节点：调用 Tavily 搜索引擎获取真实网页结果
#     3. answer 节点：基于搜索结果用 LLM 生成最终答案
#   输出采用流式（打字机效果），支持多轮对话。
# ============================================================================


# ============================================================================
# 第一部分：导入依赖
# ============================================================================

# typing 是 Python 标准库，提供类型注解工具
# TypedDict：定义"有固定字段的字典"类型，让 IDE 能提示 state 里有哪些 key
# Annotated：给类型附加额外信息（这里用来指定 messages 字段的"合并函数"）
from typing import TypedDict, Annotated

# add_messages 是 LangGraph 提供的"消息合并函数"
# 当多个节点都往 messages 里追加消息时，add_messages 会把它们合并成一个列表
# 而不是互相覆盖（这是 StateGraph 状态更新的核心机制）
from langgraph.graph.message import add_messages

# os 标准库，用来读取环境变量（.env 加载后变量存在 os.environ 里）
import os

# asyncio 标准库，用来写异步代码（ainvoke/astream_events 都是异步的）
import asyncio

# python-dotenv 库：加载 .env 文件里的环境变量
# load_dotenv：把 .env 内容加载到 os.environ
# find_dotenv：从当前目录一路向上找 .env 文件（解决子目录运行找不到 .env 的问题）
from dotenv import load_dotenv, find_dotenv

# langchain_openai 提供 ChatOpenAI 类，兼容 OpenAI 协议的所有模型（包括硅基流动）
from langchain_openai import ChatOpenAI

# langchain_core.messages 提供三种消息类：
# HumanMessage：用户说的话
# AIMessage：AI 说的话
# SystemMessage：系统提示词（给 AI 的角色设定/指令）
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# tavily 库提供 TavilyClient，用来调用 Tavily 搜索引擎 API
from tavily import TavilyClient

# langgraph.graph 提供构建工作流图的核心类：
# StateGraph：状态图，定义节点和边
# START：图的入口点（固定常量）
# END：图的出口点（固定常量）
from langgraph.graph import StateGraph, START, END

# langgraph.checkpoint.memory 提供 InMemorySaver
# 它是一个"内存版的检查点"，用来保存图的执行状态
# 有了它，多轮对话的上下文才能被记住（否则每次调用都是全新的）
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================================
# 第二部分：定义图的状态（State）
# ============================================================================

# SearchState 是整个工作流共享的"状态字典"
# 每个节点读取 state 里的某些字段，执行后返回要更新的字段
# TypedDict 让我们可以像定义类一样定义字典的字段和类型
class SearchState(TypedDict):
    # messages：对话消息列表，用 Annotated 指定 add_messages 作为合并函数
    # 意思是：节点返回的 messages 会被追加到已有列表，而不是替换
    messages: Annotated[list, add_messages]

    # user_query：LLM 理解后的用户需求总结（understand 节点产出）
    user_query: str

    # search_query：优化后的搜索关键词（understand 节点产出，给 search 节点用）
    search_query: str

    # search_results：Tavily 搜索返回的结果文本（search 节点产出）
    search_results: str

    # final_answer：最终生成的答案（answer 节点产出）
    final_answer: str

    # step：当前执行到哪一步，用来在节点间传递状态（如 search_failed）
    step: str


# ============================================================================
# 第三部分：加载环境变量 + 初始化模型和客户端
# ============================================================================

# load_dotenv(find_dotenv())：先找到 .env 文件，再把里面的 KEY=VALUE 加载到环境变量
# 这样后面 os.getenv("XXX") 就能读到 .env 里配置的值
load_dotenv(find_dotenv())

# 初始化大语言模型（LLM）
# 这个 llm 实例会被所有节点共用，用来驱动 AI 推理
llm = ChatOpenAI(
    # model：模型 ID，从环境变量 LLM_MODEL_ID 读，默认 gpt-4o-mini
    # 你用的是硅基流动，所以 .env 里配的是 zai-org/GLM-4.5V
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),

    # api_key：API 密钥，从环境变量 LLM_API_KEY 读
    api_key=os.getenv("LLM_API_KEY"),

    # base_url：API 端点地址，从环境变量 LLM_BASE_URL 读
    # 默认是 OpenAI 官方地址，硅基流动配的是 https://api.siliconflow.cn/v1
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),

    # temperature：采样温度，0.7 表示有一定随机性
    # 0 = 确定性输出，越高越随机/有创造力
    temperature=0.7,

    # streaming=True：开启流式输出
    # 关键作用：即使节点里用的是 llm.invoke()（同步调用），
    # 内部也会走流式 API，从而触发 on_chat_model_stream 事件
    # 这样 astream_events 才能拿到逐字的 token
    streaming=True,
)

# 初始化 Tavily 搜索引擎客户端
# api_key 从环境变量 TAVILY_API_KEY 读（在 https://tavily.com 免费申请）
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================================
# 第四部分：定义三个工作流节点
# ============================================================================

# 每个节点都是一个普通函数，接收 state，返回要更新的字段（dict）
# LangGraph 会自动把返回的 dict 合并到全局 state 里

# ---------- 节点 1：understand_query_node ----------
# 功能：接收用户的原始问题，用 LLM 理解需求并生成搜索关键词
def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""

    # state["messages"][-1] 取最新的一条消息（就是用户刚输入的 HumanMessage）
    # .content 取出消息的文本内容
    user_message = state["messages"][-1].content

    # 构造给 LLM 的提示词，要求它完成两个任务：总结需求 + 生成搜索词
    # 用 f-string 把用户问题嵌入提示词
    understand_prompt = f"""分析用户的查询："{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    # 调用 LLM，传入 SystemMessage 作为提示词
    # llm.invoke() 是同步调用，会阻塞直到 LLM 返回完整结果
    # 注意：因为 streaming=True，invoke 内部实际走流式，但这里我们拿完整结果
    response = llm.invoke([SystemMessage(content=understand_prompt)])

    # response 是 AIMessage 对象，.content 取出文本内容
    response_text = response.content

    # 解析 LLM 的输出，提取搜索关键词
    # 默认用用户原始问题作为搜索词（防止解析失败）
    search_query = user_message
    # 如果 LLM 输出里包含"搜索词："，就取冒号后面的内容作为搜索词
    if "搜索词：" in response_text:
        # split("搜索词：")[1] 取冒号后的部分，.strip() 去掉首尾空白
        search_query = response_text.split("搜索词：")[1].strip()

    # 返回要更新的 state 字段
    # LangGraph 会把这些字段合并到全局 state：
    #   - user_query：存 LLM 的完整理解结果
    #   - search_query：存提取出的搜索词
    #   - step：标记当前步骤为 "understood"
    #   - messages：追加一条 AIMessage（通过 add_messages 合并），让对话历史里有 AI 的回复
    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")]
    }


# ---------- 节点 2：tavily_search_node ----------
# 功能：用 search_query 调用 Tavily API 进行真实联网搜索
def tavily_search_node(state: SearchState) -> dict:
    """步骤2：使用Tavily API进行真实搜索"""

    # 从 state 取出上一步生成的搜索关键词
    search_query = state["search_query"]

    try:
        # 打印搜索提示，让用户知道正在搜索什么
        print(f"🔍 正在搜索: {search_query}")

        # 调用 Tavily 搜索 API
        # query：搜索关键词
        # search_depth="basic"：基础搜索（更快，advanced 更全面但更慢）
        # max_results=5：最多返回 5 条结果
        # include_answer=True：让 Tavily 直接生成一个摘要答案
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )

        # 格式化搜索结果
        # Tavily 返回的 response 里，results 是一个列表，每个元素是一条搜索结果
        # 每条结果包含 title（标题）、url（链接）、content（内容摘要）
        results = response.get("results", [])

        # formatted 列表用来存格式化后的每条结果字符串
        formatted = []

        # enumerate(results, 1) 从 1 开始编号遍历结果
        for i, r in enumerate(results, 1):
            # 每条结果格式化为：[序号] 标题\n内容\n来源: 链接
            formatted.append(f"[{i}] {r.get('title', '')}\n{r.get('content', '')}\n来源: {r.get('url', '')}")

        # 用两个换行符把所有结果拼起来，形成一段完整的搜索结果文本
        search_results = "\n\n".join(formatted)

        # Tavily 可能还会直接返回一个 answer 字段（AI 生成的摘要）
        # 如果有，就把它放在最前面
        if response.get("answer"):
            search_results = f"摘要: {response['answer']}\n\n{search_results}"

        # 返回更新：搜索结果 + 步骤标记 + 消息
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！正在整理答案...")]
        }

    # 如果搜索过程中出现任何异常（网络错误、API 限流等）
    except Exception as e:
        # 返回错误信息，step 标记为 search_failed
        # 这样 answer 节点就能根据 step 判断走回退逻辑
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题...")]
        }


# ---------- 节点 3：generate_answer_node ----------
# 功能：基于搜索结果，用 LLM 生成最终答案
def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""

    # 根据 step 判断搜索是否成功
    if state["step"] == "search_failed":
        # 搜索失败：走回退策略，让 LLM 基于自身知识回答
        # 不依赖搜索结果，直接用用户问题作为提示
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功：基于搜索结果生成答案
        # 把用户问题和搜索结果一起喂给 LLM，让它综合回答
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
用户问题：{state['user_query']}
搜索结果：\n{state['search_results']}
请综合搜索结果，提供准确、有用的回答..."""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    # 返回最终答案
    # final_answer：LLM 生成的完整答案文本
    # step：标记为 completed
    # messages：把答案作为 AIMessage 追加到对话历史
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }


# ============================================================================
# 第五部分：构建工作流图
# ============================================================================

def create_search_assistant():
    """创建并编译搜索助手的工作流图"""

    # 创建一个 StateGraph 实例，传入状态类型 SearchState
    # 这告诉 LangGraph：这个图的 state 有哪些字段
    workflow = StateGraph(SearchState)

    # 添加三个节点：节点名 + 对应的处理函数
    # 节点名是字符串标识，后面连边时用
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 设置节点之间的边（执行顺序）
    # START → understand：从入口进入第一个节点
    workflow.add_edge(START, "understand")
    # understand → search：理解完问题后去搜索
    workflow.add_edge("understand", "search")
    # search → answer：搜索完后生成答案
    workflow.add_edge("search", "answer")
    # answer → END：生成答案后结束
    workflow.add_edge("answer", END)

    # 创建内存检查点（用来保存图的状态，实现多轮对话记忆）
    memory = InMemorySaver()

    # 编译图，传入检查点
    # compile() 会把节点和边组装成可执行的图对象 app
    app = workflow.compile(checkpointer=memory)

    # 返回编译好的图
    return app


# ============================================================================
# 第六部分：异步输入工具
# ============================================================================

# ainput 是 input() 的异步版本
# 因为 input() 是阻塞调用，会卡住事件循环
# 用 asyncio.to_thread 把它放到线程池里执行，就不会阻塞了
async def ainput(prompt: str = "") -> str:
    """异步包装 input()，避免阻塞事件循环"""
    return await asyncio.to_thread(input, prompt)


# ============================================================================
# 第七部分：主程序（多轮对话 + 流式输出）
# ============================================================================

async def main():
    """主函数：启动对话循环，支持流式输出"""

    # ---- 1. 检查环境变量是否配置 ----
    # 如果没配 LLM_API_KEY，提示用户去 .env 里设置
    if not os.getenv("LLM_API_KEY"):
        print("❌ 请在 .env 中设置 LLM_API_KEY")
    # 如果没配 TAVILY_API_KEY，提示用户去 tavily.com 申请
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 请在 .env 中设置 TAVILY_API_KEY，到 https://tavily.com 免费申请")

    # ---- 2. 创建搜索助手图 ----
    app = create_search_assistant()

    # ---- 3. 配置会话 ID ----
    # InMemorySaver 通过 thread_id 区分不同会话
    # 用同一个 thread_id，messages 就会自动累积，实现多轮对话记忆
    config = {"configurable": {"thread_id": "chat-session-1"}}

    # 打印启动提示（流程图风格：展示助手能力）
    print("🤖 智能搜索助手启动！")
    print("我会使用 Tavily API 为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("（输入 'quit' 退出）\n")

    # ---- 4. 对话主循环 ----
    while True:
        # 异步等待用户输入（流程图风格的提示语）
        raw = await ainput("🧑 您想了解什么：")
        # 去掉首尾空白字符
        raw = raw.strip()

        # 如果用户输入 quit/exit/q，退出循环
        if raw.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        # 如果输入为空，跳过本次循环
        if not raw:
            continue

        # ---- 5. 流式执行图（流程图风格输出）----
        # 用 astream_events(version="v2") 获取 token 级别的事件流
        # 按工作流的三个阶段分别展示：
        #   🧠 理解阶段：流式输出 LLM 对问题的理解和提炼出的搜索词
        #   🔍 搜索阶段：显示搜索关键词 + 搜索完成提示（tavily_search_node 内部打印"正在搜索"）
        #   💡 最终回答：流式输出 LLM 基于搜索结果生成的答案
        #
        # 关键事件类型：
        #   on_chat_model_start  : LLM 开始生成
        #   on_chat_model_stream : LLM 每生成一个 token 触发（逐字输出靠这个）
        #   on_chat_model_end    : LLM 生成结束
        #   on_chain_end         : 节点执行结束（非 LLM 节点如 search 用这个判断完成）
        #   metadata.langgraph_node : 当前事件属于哪个节点

        # current_node：记录当前正在执行的节点名
        current_node = None
        # answer：累积最终答案的完整文本
        answer = ""
        # understand_started：理解阶段是否已打印过阶段标题（避免重复）
        understand_started = False
        # search_done：搜索阶段是否已打印过完成提示（避免重复）
        search_done = False

        # ---- 流式内容清洗状态 ----
        # GLM-4.5V 等模型会输出特殊控制 token 和思考块，需要过滤：
        #   <|begin_of_box|> / <|end_of_box|> ：模型的盒子标记
        #   ```think ... ``` ：模型的内部思考过程，不展示给用户
        in_think_block = False  # 是否处于 ```think``` 思考块内部

        def clean_delta(delta: str) -> str:
            """清洗 LLM 流式输出，去除特殊 token 和思考块"""
            nonlocal in_think_block
            result = []
            i = 0
            while i < len(delta):
                # 过滤 <|begin_of_box|> 标记
                if delta.startswith("<|begin_of_box|>", i):
                    i += len("<|begin_of_box|>")
                    continue
                # 过滤 <|end_of_box|> 标记
                if delta.startswith("<|end_of_box|>", i):
                    i += len("<|end_of_box|>")
                    continue
                # 进入思考块：```think 开头
                if not in_think_block and delta.startswith("```think", i):
                    in_think_block = True
                    i += len("```think")
                    continue
                # 处于思考块内：跳过内容直到遇到 ```
                if in_think_block:
                    end_idx = delta.find("```", i)
                    if end_idx != -1:
                        # 找到结束标记，退出思考块
                        in_think_block = False
                        i = end_idx + 3
                    else:
                        # 整个剩余内容都在思考块内，全部跳过
                        i = len(delta)
                    continue
                # 普通内容，保留
                result.append(delta[i])
                i += 1
            return "".join(result)

        # 异步遍历图的事件流
        async for ev in app.astream_events(
            # 输入：用户最新的问题（作为 HumanMessage）
            {"messages": [HumanMessage(content=raw)]},
            # 配置：thread_id 保持多轮上下文
            config=config,
            # 事件流版本，v2 是最新的
            version="v2",
        ):
            # 取出事件类型
            event = ev["event"]

            # 从 metadata 里取出当前节点名
            # 有些事件没有 langgraph_node（如图级别事件），所以用 .get 安全取值
            node = ev.get("metadata", {}).get("langgraph_node")
            # 如果有节点名，更新 current_node
            if node:
                current_node = node

            # ============================================================
            # 阶段一：🧠 理解阶段
            # 流式输出 LLM 对问题的理解过程（包括需求总结和搜索词）
            # ============================================================
            if current_node == "understand":
                # LLM 开始生成时，打印阶段标题
                if event == "on_chat_model_start" and not understand_started:
                    print("\n� 理解阶段: ", end="", flush=True)
                    understand_started = True
                # LLM 每生成一个 token，逐字打印（流式）
                elif event == "on_chat_model_stream":
                    delta = ev["data"]["chunk"].content
                    if delta:
                        # 清洗特殊 token 和思考块后再打印
                        cleaned = clean_delta(delta)
                        if cleaned:
                            print(cleaned, end="", flush=True)
                # LLM 生成结束，换行进入下一阶段
                elif event == "on_chat_model_end":
                    print()

            # ============================================================
            # 阶段二：🔍 搜索阶段
            # tavily_search_node 内部会打印 "🔍 正在搜索: {搜索词}"
            # 这里在节点执行结束时打印搜索完成提示
            # ============================================================
            elif current_node == "search":
                # 节点执行结束时打印搜索完成提示
                if event == "on_chain_end" and not search_done:
                    print("搜索阶段：✅ 搜索完成！找到了相关信息，正在为您整理答案...")
                    search_done = True

            # ============================================================
            # 阶段三：💡 最终回答
            # 流式输出 LLM 基于搜索结果生成的最终答案
            # ============================================================
            elif current_node == "answer":
                # LLM 开始生成时，打印阶段标题
                if event == "on_chat_model_start":
                    print("\n💡 最终回答:")
                # LLM 每生成一个 token，逐字打印（流式打字机效果）
                elif event == "on_chat_model_stream":
                    delta = ev["data"]["chunk"].content
                    if delta:
                        # 清洗特殊 token 和思考块后再打印
                        cleaned = clean_delta(delta)
                        if cleaned:
                            print(cleaned, end="", flush=True)
                            # 累积到 answer 变量（目前用于统计，可扩展为保存到历史）
                            answer += cleaned

        # 一轮对话结束，打印空行做视觉分隔
        print("\n")


# ============================================================================
# 第八部分：程序入口
# ============================================================================

# if __name__ == "__main__" 是 Python 的标准入口判断
# 只有直接运行这个文件时才会执行，被 import 时不会
if __name__ == "__main__":
    # asyncio.run(main()) 启动异步事件循环，执行 main() 协程
    asyncio.run(main())
