import time

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import LLMMathChain

from langchain_community.chat_models.ollama import ChatOllama


chat_model = ChatOllama(
    base_url="http://hercules.local:11434",
    model="gemma:2b",
)

llm_math = LLMMathChain(llm=chat_model, verbose=True)

# agent_prompt = hub.pull("hwchase17/react")

# agent_tools = [

# ]

# # Decide which tools to use (doesn't actually run them)
# agent = create_react_agent(chat_model, agent_tools, agent_prompt)

# agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)

print("Start.")
ts = time.time()
print(llm_math.invoke("2+2="))
te = time.time()
print("End. Time taken: %.3f s" % (te-ts))
