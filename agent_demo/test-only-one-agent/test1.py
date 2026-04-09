import json

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@tool(description='获取城市天气')
def mytool(input: str) -> str:
    return f'{input}: 天气总是晴朗!'


checkpoint = InMemorySaver()

my_agent = create_agent(
    model="deepseek-reasoner",
    tools=[mytool],
    system_prompt="你是一个聊天小助手",
    checkpointer=checkpoint,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "mytool": {
                    "allowed_decisions": ["approve", "reject"],
                }
            }
        )
    ]
)

config = RunnableConfig(configurable={"thread_id": "1"})


def run_turn(query: str, operation: str):
    ret = my_agent.invoke({"messages": [HumanMessage(query)]}, config=config)
    if "__interrupt__" in ret:
        ret = my_agent.invoke(
            Command(resume={"decisions": [{"type": operation}]}),
            config=config,
        )
    return ret


ret = run_turn("北京天气怎么样?", "approve")
# print(json.dumps(messages_to_dict(ret['messages']), ensure_ascii=False, indent=2))
for msg in ret["messages"]:
    msg.pretty_print()

ret = run_turn("那上海呢", "approve")
for msg in ret["messages"]:
    msg.pretty_print()
# print(json.dumps(messages_to_dict(ret['messages']), ensure_ascii=False, indent=2))