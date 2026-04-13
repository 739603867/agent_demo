from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agent_demo.multi_agent_1.backend.file_system import FileSystemProtocol
from agent_demo.multi_agent_1.middle_ware.file_middleware import FilesystemMiddleware
from agent_demo.multi_agent_1.middle_ware.subagent_middleware import SubAgentMiddleware, SubAgent

checkpoint = InMemorySaver()
my_multi_agent = create_agent(
    model="deepseek-reasoner",
    system_prompt="你是一个解决代码开发问题的专家",
    checkpointer=checkpoint,
    middleware=[
        FilesystemMiddleware(backend=FileSystemProtocol(root_dir="/Users/bytedance/my-agent/agent_test/", virtual_mod=False)),
        SubAgentMiddleware(
            subagents=[
                SubAgent(
                    model="deepseek-reasoner",
                    name="代码指导小助手",
                    desc="专注于代码咨询的任务处理",
                    system_prompt="你是一个代码分析专家，当用户咨询代码语法时，可以通过这个子agent进行处理",
                    tools=[],
                )
            ]
        ),
    ]
)

config = RunnableConfig(configurable={"thread_id": "test-multi-agent1"})

ret = my_multi_agent.invoke(dict(messages=[HumanMessage('帮我检查一下agent_demo/multi_agent_1/backend/protocol.py中的代码是否有问题')]), config=config)
for msg in ret["messages"]:
    msg.pretty_print()
