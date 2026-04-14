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


def invoke_agent(user_input: str):
    return my_multi_agent.invoke(
        dict(messages=[HumanMessage(user_input)]),
        config=config,
    )


def print_agent_messages(messages) -> None:
    for msg in messages:
        msg.pretty_print()


def run_console() -> None:
    print("Multi-agent console started. Type your question.")
    print("Type 'exit' or 'quit' to leave. Press Ctrl+C to exit anytime.")

    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print("\nConsole closed.")
            break
        except KeyboardInterrupt:
            print("\nConsole interrupted. Bye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        try:
            result = invoke_agent(user_input)
            print_agent_messages(result["messages"])
        except Exception as exc:
            print(f"[agent error] {exc}")


if __name__ == "__main__":
    run_console()
