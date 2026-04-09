from collections.abc import Sequence, Callable, Awaitable
from typing import TypedDict, NotRequired, Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, InterruptOnConfig
from langchain.agents.middleware.tool_selection import DEFAULT_SYSTEM_PROMPT
from langchain.agents.middleware.types import ResponseT, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, ContentBlock
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from langgraph.typing import ContextT
from pydantic import BaseModel, Field


class SubAgent(TypedDict):
    model: NotRequired[str | BaseChatModel]
    name: str
    desc: str
    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    system_prompt: str
    middleware: NotRequired[list[AgentMiddleware]]
    skills: NotRequired[list[str]]
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]


class CompliedSubAgent(TypedDict):
    name: str
    desc: str
    runnable: Runnable


DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


class TaskToolSchema(BaseModel):
    desc: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )
    subagent_type: str = Field(
        description="The type of subagent to use. Must be one of the available agent types listed in the tool description.")


# 在将 state 传给 subagent，以及从 subagent 返回更新时，需要排除的 state 键。
#
# 返回更新时：
# 1. messages 键会被显式处理，以确保只包含最终消息
# 2. todos 和 structured_response 键会被排除，因为它们没有定义 reducer，
#    而且也没有清晰的语义可以从 subagent 返回给主 agent
# 3. skills_metadata 和 memory_contents 键会通过各自 state schema 上的
#    PrivateStateAttr 注解自动从 subagent 输出中排除。但在调用 subagent
#    时，也必须从 runtime.state 中显式过滤它们，防止父级 state 泄漏到子 agent
#    （例如，通用 subagent 会通过 SkillsMiddleware 加载自己的 skills）。
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "skills_metadata", "memory_contents"}

TASK_TOOL_DESCRIPTION = """启动一个临时子代理，用隔离的上下文窗口来处理复杂、多步骤、彼此独立的任务。
可用的 agent 类型及其可访问的工具：
{available_agents}

使用 Task 工具时，你必须指定 `subagent_type` 参数来选择要使用的子代理类型。

## 使用说明：
1. 只要可能，就并行启动多个 agent，以最大化性能；要这样做，请在一条消息里发起多个工具调用
2. 当 agent 完成后，它会返回一条消息给你。agent 返回的结果用户是看不到的。若要把结果展示给用户，你应该向用户发送一条简洁的总结消息
3. 每次 agent 调用都是无状态的。你不能继续给这个 agent 发额外消息，agent 也不能在最终报告之外与你通信。因此，你的 prompt 必须包含非常详细的任务描述，让 agent 可以自主完成任务；同时你应明确说明希望它在最终且唯一的一条返回消息中带回哪些信息
4. 一般来说，应当信任 agent 的输出
5. 要清楚告诉 agent 你希望它是创建内容、执行分析，还是仅做研究（如搜索、读文件、抓网页等），因为它并不知道你的真实意图
6. 如果 agent 的描述中提到应主动使用它，那么即使用户没明确要求，你也应根据判断尽量主动使用
7. 如果只提供了一个通用 agent，那么所有任务都应使用它。它非常适合隔离上下文与 token 消耗，并完成具体的复杂任务，因为它拥有与主 agent 相同的全部能力

### 通用 agent 的示例用法：

<example_agent_descriptions>
"general-purpose": 用于通用任务，它拥有与主 agent 相同的全部工具。
</example_agent_descriptions>

<example>
用户："我想研究勒布朗·詹姆斯、迈克尔·乔丹和科比·布莱恩特各自的成就，并进行比较。"
助手：*并行使用 task 工具，分别启动隔离的研究任务来研究这三位球员*
助手：*整合三个独立研究任务的结果，并回复用户*
<commentary>
研究本身就是一个复杂的多步骤任务。
对每个球员的研究彼此独立，不依赖其他人的研究结果。
助手使用 task 工具把复杂目标拆成三个隔离任务。
每个研究任务只需要关注某一位球员相关的上下文和 token，最后只返回综合后的信息作为工具结果。
这意味着每个研究任务都可以深入投入上下文和 token 去研究单个球员，而最终比较时主线程只需要综合结果，从长期看更省 token。
</commentary>
</example>

<example>
用户："分析一个大型代码仓库中的安全漏洞，并生成报告。"
助手：*启动一个 task 子代理来分析该仓库*
助手：*收到报告并整合结果，输出最终摘要*
<commentary>
虽然这里只有一个任务，但它本身规模大、上下文重，很适合交给子代理处理。
这样可以避免主线程被过多细节压满。
如果用户之后继续追问，我们也只需要参考一份简洁报告，而不用依赖整段分析历史和全部工具调用过程，这样更好，也更省时间和成本。
</commentary>
</example>

<example>
用户："帮我安排两场会议，并为每场会议准备议程。"
助手：*并行调用 task 工具，分别为两场会议各启动一个 task 子代理来准备议程*
助手：*返回最终的会议安排和议程*
<commentary>
每个任务单独看都不复杂，但子代理可以把每场会议的议程准备工作隔离开。
每个子代理只需要关注自己那一场会议的议程。
</commentary>
</example>

<example>
用户："我想在 Dominos 订一个披萨，在 McDonald's 订一个汉堡，在 Subway 订一份沙拉。"
助手：*直接并行调用工具分别下单*
<commentary>
这里没有使用 task 工具，因为目标非常简单清晰，只需要少量直接的工具调用。
这种情况下，直接完成任务比使用 `task` 工具更好。
</commentary>
</example>

### 自定义 agent 的示例用法：

<example_agent_descriptions>
"content-reviewer": 在你完成较大篇幅内容或文档创作后使用这个 agent
"greeting-responder": 当用户打招呼时，用这个 agent 以友好的笑话进行回应
"research-analyst": 用这个 agent 对复杂主题做深入研究
</example_agent_description>

<example>
用户："请写一个判断一个数是否为质数的函数"
助手：当然，我来写一个判断质数的函数
助手：先用 Write 工具写这个函数
助手：我将使用 Write 工具写下如下代码：
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
由于已经创建了较重要的内容，并且任务已完成，此时应使用 content-reviewer agent 来审查这份代码
</commentary>
助手：现在我来使用 content-reviewer agent 审查这段代码
助手：使用 Task 工具启动 content-reviewer agent
</example>

<example>
用户："你能帮我研究不同可再生能源的环境影响，并写一份全面报告吗？"
<commentary>
 这是一个复杂的研究任务，非常适合交给 research-analyst agent 做深入分析
 </commentary>
 助手：我来帮你研究可再生能源的环境影响。先让我使用 research-analyst agent 对这个主题做全面研究。
 助手：使用 Task 工具启动 research-analyst agent，并提供详细说明，告诉它需要研究什么，以及报告应该如何组织
 </example>

 <example>
 用户："你好"
 <commentary>
 因为用户是在打招呼，所以应使用 greeting-responder agent 以友好的笑话作出回应
 </commentary>
 助手："我将使用 Task 工具启动 greeting-responder agent"
 </example>"""

TASK_SYSTEM_PROMPT = """## `task`（子代理启动器）
你可以使用一个名为 `task` 的工具来启动短生命周期的子代理，这些子代理用于处理隔离的任务。这些 agent 是临时的——它们只在任务执行期间存在，并返回单一结果。
什么时候该使用 task 工具：
  - 当一个任务复杂、包含多个步骤，并且可以被完整地独立委托出去时
  - 当一个任务与其他任务相互独立，且可以并行执行时
  - 当一个任务需要高度专注的推理，或者需要消耗大量 token / 上下文，若放在主线程会使主线程变得臃肿时
  - 当把任务放进隔离环境中执行会更可靠时（例如代码执行、结构化搜索、数据格式化）
  - 当你只关心子代理最终产出的结果，而不关心它完成过程中的中间步骤时
    （例如：做大量研究后只返回综合报告；执行一系列计算或查找后只返回简洁且相关的答案）
子代理生命周期：
  1. **启动（Spawn）** → 提供清晰的角色、说明和期望输出
  2. **运行（Run）** → 子代理自主完成任务
  3. **返回（Return）** → 子代理返回一个单一的结构化结果
  4. **整合（Reconcile）** → 将结果整合或综合回主线程

什么时候不要使用 task 工具：
  - 如果你需要在子代理完成后仍能看到它的中间推理过程或执行步骤（task 工具会隐藏这些过程）
  - 如果任务很简单（只需要几个工具调用或一次简单查询）
  - 如果委托出去并不能减少 token 消耗、复杂度或上下文切换
  - 如果拆分任务只会增加延迟，却没有实际收益

## 关于 Task 工具，你需要记住的重要使用原则
  - 只要可能，就尽量并行化你的工作。这既适用于普通 tool calls，也适用于 task。只要步骤彼此独立，就应并行调用工具或启动多个 task（子代理）来更快完成任务。这能节省用户时间，而这非常重要。
  - 当你面对一个由多个独立部分组成的目标时，要记得使用 `task` 工具把这些独立任务隔离开来。
  - 只要你遇到一个复杂任务，它需要多个步骤，并且它与 agent 需要完成的其他任务相互独立，你就应该使用 `task` 工具。这些子代理能力很强，也很高效。
"""

DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent 用于研究复杂问题、搜索文件和内容，以及执行多步骤任务的通用型 agent。当你在搜索某个关键词或文件，并且不确定前几次尝试就能找到正确结果时，应该使用这个 agent 来帮你完成搜索。 这个 agent 拥有与主 agent 相同的全部工具访问权限"  # noqa: E501

# Base spec for general-purpose subagent (caller adds model, tools, middleware)
GENERAL_PURPOSE_SUBAGENT: SubAgent = dict(
    name="general-purpose",
    model="deepseek-reasoner",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    desc=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
)


class _SubagentSpec(TypedDict):
    name: str
    description: str
    runnable: Runnable


def _build_task_tool(subagents: list[_SubagentSpec], task_desc: str | None = None) -> BaseTool:
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in subagents}
    subagent_desc = "\n".join(f"- {spec['name']}: {spec['description']}" for spec in subagents)

    if task_desc is None:
        description = DEFAULT_GENERAL_PURPOSE_DESCRIPTION.format(available_agents=subagent_desc)
    elif '{available_agents}' in task_desc:
        description = task_desc.format(available_agents=subagent_desc)
    else:
        description = task_desc

    def _run_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        if 'messages' not in result:
            error_msg = (
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        # 去掉尾随空白，防止 Anthropic API 出错
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(subagent_type: str, desc: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        subagent = subagent_graphs[subagent_type]
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state['messages'] = [HumanMessage(content=desc)]
        return subagent, subagent_state

    def task(subagent_type: str, desc: str, runtime: ToolRuntime) -> str | Command:
        if subagent_type not in subagent_graphs:
            return f"can not find sub agent type:{subagent_type}"

        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, desc, runtime)
        result = subagent.invoke(subagent_state)
        return _run_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
            description: str,
            subagent_type: str,
            runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state)
        return _run_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name='task',
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=TaskToolSchema,
    )


class SubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """通过 `task` 工具为 agent 提供 subagent 的中间件。

       该中间件会给 agent 增加一个 `task` 工具，可用来调用 subagent。
       subagent 适合处理需要多步完成的复杂任务，或需要大量上下文才能解决的任务。

       subagent 的一个主要优点是，它们可以完成多步任务，
       然后向主 agent 返回简洁、干净的响应。

       subagent 也很适合处理不同的专业领域，这些领域只需要更窄的一组工具和更聚焦的上下文。

       Args:
           subagents: 完整指定的 subagent 配置列表。每个 `SubAgent`
               都必须指定 `model` 和 `tools`。各 subagent 上可选的
               `interrupt_on` 配置也会被尊重。
           system_prompt: 追加到主 agent 系统提示中的说明文本，
               用于指导如何使用 `task` 工具。
           task_description: `task` 工具的自定义描述。

       Example:
           ```python
           from deepagents.middleware import SubAgentMiddleware
           from langchain.agents import create_agent

           agent = create_agent(
               "openai:gpt-4o",
               middleware=[
                   SubAgentMiddleware(
                       backend=my_backend,
                       subagents=[
                           {
                               "name": "researcher",
                               "description": "Research agent",
                               "system_prompt": "You are a researcher.",
                               "model": "openai:gpt-4o",
                               "tools": [search_tool],
                           }
                       ],
                   )
               ],
           )
           ```

       .. deprecated::
           以下参数已弃用，并将在 0.5.0 版本中移除：
           `default_model`、`default_tools`、`default_middleware`、
           `default_interrupt_on`、`general_purpose_agent`。请改用 `backend` 和 `subagents`。
       """
    # 用于运行时校验的合法弃用 ≥ 名称
    _VALID_DEPRECATED_KWARGS = frozenset(
        {
            "default_model",
            "default_tools",
            "default_middleware",
            "default_interrupt_on",
            "general_purpose_agent",
        }
    )

    def __init__(
            self,
            *,
            subagents: Sequence[SubAgent | CompliedSubAgent] | None = None,
            system_prompt: str | None = TASK_SYSTEM_PROMPT,
            task_description: str | None = None,
    ) -> None:
        super().__init__()

        self._subagents = subagents
        if subagents is None:
            msg = 'empty subagents is invalid'
            raise ValueError(msg)

        subagent_specs = self._get_subagents()

        task_tool = _build_task_tool(subagent_specs, task_description)
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagent_specs)
            self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

        self.tools = [task_tool]


    def _get_subagents(self) -> list[_SubagentSpec]:
        specs: list[_SubagentSpec] = []
        for subagent in self._subagents:
            if 'runnable' in subagent:
                # CompiledSubAgent：直接使用
                compiled = cast("CompliedSubAgent", subagent)
                specs.append(
                    {"name": compiled["name"], "description": compiled["desc"], "runnable": compiled["runnable"]})
                continue

            if 'model' not in subagent:
                msg = f"SubAgent '{subagent['name']}' must specify 'model'"
                raise ValueError(msg)
            if 'tools' not in subagent:
                msg = f"SubAgent '{subagent['name']}' must specify 'tools'"
                raise ValueError(msg)

            model = init_chat_model(subagent['model'])
            specs.append(
                {
                    "name": subagent["name"],
                    "description": subagent["desc"],
                    "runnable": create_agent(
                        model,
                        system_prompt=subagent["system_prompt"],
                        tools=subagent["tools"],
                        name=subagent["name"],
                    ),
                }
            )
        return specs

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if self.system_prompt is not None:
            new_system_msg = append_to_system_message(request.system_message, self.system_prompt)
            request.system_message = new_system_msg
            return handler(request.override(system_message=new_system_msg))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """（异步）更新系统消息，补充 subagent 的使用说明。"""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)

def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """将文本追加到系统消息中。

    Args:
        system_message: 现有的系统消息，或 `None`。
        text: 要追加到系统消息中的文本。

    Returns:
        追加文本后的新 `SystemMessage`。
    """
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)