from langchain_core.messages import SystemMessage, ContentBlock


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