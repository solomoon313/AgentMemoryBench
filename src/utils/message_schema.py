"""
Message format compatibility layer - unified handling of different message object types

This module provides unified message parsing utilities for handling multiple message formats:
- Plain dicts
- Objects wrapped in a Pydantic RootModel
- Pydantic model objects
- Non-chat messages such as RewardHistoryItem

It also provides memory-mechanism utilities shared across all memory modules:
- Injecting memory content into messages
- Extracting the original question from augmented messages
- Checking whether content contains injected memory

All memory modules should use these shared utilities rather than implementing their own.
"""
from typing import Any, Dict, List, Optional


def extract_message_info(msg: Any) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Extract role, content, and the full message dict from a message object.

    Handles multiple message formats:
    - Plain dict: used directly
    - RootModel-wrapped object: accessed via the .root attribute
    - Non-chat messages (e.g. RewardHistoryItem with a reward but no role): returns None
    - Other Pydantic models: converted to dict via model_dump

    Args:
        msg: message object — can be a dict, a Pydantic model, or another type

    Returns:
        (role, content, msg_dict) tuple:
        - role: message role ("user", "assistant", "system", etc.), or None if not extractable
        - content: message content string, or "" if not extractable
        - msg_dict: full message dict, or None if conversion fails

    Examples:
        >>> # Plain dict
        >>> extract_message_info({"role": "user", "content": "Hello"})
        ("user", "Hello", {"role": "user", "content": "Hello"})

        >>> # Pydantic model
        >>> from pydantic import BaseModel
        >>> class Message(BaseModel):
        ...     role: str
        ...     content: str
        >>> msg = Message(role="assistant", content="Hi")
        >>> extract_message_info(msg)
        ("assistant", "Hi", {"role": "assistant", "content": "Hi"})

        >>> # Non-chat message (e.g. RewardHistoryItem)
        >>> extract_message_info({"reward": 1.0})
        (None, None, None)
    """
    # Plain dict: use directly
    if isinstance(msg, dict):
        return msg.get("role"), msg.get("content", ""), msg

    # RootModel-wrapped object: access via .root
    if hasattr(msg, 'root'):
        root = msg.root
        # RewardHistoryItem has a reward attribute but no role attribute — skip it
        if hasattr(root, 'reward') and not hasattr(root, 'role'):
            return None, None, None
        # If root is a plain dict, use directly
        if isinstance(root, dict):
            return root.get("role"), root.get("content", ""), root
        # If root is a Pydantic model, convert to dict
        if hasattr(root, 'model_dump'):
            root_dict = root.model_dump(exclude_none=True)
            return root_dict.get("role"), root_dict.get("content", ""), root_dict

    # Non-chat message objects such as RewardHistoryItem (reward but no role): skip
    if hasattr(msg, 'reward') and not hasattr(msg, 'role'):
        return None, None, None

    # Pydantic model: convert to dict
    if hasattr(msg, 'model_dump'):
        msg_dict = msg.model_dump(exclude_none=True)
        return msg_dict.get("role"), msg_dict.get("content", ""), msg_dict

    # Fallback: try attribute access
    if hasattr(msg, 'role') and hasattr(msg, 'content'):
        return getattr(msg, 'role', None), getattr(msg, 'content', ""), None

    return None, None, None


def is_chat_message(msg: Any) -> bool:
    """
    Return True if the message is a chat message (has a role and content).

    Args:
        msg: message object

    Returns:
        True if it is a chat message, False otherwise

    Examples:
        >>> is_chat_message({"role": "user", "content": "Hello"})
        True
        >>> is_chat_message({"reward": 1.0})
        False
    """
    role, _, _ = extract_message_info(msg)
    return role is not None


def filter_chat_messages(messages: list[Any]) -> list[Dict[str, Any]]:
    """
    Filter all chat messages from a message list and convert them to dicts.

    Args:
        messages: list of message objects

    Returns:
        list of chat message dicts

    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"reward": 1.0},
        ...     {"role": "assistant", "content": "Hi"}
        ... ]
        >>> filter_chat_messages(messages)
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    """
    chat_messages = []
    for msg in messages:
        role, content, msg_dict = extract_message_info(msg)
        if role is not None:
            # Use the full dict if available; otherwise construct one
            if msg_dict is not None:
                chat_messages.append(msg_dict)
            else:
                chat_messages.append({"role": role, "content": content})
    return chat_messages


# ============================================================================
# Shared utilities for memory mechanisms
# ============================================================================

# Separator used to mark the boundary of the original question in where=front mode
ORIGINAL_QUESTION_SEPARATOR = "--- Original Question Below ---"


def enhance_messages_with_memory(
    messages: List[Dict[str, Any]],
    memory_content: str,
    where: str = "tail",
) -> List[Dict[str, Any]]:
    """
    Inject memory content into the first user message.

    This is the shared logic for all memory mechanisms: find the first user message
    and insert memory content either before or after the question based on `where`.

    Args:
        messages: original message list
        memory_content: pre-formatted memory text to inject
        where: injection position
            - "tail": memory appended after the original question (default)
            - "front": memory prepended before the original question, separated by a delimiter

    Returns:
        augmented message list (a copy of the original)

    Examples:
        >>> messages = [{"role": "user", "content": "What is 1+1?"}]
        >>> memory = "Example: 2+2=4"
        >>> enhanced = enhance_messages_with_memory(messages, memory, where="tail")
        >>> enhanced[0]["content"]
        'What is 1+1?\\n\\nExample: 2+2=4'

        >>> enhanced = enhance_messages_with_memory(messages, memory, where="front")
        >>> enhanced[0]["content"]
        'Example: 2+2=4\\n\\n--- Original Question Below ---\\n\\nWhat is 1+1?'
    """
    enhanced = list(messages) if messages is not None else []

    if not memory_content:
        return enhanced

    # Find the first user message
    for i, msg in enumerate(enhanced):
        role, content, msg_dict = extract_message_info(msg)
        if role == "user":
            content = content if content else ""

            if where == "front":
                # where=front: memory first, then delimiter, then original question
                new_content = f"{memory_content}\n\n{ORIGINAL_QUESTION_SEPARATOR}\n\n{content}"
            else:  # tail
                # where=tail: original question first, then memory
                new_content = f"{content}\n\n{memory_content}"

            # Spread msg_dict to correctly handle Pydantic models
            enhanced[i] = {
                **msg_dict,
                "content": new_content
            }
            break

    return enhanced


def extract_original_question(
    messages: List[Dict[str, Any]],
    where: str = "tail",
    template_titles: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Extract the original question from messages that may contain injected memory.

    Handles three cases:
    1. Unaugmented message (no memory injected) -> return the first user message directly
    2. where=front mode -> extract the original question using the separator
    3. where=tail mode -> split on the template title and take the part before it

    Args:
        messages: message list (may contain injected memory)
        where: injection position matching the one used in enhance_messages_with_memory
            - "tail": memory is after the question
            - "front": memory is before the question
        template_titles: list of template title strings used to identify injected memory
            in where=tail mode, e.g. ["Here are some examples", "Based on your previous interactions"]

    Returns:
        original question text, or None if it cannot be extracted

    Examples:
        >>> # Unaugmented message
        >>> messages = [{"role": "user", "content": "What is 1+1?"}]
        >>> extract_original_question(messages)
        'What is 1+1?'

        >>> # where=front augmented message
        >>> messages = [{"role": "user", "content": "Memory\\n\\n--- Original Question Below ---\\n\\nWhat is 1+1?"}]
        >>> extract_original_question(messages, where="front")
        'What is 1+1?'

        >>> # where=tail augmented message
        >>> messages = [{"role": "user", "content": "What is 1+1?\\n\\nHere are some examples"}]
        >>> extract_original_question(messages, where="tail", template_titles=["Here are some examples"])
        'What is 1+1?'
    """
    template_titles = template_titles or []

    for msg in messages:
        role, content, _ = extract_message_info(msg)
        if role == "user":
            content = content if content else ""

            # Case 1: check for the standard separator (where=front mode)
            if ORIGINAL_QUESTION_SEPARATOR in content:
                parts = content.split(ORIGINAL_QUESTION_SEPARATOR, 1)
                if len(parts) > 1:
                    question = parts[1].strip()
                    return question if question else None
                # Nothing after the separator
                return None

            # Case 2: where=tail mode — check for any template title
            if where == "tail" and template_titles:
                for template_title in template_titles:
                    if template_title in content:
                        # Original question is before the template title
                        question = content.split(template_title)[0].strip()
                        return question if question else None

            # Case 3: unaugmented message — return content directly
            return content

    return None


def is_memory_content(content: str, template_titles: List[str]) -> bool:
    """
    Return True if the content contains injected memory.

    Checks for either:
    1. The standard separator (where=front mode)
    2. Any of the given template titles

    Args:
        content: content string to check
        template_titles: list of template title strings

    Returns:
        True if memory is present, False otherwise

    Examples:
        >>> is_memory_content("Normal question", ["Examples:"])
        False

        >>> is_memory_content("Examples:\\nSome examples\\n\\nNormal question", ["Examples:"])
        True

        >>> is_memory_content("Memory\\n\\n--- Original Question Below ---\\n\\nQuestion", ["Examples:"])
        True
    """
    if not content:
        return False

    # Check for the standard separator
    if ORIGINAL_QUESTION_SEPARATOR in content:
        return True

    # Check for template titles
    for template_title in template_titles:
        if template_title in content:
            return True

    return False


def extract_question_from_history(
    history: List[Dict[str, Any]],
    where: str = "tail",
    template_titles: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Extract the original question from a full conversation history.

    This is an alias for extract_original_question with clearer semantics.
    Intended for use inside update_memory when extracting the question from history.

    Args:
        history: conversation history
        where: injection position
        template_titles: list of template title strings

    Returns:
        original question text
    """
    return extract_original_question(history, where, template_titles)
