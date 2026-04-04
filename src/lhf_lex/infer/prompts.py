from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, List, TypedDict

# Ultra-minimal system prompt: avoids domain/style/length tokens.
SYSTEM_CONTINUE_V1 = (
    "Reply only with the continuation of the user's text. "
    "Do not repeat any of the user's text. No preface."
)

SchemaName = Literal["none", "chat_v1"]

@dataclass(frozen=True)
class PromptSchema:
    name: SchemaName
    system: str | None = None  # None when using "none"

PROMPT_SCHEMAS: Dict[SchemaName, PromptSchema] = {
    "none": PromptSchema(name="none", system=None),
    "chat_v1": PromptSchema(name="chat_v1", system=SYSTEM_CONTINUE_V1),
}

def render_prompt(schema: SchemaName, user_text: str) -> str:
    """
    Return the fully realised prompt string that will be passed to the model.
    - "none": returns user_text as-is (no tags).
    - "chat_v1": returns <|system|>...<|user|>...<|assistant|>\\n
    Notes:
    - Keep decoding controls (max_new_tokens, temperature, top_p) outside the prompt.
    - Do not append trailing spaces to user_text; log raw inputs as given.
    """
    if schema == "none":
        return user_text
    if schema == "chat_v1":
        sysmsg = SYSTEM_CONTINUE_V1
        return f"<|system|>\n{sysmsg}\n<|user|>\n{user_text}\n<|assistant|>\n"
    raise ValueError(f"Unknown schema: {schema}")

# Optional: native HF chat-template path (use instead of render_prompt if preferred).
class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

def render_messages(schema: SchemaName, user_text: str) -> List[ChatMessage]:
    """
    Return messages for tokenizer.apply_chat_template(..., add_generation_prompt=True).
    Raises for 'none' because base models expect plain continuation.
    """
    if schema == "chat_v1":
        return [
            {"role": "system", "content": SYSTEM_CONTINUE_V1},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ""},  # empty turn to start generation
        ]
    if schema == "none":
        raise ValueError("render_messages not supported for 'none' (use render_prompt).")
    raise ValueError(f"Unknown schema: {schema}")
