"""GInfer function tools, without computer-action or sampling policy."""

from typing import Any

import httpx


def text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        item.get("type") in {"input_text", "output_text", "text"} for item in content
    ):
        return "".join(item["text"] for item in content)
    raise ValueError("GInfer function provider requires text messages")


def to_chat_messages(items: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "function_call":
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": None})
            messages[-1].setdefault("tool_calls", []).append(
                {"id": item["call_id"], "type": "function", "function": {
                    "name": item["name"], "arguments": item["arguments"]
                }}
            )
        elif kind == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"],
                             "content": text_content(item["output"])})
        elif kind == "message" and item.get("role") in {
            "system", "developer", "user", "assistant"
        }:
            message = {"role": item["role"], "content": text_content(item.get("content"))}
            if item.get("reasoning_content"):
                message["reasoning_content"] = item["reasoning_content"]
            messages.append(message)
        else:
            raise ValueError(f"Unsupported GInfer input item: {kind}")
    return messages


def from_chat_messages(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for message in messages:
        if message["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": text_content(message["content"])})
            continue
        calls = message.get("tool_calls") or []
        if not calls or message.get("content") or message.get("reasoning_content"):
            item = {"type": "message", "role": message["role"],
                    "content": [{"type": "output_text" if message["role"] == "assistant"
                                 else "input_text", "text": text_content(message.get("content"))}]}
            if message.get("reasoning_content"):
                item["reasoning_content"] = message["reasoning_content"]
            items.append(item)
        for call in calls:
            if call.get("type") != "function":
                raise ValueError("GInfer returned a non-function tool call")
            function = call["function"]
            items.append({"type": "function_call", "call_id": call["id"],
                          "name": function["name"], "arguments": function["arguments"]})
    return items


class GInferConfig:
    """ComputerAgent provider for an explicitly assigned OpenAI-compatible server."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client

    async def predict_step(self, messages, model, tools=None, max_retries=None,
                           stream=False, computer_handler=None, _on_api_start=None,
                           _on_api_end=None, _on_usage=None, _on_screenshot=None,
                           api_base=None, api_key=None, use_prompt_caching=False, **generation):
        if stream or not api_base:
            raise ValueError("GInfer requires an assigned endpoint and non-streaming steps")
        if any(tool.get("type") != "function" for tool in tools or []):
            raise ValueError("GInfer provider supports function tools only")
        body = {"model": model, "messages": to_chat_messages(messages), **generation}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if _on_api_start:
            await _on_api_start(body)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        url = api_base.rstrip("/") + "/chat/completions"

        async def request(client):
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()

        if self.client:
            raw = await request(self.client)
        else:
            async with httpx.AsyncClient(timeout=None) as client:
                raw = await request(client)
        if _on_api_end:
            await _on_api_end(body, raw)
        choices = raw.get("choices", [])
        if len(choices) != 1:
            raise ValueError("GInfer must return one assistant choice")
        message = choices[0]["message"]
        if message.get("role") != "assistant":
            raise ValueError("GInfer returned no assistant message")
        usage = raw.get("usage", {})
        if _on_usage:
            await _on_usage(usage)
        return {"output": from_chat_messages([message]), "usage": [usage] if usage else []}

    async def predict_click(self, model, image_b64, instruction, **generation):
        return None

    def get_capabilities(self):
        return ["step"]
