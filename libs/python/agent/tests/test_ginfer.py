import asyncio
import json
import unittest

import httpx
from cua_agent import ComputerAgent
from cua_agent.loops.ginfer import GInferConfig, from_chat_messages, to_chat_messages
from cua_agent.tools.base import BaseTool


class AsyncEcho(BaseTool):
    name = "echo"
    description = "Echo a value"
    parameters = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"]}

    async def call(self, params, **kwargs):
        await asyncio.sleep(0)
        return params["value"]


class GInferTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_transcript_round_trip(self):
        messages = [{"role": "user", "content": "echo"},
                    {"role": "assistant", "content": "checking", "tool_calls": [
                        {"id": "a", "type": "function", "function": {
                            "name": "echo", "arguments": '{"value":"ok"}'}},
                        {"id": "b", "type": "function", "function": {
                            "name": "echo", "arguments": '{"value":"also"}'}}]},
                    {"role": "tool", "tool_call_id": "a", "content": "ok"},
                    {"role": "tool", "tool_call_id": "b", "content": "also"}]
        self.assertEqual(to_chat_messages(from_chat_messages(messages)), messages)

    async def test_computer_agent_runs_custom_provider_and_awaits_tool(self):
        requests = []

        def serve(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertNotIn("reasoning", body)
            self.assertNotIn("truncation", body)
            self.assertNotIn("temperature", body)
            self.assertNotIn("max_tokens", body)
            if len(requests) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-1", "type": "function", "function": {
                        "name": "echo", "arguments": '{"value":"completed"}'}}]}
            else:
                self.assertEqual(body["messages"][-1], {
                    "role": "tool", "tool_call_id": "call-1", "content": "completed"})
                message = {"role": "assistant", "content": "Done"}
            return httpx.Response(200, json={"choices": [{"message": message}],
                                           "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            agent = ComputerAgent(model="assigned-model", tools=[AsyncEcho()],
                                  custom_loop=GInferConfig(client), telemetry_enabled=False)
            chunks = [chunk async for chunk in agent.run(
                [{"role": "user", "content": "echo"}], api_base="http://ginfer/v1")]
        self.assertEqual(len(requests), 2)
        self.assertEqual(chunks[-1]["output"][0]["content"][0]["text"], "Done")

    def test_text_only_provider_refuses_images(self):
        with self.assertRaises(ValueError):
            to_chat_messages([{"role": "user", "content": [{"type": "input_image"}]}])
