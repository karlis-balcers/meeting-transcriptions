from __future__ import annotations

from concurrent.futures import thread
from urllib import response
#from openai import AsyncOpenAI
#from openai import OpenAI
import time
from queue import Queue, Empty
from threading import Thread, Event, Lock


from dataclasses import dataclass
import os
from typing import Iterable, Optional, Any
import json
import requests


@dataclass
class Message:
    user: str
    text: str


class Assistant:

    def __init__(self, vector_store_id: str, your_name: str, agent_name: str = "Agent", answer_queue: Queue = None):
        self.vector_store_id = vector_store_id
        self.your_name = your_name
        self.agent_name = agent_name
        self.answer_queue = answer_queue
        self.messages_in = Queue()
        self.stop_event = Event()
        self.message_thread = Thread(target=self._store_message, args=())
        self.message_thread.start()
        self.last_answer = None
        self.messages = []
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-5-mini"

    def start_new_thread(self):
        """
        Start a new thread for the assistant.
        """
        print(f"New thread started: {self.thread.id}")
        self.messages = []
        self.last_answer = None
        self.messages_in = Queue()

    def add_message(self, timestamp: float, message: str, role: str = "user"):
        """
        Add a message to the thread.
        """
        self.messages_in.put((role, message, timestamp))

    def _store_message(self):
        """
        Store a message in the queue for later processing.
        """
        while not self.stop_event.is_set():
            try:
                role, message, timestamp = self.messages_in.get(block=True, timeout=1)
                try:
                    self.messages.append(Message(user=role, text=message))
                except Exception as e:
                    print(f"Error storing message in Assistant thread: {e}")

                if role != "assistant": #and message.find("?")>0:
                    self._answer(timestamp)
            except Empty:
                continue
           

    def stop(self):
        """
        Stop the message thread.
        """
        print("Stopping assistant thread...")
        self.stop_event.set()
        self.message_thread.join()
        print("Stopping assistant thread... DONE")


    def _answer(self, timestamp: float):
        print("Answering question...")
        time_val = timestamp

        system = f"""You are helping user with name '{self.your_name}'. Your task is to help the user by suggesting what to say next. Never summarize the transcript.
The user is sharing with you the raw meeting transcript.
The user is never addressing you - if the user says "hi"or "hello" or asks a question he is not talking to you but to the participants of the meeting.
Make your responses short and to the point - no more than 3 sentences.
Do not return references to the documents. Skip intros and outros. 
Do not use Markdown, but plain-text.
If there is not enough transcript for a reasonable answer then reply with '---'. 
Do not repeat yourself - do not suggest same as before, do not provide generic answers.
If there is nothing new to say then respond with '---'. 
            """

        # Build structured content for the Responses API
        content: list[dict[str, Any]] = []
        for m in self.messages:
            content.append({"type": "input_text", "text": f"{m.user}: {m.text}"})
        last_message = f"Latest from {self.messages[-1].user}: {self.messages[-1].text}"
        content.append({"type": "input_text", "text": last_message})

        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_message: dict[str, Any] = {
            "role": "user",
            "content": content,
        }

        # Attach vector store for file_search if configured
        payload: dict[str, Any] = {
            "model": self.model or "gpt-5-mini",
            "instructions": system,
            "input": [user_message],
            "max_output_tokens": 1000,
        }

        if self.vector_store_id:
            payload["model"] = "gpt-4o-mini" # `file_search` is only supported by previous models and not by GPT-5.
            # For /v1/responses, pass the store on the tool itself (NOT top-level tool_resources)
            payload["tools"] = [
                {"type": "file_search", "vector_store_ids": [self.vector_store_id]}
            ]
            payload["tool_choice"] = {"type": "file_search"}  # enforce tool usage

        if payload["model"].find('5')>=0:
            payload["reasoning"] =  {"effort": "low"} #only gpt-5 supports reasoning effort config.

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException:
            return ":eyes:"
        if not (200 <= resp.status_code < 300):
            # Print brief error context for troubleshooting
            try:
                err = resp.json()
                print("Responses API error:", json.dumps(err, indent=2)[:2000])
            except Exception:
                print("Responses API error (raw):", (resp.text or "")[:2000])
            return ":eyes:"

        print("RAW:",resp)
        try:
            data = resp.json()
        except Exception:
            return ":eyes:"

        # Extract friendly text output; fall back to raw JSON text if schema differs
        text = data.get("output_text")
        if not text:
            out_parts: list[str] = []
            for item in data.get("output", []) or []:
                for c in item.get("content", []) or []:
                    if "text" in c:
                        t = c["text"]["value"] if isinstance(c["text"], dict) else c["text"]
                        if isinstance(t, str):
                            out_parts.append(t)
            text = "\n".join(out_parts) or json.dumps(data, indent=2)
        if text =="---":
            self.last_answer = None
        else:
            self.last_answer = text
            if self.answer_queue:
                print(f"Answered at {time_val}: {self.last_answer}")
                self.answer_queue.put((self.agent_name, self.last_answer, time_val))
                self.add_message(time_val, self.last_answer, role="assistant")

        print("Answering question... DONE")
