from __future__ import annotations

from concurrent.futures import thread
from urllib import response
from openai import OpenAI, APIError
import time
from queue import Queue, Empty
from threading import Thread, Event, Lock


from dataclasses import dataclass
import os
from typing import Iterable, Optional, Any
import json


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
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL_FOR_ASSISTANT", "gpt-4o-mini")

    def reset_conversation_state(self):
        """
        Reset the assistant's conversation state.
        """
        print("Resetting assistant state.")
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

        system_prompt = f"""You are helping user with name '{self.your_name}'. Your task is to help the user by suggesting what to say next. Never summarize the transcript.
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

        user_message: dict[str, Any] = {
            "role": "user",
            "content": content,
        }

        try:
            request_params = {
                "model": self.model,
                "instructions": system_prompt,
                "input": [user_message],
                "max_output_tokens": 1000,
            }

            if self.vector_store_id:
                request_params["model"] = "gpt-4o-mini" # `file_search` is only supported by previous models and not by GPT-5.
                request_params["tools"] = [
                    {"type": "file_search", "vector_store_ids": [self.vector_store_id]}
                ]
                request_params["tool_choice"] = {"type": "file_search"}  # enforce tool usage

            if "5" in request_params["model"]:
                request_params["reasoning"] = {"effort": "low"} #only gpt-5 supports reasoning effort config.

            response = self.client.responses.create(**request_params)

            text = response.output_text
            if not text:
                out_parts: list[str] = []
                for item in response.output or []:
                    for c in item.content or []:
                        if hasattr(c, "text"):
                            t = c.text.value if hasattr(c.text, "value") else c.text
                            if isinstance(t, str):
                                out_parts.append(t)
                text = "\n".join(out_parts)

            if text == "---":
                self.last_answer = None
            else:
                self.last_answer = text
                if self.answer_queue:
                    print(f"Answered at {time_val}: {self.last_answer}")
                    self.answer_queue.put((self.agent_name, self.last_answer, time_val))
                    self.add_message(time_val, self.last_answer, role="assistant")

        except APIError as e:
            print(f"OpenAI API Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        print("Answering question... DONE")
