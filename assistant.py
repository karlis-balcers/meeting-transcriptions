from concurrent.futures import thread
from urllib import response
from openai import AsyncOpenAI
from openai import OpenAI
import time
from queue import Queue, Empty
from threading import Thread, Event, Lock

class Assistant:
    
    def __init__(self, assistant_id: str, your_name: str, agent_name: str = "Agent", answer_queue: Queue = None):
        self.assistant_id = assistant_id
        self.your_name = your_name
        self.agent_name = agent_name
        self.client = OpenAI()
        self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
        self.answer_queue = answer_queue
        self.thread = self.client.beta.threads.create()
        self.messages_in = Queue()
        self.stop_event = Event()
        self.message_thread = Thread(target=self._store_message, args=())
        self.message_thread.start()
        self.last_answer = None

    def start_new_thread(self):
        """
        Start a new thread for the assistant.
        """
        self.thread = self.client.beta.threads.create()
        print(f"New thread started: {self.thread.id}")
        self.last_answer = None
        self.messages_in = Queue()

    def add_message(self, message: str, role: str = "user"):
        """
        Add a message to the thread.
        """
        self.messages_in.put((role, message))

    def _store_message(self):
        """
        Store a message in the queue for later processing.
        """
        while not self.stop_event.is_set():
            try:
                role, message = self.messages_in.get(block=True, timeout=1)
                try:
                    _ = self.client.beta.threads.messages.create(
                            thread_id=self.thread.id,
                            role=role,
                            content=message
                        )
                except Exception as e:
                    print(f"Error storing message in Assistant thread: {e}")

                if role != "assistant": #and message.find("?")>0:
                    self._answer()
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

    def _answer(self):
        print("Answering question...")
        time_val = time.time()

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant_id,
            instructions=f"""{self.assistant.instructions}
You are helping user with name '{self.your_name}'. Your task is to help the user by suggesting what to say next. Never summarize the transcript.
The user is sharing with you the raw meeting transcript.
The user is never addressing you - if the user says "hi"or "hello" or asks a question he is not talking to you but to the participants of the meeting.
Make your responses short and to the point - no more than 3 sentences.
Do not return references to the documents. Skip intros and outros. 
Do not use Markdown, but plain-text.
If there is not enough transcript for a reasonable answer then reply with '---'. 
Do not repeat yourself - do not suggest same as before, do not provide generic answers.
If there is nothing new to say then respond with '---'. 
            """,
        )

        while not run.completed_at:
            time.sleep(0.5)
            run = self.client.beta.threads.runs.retrieve(run.id)
        
        if run.status == 'completed': 
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            # Response contains all messages in the thread. Therefore we always take the last message which is the answer.
            text = messages.data[0].content[0].text.value.strip()
            print(f"Assistant Message: {text}")
            if text =="---":
                self.last_answer = None
            else:
                self.last_answer = text
                if self.answer_queue:
                    print(f"Answered at {time_val}: {self.last_answer}")
                    self.answer_queue.put((self.agent_name, self.last_answer, time_val))
                    self.add_message(self.last_answer, role="assistant")
        else:
            print(f"Run failed: {run}")
            self.last_answer = None
        print("Answering question... DONE")
        

 