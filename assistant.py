from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from urllib import response
#from openai import AsyncOpenAI
#from openai import OpenAI
import time
from queue import Queue, Empty
from threading import Thread, Event, Lock


from dataclasses import dataclass
import os
import logging
from typing import Iterable, Optional, Any, Callable
import json
import requests

logger = logging.getLogger("assistant")

DEFAULT_ASSISTANT_MODEL = "gpt-5.1-mini"
DEFAULT_SYSTEM_PROMPT_BASE_TEMPLATE = """You are helping user with name '{your_name}'.
The user is in a live meeting and shares raw transcript snippets. Treat earlier messages as context but focus your reply on the most recent entry. Keep responses in plain text (no Markdown) and limit yourself to at most three sentences. Do not include document references, filenames, or URLs. Avoid repeating previous guidance."""
DEFAULT_MODE_DIRECTIVES: dict[str, str] = {
    "answer_question": "Address the most recent message directly. When you need additional context, perform a file search before responding. If the message is not a question or you lack sufficient information, reply with '---'. Do not ask the user follow-up questions.",
    "suggest_questions": "Produce one to three concise questions the user could ask next, each on its own line. Use available context and perform a file search first if it helps craft better prompts.",
    "explain_it": "Explain key concepts, decisions, or reasoning referenced in the latest message so the user understands them quickly. Draw on file search when needed. Provide clear, practical insight within three sentences.",
    "get_facts": "Investigate the latest message by performing an up-to-date internet search along with any relevant file search. Return a brief factual summary that highlights the most relevant information you find.",
    "custom_prompt": "The user has asked a specific question during the meeting. Address their question directly and comprehensively. Perform a file search or web search if additional information would help. Provide a clear, actionable answer.",
}
PROMPT_ENV_KEYS: dict[str, str] = {
    "base": "ASSISTANT_PROMPT_BASE",
    "answer_question": "ASSISTANT_PROMPT_ANSWER_QUESTION",
    "suggest_questions": "ASSISTANT_PROMPT_SUGGEST_QUESTIONS",
    "explain_it": "ASSISTANT_PROMPT_EXPLAIN_IT",
    "get_facts": "ASSISTANT_PROMPT_GET_FACTS",
    "custom_prompt": "ASSISTANT_PROMPT_CUSTOM_PROMPT",
}


@dataclass
class Message:
    user: str
    text: str


class Assistant:

    def __init__(self, vector_store_id: str, your_name: str, agent_name: str = "Agent", answer_queue: Queue = None, status_callback: Optional[Callable[[str, str], None]] = None):
        self.vector_store_id = vector_store_id
        self.your_name = your_name or "You"
        self.agent_name = agent_name
        self.answer_queue = answer_queue
        self.status_callback = status_callback
        self.messages_in = Queue()
        self.stop_event = Event()
        self.messages_lock = Lock()
        self.last_message_timestamp: Optional[float] = None
        # Up to 5 concurrent answer computations
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.message_thread = Thread(target=self._process_messages, args=())
        self.message_thread.start()
        self.last_answer = None
        self.messages = []
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key or not self.api_key.strip():
            raise ValueError("OPENAI_API_KEY is required to initialize Assistant")

        configured_model = os.getenv("OPENAI_MODEL_FOR_ASSISTANT")
        self.model = configured_model.strip() if configured_model and configured_model.strip() else DEFAULT_ASSISTANT_MODEL
        self.api_max_retries = int(os.getenv("ASSISTANT_API_MAX_RETRIES", "3"))
        self.api_retry_base_seconds = float(os.getenv("ASSISTANT_API_RETRY_BASE_SECONDS", "1.0"))
        self.answer_timeout_seconds = float(os.getenv("ASSISTANT_API_TIMEOUT_SECONDS", "60"))
        self.summary_timeout_seconds = float(os.getenv("ASSISTANT_SUMMARY_TIMEOUT_SECONDS", "120"))
        self.title_timeout_seconds = float(os.getenv("ASSISTANT_TITLE_TIMEOUT_SECONDS", "30"))
        self.custom_prompt_web_search_enabled = self._parse_bool_env(
            os.getenv("ASSISTANT_ENABLE_WEB_SEARCH_FOR_CUSTOM_PROMPTS"),
            default=True,
        )
        self.background_context: Optional[str] = None
        logger.info("Assistant model: %s", self.model)
        logger.info("Assistant custom-prompt web search enabled: %s", self.custom_prompt_web_search_enabled)

    @staticmethod
    def _parse_bool_env(raw: Optional[str], default: bool) -> bool:
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def set_background_context(self, context: Optional[str]) -> None:
        """Set background context that is included in every AI call."""
        self.background_context = context.strip() if context else None
        logger.info("Assistant background context %s (%s chars)",
                    "loaded" if self.background_context else "cleared",
                    len(self.background_context) if self.background_context else 0)

    def set_custom_prompt_web_search_enabled(self, enabled: bool) -> None:
        self.custom_prompt_web_search_enabled = bool(enabled)
        logger.info("Assistant custom-prompt web search set to: %s", self.custom_prompt_web_search_enabled)

    def set_model(self, model: Optional[str]) -> None:
        configured = (model or "").strip()
        self.model = configured if configured else DEFAULT_ASSISTANT_MODEL
        logger.info("Assistant model updated to: %s", self.model)

    def _emit_status(self, message: str, level: str = "info") -> None:
        if self.status_callback:
            try:
                self.status_callback(message, level)
            except Exception as e:
                logger.debug("Status callback failed: %s", e)

    @staticmethod
    def _decode_env_prompt(raw: Optional[str], default: str) -> str:
        if raw is None:
            return default
        value = raw.strip()
        if not value:
            return default
        return value.replace("\\r\\n", "\n").replace("\\n", "\n")

    def _get_prompt_template(self, env_key: str, default: str) -> str:
        return self._decode_env_prompt(os.getenv(env_key), default)

    def _compute_backoff(self, attempt: int) -> float:
        return self.api_retry_base_seconds * (2 ** max(0, attempt - 1))

    def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        operation_name: str,
    ) -> tuple[Optional[requests.Response], bool]:
        """Post with exponential backoff.

        Returns:
            Tuple[response_or_none, is_hard_failure]
        """
        max_attempts = max(1, self.api_max_retries + 1)

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            except (requests.Timeout, requests.ConnectionError) as e:
                should_retry = attempt < max_attempts
                if should_retry:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "%s transient network error on attempt %s/%s: %s. Retrying in %.2fs.",
                        operation_name,
                        attempt,
                        max_attempts,
                        e,
                        delay,
                    )
                    self._emit_status(f"{operation_name}: transient network issue, retrying...", "warning")
                    time.sleep(delay)
                    continue

                logger.error("%s failed after %s attempts due to network error: %s", operation_name, max_attempts, e)
                self._emit_status(f"{operation_name} failed: network error.", "error")
                return None, False
            except requests.RequestException as e:
                logger.error("%s request exception (non-retryable): %s", operation_name, e)
                self._emit_status(f"{operation_name} failed: request error.", "error")
                return None, False

            if 200 <= resp.status_code < 300:
                return resp, False

            if resp.status_code in (401, 403):
                try:
                    body_preview = json.dumps(resp.json(), indent=2)[:1500]
                except Exception:
                    body_preview = (resp.text or "")[:1500]
                logger.error("%s hard failure %s (auth/permission): %s", operation_name, resp.status_code, body_preview)
                self._emit_status(f"{operation_name} failed: authentication/permissions issue.", "error")
                return None, True

            transient_status = resp.status_code in (408, 409, 429) or resp.status_code >= 500
            if transient_status and attempt < max_attempts:
                delay = self._compute_backoff(attempt)
                logger.warning(
                    "%s transient API status %s on attempt %s/%s. Retrying in %.2fs.",
                    operation_name,
                    resp.status_code,
                    attempt,
                    max_attempts,
                    delay,
                )
                self._emit_status(f"{operation_name}: service busy ({resp.status_code}), retrying...", "warning")
                time.sleep(delay)
                continue

            try:
                body_preview = json.dumps(resp.json(), indent=2)[:1500]
            except Exception:
                body_preview = (resp.text or "")[:1500]

            if transient_status:
                logger.error("%s failed after retries. Last status=%s body=%s", operation_name, resp.status_code, body_preview)
                self._emit_status(f"{operation_name} failed after retries ({resp.status_code}).", "error")
                return None, False

            logger.error("%s hard failure status=%s body=%s", operation_name, resp.status_code, body_preview)
            self._emit_status(f"{operation_name} failed ({resp.status_code}).", "error")
            return None, True

        self._emit_status(f"{operation_name} failed: unknown error.", "error")
        return None, False

    def start_new_thread(self):
        """
        Start a new thread for the assistant.
        """
        logger.info("New conversation started")
        with self.messages_lock:
            self.messages = []
            self.last_answer = None
            self.last_message_timestamp = None
        # Keep the same queue instance to avoid races with the background reader thread.
        # Instead, drain any pending items from the current queue.
        self._drain_incoming_queue()

    def _append_message(self, role: str, message: str, timestamp: float) -> None:
        """Append one message to the in-memory conversation state."""
        with self.messages_lock:
            self.messages.append(Message(user=role, text=message))
            self.last_message_timestamp = timestamp

    def _drain_incoming_queue(self) -> int:
        """Drain queued inbound messages into conversation state.

        Returns:
            Number of drained items.
        """
        drained = 0
        while True:
            try:
                role, message, timestamp = self.messages_in.get_nowait()
            except Empty:
                break
            except Exception as e:
                logger.warning("Error draining Assistant queue: %s", e)
                break

            try:
                self._append_message(role, message, timestamp)
                drained += 1
            except Exception as e:
                logger.warning("Error storing drained message in Assistant thread: %s", e)
        return drained

    def add_message(self, timestamp: float, message: str, role: str = "user"):
        """
        Add a message to the thread.
        """
        self.messages_in.put((role, message, timestamp))
    
    def add_custom_prompt(self, timestamp: float, prompt: str, user_name: str):
        """
        Add a custom prompt from the user directly.
        """
        formatted_message = f"{user_name} asks: {prompt}"
        self.messages_in.put(("user", formatted_message, timestamp))

    def _process_messages(self):
        """
        Process inbound messages and store them for later analysis.

        NOTE: This assistant is intentionally manual-trigger only; no answers are
        generated from this background loop. Answers are created only when
        trigger_answer()/trigger_custom_prompt_answer() is called from UI actions.
        """
        while not self.stop_event.is_set():
            try:
                # Block for the first message
                first = self.messages_in.get(block=True, timeout=1)
            except Empty:
                continue

            # Collect a small batch (up to 5 total including the first), non-blocking
            batch = [first]
            for _ in range(4):
                try:
                    batch.append(self.messages_in.get_nowait())
                except Empty:
                    break
                except Exception as e:
                    logger.warning("Batch collection error: %s", e)
                    break

            # Append messages in order
            for role, message, timestamp in batch:
                try:
                    self._append_message(role, message, timestamp)
                except Exception as e:
                    logger.warning("Error storing message in Assistant thread: %s", e)
                    continue

    def stop(self):
        """Stop the message-processing thread and executor."""
        logger.info("Stopping assistant thread...")
        self.stop_event.set()
        self.message_thread.join()
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning("Executor shutdown error: %s", e)
        logger.info("Stopping assistant thread... DONE")


    def _extract_text_from_response(self, data: dict[str, Any]) -> Optional[str]:
        """Extract assistant text from Responses API payload.

        Prefers top-level output_text when present; otherwise, walks the
        output -> message -> content[] structure and collects text chunks
        from content items of type 'output_text' or 'text'.
        Returns a trimmed string or None when nothing is found.
        """
        # 1) Convenience field present on most responses
        top_text = data.get("output_text")
        if isinstance(top_text, str) and top_text.strip():
            return top_text.strip()

        # 2) Walk the structured output list
        parts: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "message":
                    # Ignore tool calls and non-message entries
                    continue
                content_list = item.get("content") or []
                if not isinstance(content_list, list):
                    continue
                for chunk in content_list:
                    if not isinstance(chunk, dict):
                        continue
                    ctype = chunk.get("type")
                    if ctype not in ("output_text", "text"):
                        continue
                    val = chunk.get("text")
                    if isinstance(val, dict):
                        val = val.get("value")
                    if isinstance(val, str) and val:
                        parts.append(val)

        text = "\n".join(parts).strip() if parts else None
        return text if text else None

    def _log_response_summary(self, data: dict[str, Any]) -> None:
        """Print a concise, informative summary of the Responses API result."""
        logger.debug("RAW response payload: %s", data)
        try:
            rid = data.get("id")
            status = data.get("status")
            model = data.get("model")
            created_at = data.get("created_at")
            error = data.get("error")
            incomplete = data.get("incomplete_details")
            max_out = data.get("max_output_tokens")
            usage = data.get("usage") or {}
            in_tok = (usage.get("input_tokens") or 0)
            out_tok = (usage.get("output_tokens") or 0)
            total_tok = (usage.get("total_tokens") or (in_tok + out_tok))
            out_tok_details = usage.get("output_tokens_details") or {}
            reasoning_tok = out_tok_details.get("reasoning_tokens")
            reasoning = data.get("reasoning") or {}
            effort = reasoning.get("effort")
            tool_choice_raw = data.get("tool_choice")
            if isinstance(tool_choice_raw, dict):
                tool_choice = tool_choice_raw.get("type") or tool_choice_raw.get("name")
            else:
                tool_choice = tool_choice_raw
            parallel = data.get("parallel_tool_calls")
            temperature = data.get("temperature")
            service_tier = data.get("service_tier")

            fs_summaries: list[str] = []
            fs_used = False
            output = data.get("output")
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "file_search_call":
                        fs_used = True
                        q = item.get("queries") or []
                        q_preview = "; ".join(q[:3]) if isinstance(q, list) else str(q)
                        q_more = f" (+{len(q)-3} more)" if isinstance(q, list) and len(q) > 3 else ""
                        st = item.get("status")
                        res = item.get("results")
                        res_count = (len(res) if isinstance(res, list) else (0 if res is None else 1))
                        fs_summaries.append(f"status={st}; queries={len(q)}: {q_preview}{q_more}; results={res_count}")

            output_types: list[str] = []
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        t = item.get("type")
                        if isinstance(t, str):
                            output_types.append(t)

            tools_cfg = data.get("tools")
            tool_kinds: list[str] = []
            if isinstance(tools_cfg, list):
                for t in tools_cfg:
                    if isinstance(t, dict) and isinstance(t.get("type"), str):
                        tool_kinds.append(t.get("type"))

            truncated_by_tokens = False
            if incomplete:
                reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
                truncated_by_tokens = (reason == "max_tokens")
            if not truncated_by_tokens and isinstance(max_out, int) and max_out > 0:
                truncated_by_tokens = (out_tok >= max_out)

            text = self._extract_text_from_response(data) or ""
            is_placeholder = (text.strip() == "---" or not text.strip())

            lines = []
            lines.append("Response summary:")
            lines.append(f"  id={rid} model={model} status={status} created_at={created_at}")
            lines.append(f"  error={error}")
            lines.append(f"  incomplete_details={incomplete}")
            if reasoning_tok is not None:
                lines.append(f"  tokens: input={in_tok} output={out_tok} (reasoning={reasoning_tok}) total={total_tok} max_output={max_out}")
            else:
                lines.append(f"  tokens: input={in_tok} output={out_tok} total={total_tok} max_output={max_out}")
            lines.append(f"  truncated_by_tokens={truncated_by_tokens}")
            lines.append(f"  reasoning.effort={effort}")
            lines.append(f"  tool_choice={tool_choice} parallel_tool_calls={parallel}")
            if tool_kinds:
                lines.append(f"  tools_configured={', '.join(tool_kinds)}")
            if output_types:
                lines.append(f"  output_item_types={', '.join(output_types)}")
            lines.append(f"  temperature={temperature} service_tier={service_tier}")
            if fs_used:
                lines.append("  file_search_calls:")
                for i, s in enumerate(fs_summaries, 1):
                    lines.append(f"    {i}. {s}")
            else:
                lines.append("  file_search_calls: none")
            lines.append("  message:")
            lines.append(f"    {text if text else ''}")
            lines.append(f"  placeholder_answer={is_placeholder}")

            logger.info("%s", "\n".join(lines))
        except Exception as e:
            logger.warning("Failed to print response summary: %s", e)


    def _build_system_prompt(self, mode: str) -> str:
        mode = (mode or "answer_question").strip().lower()
        base_template = self._get_prompt_template(PROMPT_ENV_KEYS["base"], DEFAULT_SYSTEM_PROMPT_BASE_TEMPLATE)
        base = base_template.replace("{your_name}", self.your_name)
        mode_directives = {
            current_mode: self._get_prompt_template(PROMPT_ENV_KEYS[current_mode], default_text)
            for current_mode, default_text in DEFAULT_MODE_DIRECTIVES.items()
        }

        directive = mode_directives.get(mode)
        if directive is None:
            directive = mode_directives["answer_question"]

        prompt = f"{base}\n\nMode directive: {directive}"

        if self.background_context:
            prompt += f"\n\n**Background Context:**\nUse this information to better understand meeting participants, terminology, and project context:\n\n{self.background_context}"

        return prompt


    def _answer(
        self,
        timestamp: float,
        messages_snapshot: Optional[list[Message]] = None,
        mode: str = "answer_question",
        result_callback: Optional[Callable[[Optional[str]], None]] = None,
        append_response_to_messages: bool = True,
        enqueue_response: bool = True,
    ):
        logger.info("Answering question... (mode=%s)", mode)
        time_val = timestamp
        msgs = messages_snapshot if messages_snapshot is not None else self.messages

        def _notify_result(result_text: Optional[str]) -> None:
            if result_callback:
                try:
                    result_callback(result_text)
                except Exception as e:
                    logger.debug("Assistant result callback failed: %s", e)

        if not msgs:
            logger.info("No messages available for assistant to process.")
            _notify_result(None)
            return

        system = self._build_system_prompt(mode)

        content: list[dict[str, Any]] = []
        previous_messages = ""
        for m in msgs[:-1]:
            previous_messages += f"{m.user}: {m.text}\n"

        content.append({"type": "input_text", "text": f"Meeting transcript so far:\n```\n{previous_messages}\n```\n"})
        content.append({"type": "input_text", "text": f"Latest message from {msgs[-1].user}: {msgs[-1].text}"})

        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        user_message: dict[str, Any] = {
            "role": "user",
            "content": content,
        }

        payload: dict[str, Any] = {
            "model": self.model or DEFAULT_ASSISTANT_MODEL,
            "instructions": system,
            "input": [user_message],
            "max_output_tokens": 2000,
        }

        tools: list[dict[str, Any]] = []
        if self.vector_store_id:
            tools.append({"type": "file_search", "vector_store_ids": [self.vector_store_id]})

        if mode == "get_facts":
            tools.append({"type": "web_search"})
        
        # For custom prompts, optionally enable web search to provide comprehensive answers
        if mode == "custom_prompt" and self.custom_prompt_web_search_enabled:
            tools.append({"type": "web_search"})

        if tools:
            payload["tools"] = tools

        if payload["model"].find('gpt-5') >= 0:
            payload["reasoning"] = {"effort": "low"}
        else:
            if mode == "get_facts" or mode == "custom_prompt":
                payload["tool_choice"] = "auto"
            elif self.vector_store_id:
                payload["tool_choice"] = {"type": "file_search"}

        self._emit_status("Assistant request in progress...", "info")
        resp, is_hard_failure = self._post_with_retry(
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=self.answer_timeout_seconds,
            operation_name="Assistant reply",
        )
        if resp is None:
            if is_hard_failure:
                self._emit_status("Assistant disabled by API auth/permission error.", "error")
            _notify_result(None)
            return ":eyes:"

        try:
            data = resp.json()
        except Exception:
            _notify_result(None)
            return ":eyes:"

        self._log_response_summary(data)

        text = self._extract_text_from_response(data)
        if not text or text.strip() == "---":
            self.last_answer = None
            self._emit_status("Assistant: no actionable response.", "info")
            _notify_result(None)
        else:
            self.last_answer = text
            if append_response_to_messages:
                with self.messages_lock:
                    self.messages.append(Message(user="assistant", text=text))
                    self.last_message_timestamp = timestamp
            if enqueue_response and self.answer_queue:
                logger.info("Answered at %s", time_val)
                self.answer_queue.put((self.agent_name, self.last_answer, time_val))
            _notify_result(text)
            self._emit_status("Assistant response ready.", "info")

        logger.info("Answering question... DONE")

    def trigger_answer(
        self,
        mode: str = "answer_question",
        result_callback: Optional[Callable[[Optional[str]], None]] = None,
        append_response_to_messages: bool = True,
        enqueue_response: bool = True,
    ) -> bool:
        """Manually request the assistant to craft a reply based on collected messages."""
        if self.stop_event.is_set():
            logger.warning("Assistant is stopped; cannot trigger answer.")
            return False

        # Ensure very recent messages queued from other threads are included when
        # the user presses the button.
        drained = self._drain_incoming_queue()
        if drained:
            logger.debug("Drained %s pending message(s) before trigger.", drained)

        normalized_mode = (mode or "answer_question").strip().lower()

        with self.messages_lock:
            if not self.messages:
                logger.info("Assistant has no messages to analyze yet.")
                return False
            snapshot = list(self.messages)

        timestamp = time.time()

        try:
            self.executor.submit(
                self._answer,
                timestamp,
                snapshot,
                normalized_mode,
                result_callback,
                append_response_to_messages,
                enqueue_response,
            )
            return True
        except Exception as e:
            logger.error("Failed to trigger assistant answer: %s", e)
            return False
    
    def trigger_custom_prompt_answer(self) -> bool:
        """Generate a response to a custom prompt."""
        return self.trigger_answer(mode="custom_prompt")
    
    def generate_meeting_summary(self, transcript: str, meeting_title: Optional[str] = None, context: Optional[str] = None) -> Optional[dict[str, str]]:
        """Generate a detailed meeting summary in Markdown format with a title.
        
        Args:
            transcript: Full meeting transcript text
            meeting_title: Optional meeting title from MS Teams window
            context: Optional background context about people, terminology, and projects
            
        Returns:
            Dictionary with 'title' and 'summary' keys, or None if generation fails
        """
        logger.info("Generating meeting summary...")
        if meeting_title:
            logger.info("Using meeting title from Teams: %s", meeting_title)
        if context:
            logger.info("Using context information: %s characters", len(context))
        
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Build system prompt with context if available
        system_prompt = """You are an expert meeting analyst. Generate a comprehensive meeting summary in Markdown format.

Your summary should include:
1. **Key Discussion Points** - Main topics discussed
2. **Decisions Made** - Any decisions or agreements reached
3. **Action Items** - Tasks assigned with owners (if mentioned)
4. **Next Steps** - Follow-up actions and timelines
5. **Important Dates/Deadlines** - Any mentioned dates or deadlines

Format the output in well-structured Markdown with headers, bullet points, and emphasis where appropriate.
Be concise but comprehensive. Focus on actionable information."""

        if context:
            system_prompt += f"\n\n**Background Context:**\nUse this information to better understand the meeting participants, terminology, and project context:\n\n{context}"

        # Use meeting title from Teams if available, otherwise generate one
        if meeting_title and meeting_title.strip():
            title_prompt = f"""The meeting title is: "{meeting_title}"

Based on this title and the meeting transcript, generate a concise 3-4 word summary title that captures the main outcome or focus.
The title should be professional and descriptive. Only return the title text, nothing else."""
        else:
            title_prompt = """Based on this meeting transcript, generate a concise 3-4 word title that captures the main topic.
The title should be professional and descriptive. Only return the title text, nothing else."""
        
        # First, generate the summary
        meeting_context = f"Meeting: {meeting_title}\n\n" if meeting_title else ""
        input_text = f"{meeting_context}Meeting transcript:\n\n{transcript}"
        
        # Log what we're sending
        logger.info("Generating summary with input length: %s characters", len(input_text))
        if meeting_title:
            logger.info("Including meeting title in context: '%s'", meeting_title)
        else:
            logger.info("No meeting title available from Teams window")
        
        summary_payload = {
            "model": self.model or DEFAULT_ASSISTANT_MODEL,
            "instructions": system_prompt,
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": input_text}]
            }],
            "max_output_tokens": 4000,
        }
        
        # Add file search if available
        tools = []
        if self.vector_store_id:
            tools.append({"type": "file_search", "vector_store_ids": [self.vector_store_id]})
        
        if tools:
            summary_payload["tools"] = tools
        
        summary = None
        try:
            self._emit_status("Summary generation in progress...", "info")
            resp, _ = self._post_with_retry(
                url=url,
                headers=headers,
                payload=summary_payload,
                timeout_seconds=self.summary_timeout_seconds,
                operation_name="Summary generation",
            )
            if resp is None:
                return None
            
            data = resp.json()
            self._log_response_summary(data)
            
            summary = self._extract_text_from_response(data)
            if not summary or not summary.strip():
                logger.warning("No summary text extracted from response")
                self._emit_status("Summary generation produced no text.", "warning")
                return None
            
            logger.info("Meeting summary generated successfully")
            self._emit_status("Summary generated successfully.", "info")
                
        except Exception as e:
            logger.exception("Error generating summary: %s", e)
            return None
        
        # Use meeting title directly if available, otherwise generate with AI
        if meeting_title and meeting_title.strip():
            title = meeting_title.strip()
            logger.info("Using meeting title from Teams: %s", title)
        else:
            # Generate a concise title based on the summary using a small/fast model
            logger.info("No meeting title detected, generating title from summary...")
            title_context_parts = []
            if context:
                title_context_parts.append(f"Background context: {context[:500]}")  # Limit context for title
            
            title_context = "\n\n".join(title_context_parts) + "\n\n" if title_context_parts else ""
            
            title_generation_prompt = """Generate a concise 3-4 word title for this meeting summary.
The title should be professional, descriptive, and capture the main topic or outcome.
Use the background context to understand terminology and proper names.
Only return the title text, nothing else. Do not use quotes."""
            
            title_payload = {
                "model": "gpt-4o-mini",  # Use small, fast model for title generation
                "instructions": title_generation_prompt,
                "input": [{
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"{title_context}Summary:\n\n{summary[:1000]}"}]
                }],
                "max_output_tokens": 20,
            }
            
            title = "Meeting Summary"  # Default fallback
            try:
                resp, _ = self._post_with_retry(
                    url=url,
                    headers=headers,
                    payload=title_payload,
                    timeout_seconds=self.title_timeout_seconds,
                    operation_name="Summary title generation",
                )
                if resp is not None and 200 <= resp.status_code < 300:
                    data = resp.json()
                    extracted_title = self._extract_text_from_response(data)
                    if extracted_title and extracted_title.strip():
                        title = extracted_title.strip()
                        # Clean up quotes if AI added them
                        title = title.strip('"').strip("'").strip()
                        logger.info("Generated title from summary: %s", title)
            except Exception as e:
                logger.warning("Error generating title: %s", e)
        
        return {
            "title": title,
            "summary": summary.strip()
        }
