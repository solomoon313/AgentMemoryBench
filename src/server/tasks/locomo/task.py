import json
import logging
import math
import re
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from collections import Counter

# Python 3.8 compatibility: functools.cache is available from Python 3.9+
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)

from openai import OpenAI

from .task_base import Task, Session
from .typings import (
    SampleIndex,
    SampleStatus,
    TaskSampleExecutionResult,
    RewardHistoryItem
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from memory.base import parse_llm_json_response
from src.runner.agent import _extract_api_key, _message_to_dict, _normalize_base_url

SYSTEM_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   use the memory timestamp to understand the reference carefully
6. Preserve the time form and granularity supported by the memories. Do not convert
   relative time references into absolute dates unless the evidence clearly supports
   that exact answer form.
7. Focus only on the content of the memories. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. Do not invent facts or add unsupported details.
9. If the memories do not support the answer, reply with "I don't know".
10. The answer should be less than 5-6 words when possible.
11. Think briefly about the relevant evidence before answering, then give only the final answer."""

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def convert_session_to_history(session_dialogues: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert a session dialogue list to a memory-format history.

    Args:
        session_dialogues: list of dicts with format [{"speaker": "...", "dia_id": "...", "text": "..."}, ...]

    Returns:
        history: list of dicts with format:
        [
            {"role": "assistant", "content": "--- Session N started at ... ---"},
            {"role": "user", "content": "Speaker: text\n  [caption] ...\n  [query] ..."},
            ...
        ]
    """
    history = []
    if not session_dialogues:
        return history

    first_dialogue = session_dialogues[0]
    session_index = first_dialogue.get("session_index")
    session_date_time = str(first_dialogue.get("session_date_time", "") or "")
    if session_index is not None:
        history.append({
            "role": "assistant",
            "content": f"--- Session {session_index} started at {session_date_time} ---",
        })

    for dialogue in session_dialogues:
        speaker = str(dialogue.get("speaker", "") or "")
        text = str(dialogue.get("text", "") or "")
        lines = [f"{speaker}: {text}"]

        images = dialogue.get("img_url") or []
        if isinstance(images, str):
            images = [images]
        blip_caption = str(dialogue.get("blip_caption", "") or "").strip()
        query = str(dialogue.get("query", "") or "").strip()

        if images:
            lines.append(f"  [images] {', '.join(str(item) for item in images)}")
        if blip_caption:
            lines.append(f"  [caption] {blip_caption}")
        if query:
            lines.append(f"  [query] {query}")

        history.append({
            "role": "user",
            "content": "\n".join(lines)
        })
    return history


def _build_session_header(session_id: int, session_date_time: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"--- Session {session_id} started at {session_date_time} ---",
    }


def _normalize_answer_candidates(raw_candidates: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(raw_candidates, list):
        candidates.extend(str(candidate).strip() for candidate in raw_candidates if str(candidate).strip())
    elif raw_candidates not in (None, ""):
        candidate = str(raw_candidates).strip()
        if candidate:
            candidates.append(candidate)
    if not candidates:
        candidates.append("")
    return candidates


def _extract_braced_content(text: str, opening_brace_index: int) -> Optional[str]:
    if opening_brace_index < 0 or opening_brace_index >= len(text) or text[opening_brace_index] != "{":
        return None

    depth = 0
    content_start = opening_brace_index + 1
    for index in range(opening_brace_index, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
            continue
        if ch != "}":
            continue
        depth -= 1
        if depth == 0:
            return text[content_start:index]
    return text[content_start:]


def _extract_last_boxed_value(text: str) -> Optional[str]:
    last_value: Optional[str] = None
    for match in re.finditer(r"\\box(?:ed)?\{", text):
        opening_brace_index = match.end() - 1
        extracted = _extract_braced_content(text, opening_brace_index)
        if extracted is not None:
            last_value = extracted.strip()
    return last_value


def _normalize_prediction_text(text: str) -> str:
    cleaned = str(text or "").strip()
    boxed_value = _extract_last_boxed_value(cleaned)
    if boxed_value is not None:
        return boxed_value
    lower = cleaned.lower()
    if "final answer:" in lower:
        idx = lower.index("final answer:")
        cleaned = cleaned[idx + len("final answer:"):].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Simple recursive dict merge: override takes precedence over base."""
    result = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def load_evaluate_agent_config(agent_name: str) -> Optional[Dict[str, Any]]:
    """
    Load and merge evaluate_api.yaml + evaluate_agent.yaml for the specified agent. Returns:
    {
        "url": ...,
        "headers": {...},
        "body": {...},
    }
    """
    # Navigate 4 levels up from src/server/tasks/locomo/task.py to the project root:
    # parents[0] = src/server/tasks/locomo/
    # parents[1] = src/server/tasks/
    # parents[2] = src/server/
    # parents[3] = src/
    # parents[4] = project root
    ROOT_DIR = Path(__file__).resolve().parents[4]
    LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"

    agent_cfg_path = LLMAPI_DIR / "evaluate_agent.yaml"

    if not agent_cfg_path.exists():
        return None

    try:
        with agent_cfg_path.open("r", encoding="utf-8") as f:
            agents_cfg = yaml.safe_load(f) or {}

        logger = logging.getLogger(__name__)
        logger.info(f"Loading LLM judge config for agent: '{agent_name}'")
        logger.info(f"Available agents in evaluate_agent.yaml: {list(agents_cfg.keys())}")

        if agent_name not in agents_cfg:
            logger.warning(f"Agent '{agent_name}' not found in evaluate_agent.yaml. Available agents: {list(agents_cfg.keys())}")
            return None

        agent_cfg = agents_cfg[agent_name] or {}
        logger.info(f"Found agent config: {agent_cfg}")

        # Handle the import field: if present, load the referenced file; otherwise fall back to evaluate_api.yaml
        import_path = agent_cfg.get("import", "./evaluate_api.yaml")
        logger.info(f"Import path from agent config: {import_path}")

        if import_path.startswith("./"):
            api_cfg_path = LLMAPI_DIR / import_path[2:]  # strip "./"
        else:
            api_cfg_path = LLMAPI_DIR / import_path

        logger.info(f"Trying to load API config from: {api_cfg_path}")

        if not api_cfg_path.exists():
            # If the imported file does not exist, fall back to evaluate_api.yaml
            logger.warning(f"Import file {api_cfg_path} does not exist, trying evaluate_api.yaml")
            api_cfg_path = LLMAPI_DIR / "evaluate_api.yaml"
            if not api_cfg_path.exists():
                logger.error(f"evaluate_api.yaml also does not exist at {api_cfg_path}")
                return None

        # Load base config (from the imported file or evaluate_api.yaml)
        with api_cfg_path.open("r", encoding="utf-8") as f:
            api_cfg = yaml.safe_load(f) or {}

        base_params = api_cfg.get("parameters", {}) or {}
        agent_params = agent_cfg.get("parameters", {}) or {}

        # Deep-merge parameters (agent overrides api)
        merged_params = _deep_merge_dict(base_params, agent_params)

        url = merged_params.get("url") or api_cfg.get("parameters", {}).get("url")
        if not url:
            logger.error(f"No URL found in merged config. merged_params keys: {list(merged_params.keys())}, api_cfg parameters keys: {list(api_cfg.get('parameters', {}).keys())}")
            return None

        headers = merged_params.get("headers", {}) or api_cfg.get("parameters", {}).get("headers", {})
        body = merged_params.get("body", {}) or api_cfg.get("parameters", {}).get("body", {})
        api_key = _extract_api_key(headers)
        base_url = _normalize_base_url(url)

        logger.info(f"Successfully loaded LLM judge config: url={url}, model={body.get('model', 'N/A')}")

        return {
            "url": url,
            "base_url": base_url,
            "api_key": api_key,
            "body": body,
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load evaluate agent config: {e}", exc_info=True)
        return None


class LocomoBaseTask(Task):
    """Base class for Locomo tasks."""

    def __init__(
        self,
        data_file: str,
        llm_judge_agent: str = "Qwen/Qwen3-14B",  # agent name from evaluate_agent.yaml
        tokenizer_path: Optional[str] = None,   # legacy arg, kept for config compatibility
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file
        self.llm_judge_agent = llm_judge_agent

        if tokenizer_path:
            self.logger.info("tokenizer_path is ignored; LoCoMo scoring now uses internal tokenization.")

        # Load LLM judge config from configs/llmapi/evaluate_api.yaml and evaluate_agent.yaml
        self.llm_judge_config = None
        self.llm_judge_client = None
        try:
            self.logger.info(f"Attempting to load LLM judge config for agent: '{llm_judge_agent}'")
            self.llm_judge_config = load_evaluate_agent_config(llm_judge_agent)
            if self.llm_judge_config:
                self.llm_judge_client = OpenAI(
                    base_url=self.llm_judge_config["base_url"],
                    api_key=self.llm_judge_config["api_key"],
                )
                self.logger.info(f"Successfully loaded LLM judge config for agent: {llm_judge_agent}")
                self.logger.info(f"  -> URL: {self.llm_judge_config.get('url', 'N/A')}")
                self.logger.info(f"  -> Model: {self.llm_judge_config.get('body', {}).get('model', 'N/A')}")
            else:
                self.logger.error(f"LLM judge agent '{llm_judge_agent}' not found or config invalid. LLM judge will return 0.")
        except Exception as e:
            self.logger.error(f"Failed to load LLM judge config: {e}. LLM judge will return 0.", exc_info=True)
            self.llm_judge_config = None
            self.llm_judge_client = None

        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data or len(data) == 0:
            raise ValueError(f"Empty data file: {data_file}")

        item = data[0]
        self.conversation = item.get("conversation", {})
        self.qa_list = item.get("qa", [])

        # Index QA items by session (where field -> [qa_index, ...])
        self.qa_by_session: Dict[int, List[int]] = defaultdict(list)
        for idx, qa in enumerate(self.qa_list):
            where = qa.get("where")
            if where is not None:
                self.qa_by_session[where].append(idx)

        # Get all session IDs (sorted numerically)
        self.session_ids = sorted([int(k.replace("session_", ""))
                                   for k in self.conversation.keys()
                                   if k.startswith("session_") and not k.endswith("_date_time")])

        self.logger.info(f"Loaded {len(self.qa_list)} QA pairs, {len(self.session_ids)} sessions")
        self.logger.info(f"Session IDs: {self.session_ids}")
        self.logger.info(f"QA distribution by session: {dict(self.qa_by_session)}")

    def get_session_history(self, session_id: int) -> List[Dict[str, Any]]:
        """Return the dialogue history for the given session in memory format."""
        session_key = f"session_{session_id}"
        session_dialogues = self.conversation.get(session_key, [])
        if not session_dialogues:
            self.logger.warning(f"Session {session_id} not found")
            return []
        history = convert_session_to_history(session_dialogues)
        session_date_time = str(self.conversation.get(f"{session_key}_date_time", "") or "").strip()
        if session_date_time:
            header = _build_session_header(session_id, session_date_time)
            first_content = str(history[0].get("content", "") or "").strip() if history else ""
            if first_content != header["content"]:
                history = [header] + history
        return history

    def get_qa_indices_for_session(self, session_id: int) -> List[int]:
        """Return all QA indices for the given session, excluding category=5 items."""
        indices = self.qa_by_session.get(session_id, [])
        # Filter out category=5 items (following mem0's approach)
        filtered_indices = []
        for idx in indices:
            if idx < len(self.qa_list):
                qa_item = self.qa_list[idx]
                category = qa_item.get("category", None)
                if category is not None and int(category) == 5:
                    continue  # skip category=5 items
                filtered_indices.append(idx)
        return filtered_indices

    def _build_memories_text_for_qa(self, qa_item: Dict[str, Any], training_mode: str) -> str:
        """
        Build the exact memories text that should be exposed to the model for this QA.

        - offline: all sessions
        - other modes: sessions 1..max(where_ground_truth), or fallback to `where`
        """
        where_ground_truth = qa_item.get("where_ground_truth", []) or []
        session_ids: List[int] = []

        if training_mode == "offline":
            session_ids = list(self.session_ids)
        else:
            if not where_ground_truth:
                current_session_id = qa_item.get("where")
                if current_session_id is not None:
                    where_ground_truth = [current_session_id]

            if where_ground_truth:
                max_session = max(int(session_id) for session_id in where_ground_truth)
                session_ids = [session_id for session_id in self.session_ids if session_id <= max_session]

        rendered_sections: List[str] = []
        for session_id in session_ids:
            session_history = self.get_session_history(session_id)
            for item in session_history:
                content = str(item.get("content", "") or "").strip()
                if content:
                    rendered_sections.append(content)

        return "\n".join(rendered_sections)

    @cache
    def get_indices(self) -> List[SampleIndex]:
        """Return all QA indices (for offline mode data splitting), excluding category=5 items."""
        # Filter out category=5 items (following mem0's approach)
        indices = []
        for idx, qa_item in enumerate(self.qa_list):
            category = qa_item.get("category", None)
            if category is not None and int(category) == 5:
                continue  # skip category=5 items
            indices.append(idx)
        return indices

    def _simple_tokenize(self, text: str) -> List[str]:
        """LoCoMo-Refined-compatible tokenization for lexical metrics."""
        return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]

    def _calculate_f1_score(self, predicted: str, gold: str) -> float:
        """Compute LoCoMo-Refined token-level F1 from bag-of-words overlap."""
        pred_tokens = self._simple_tokenize(predicted)
        gold_tokens = self._simple_tokenize(gold)

        if not pred_tokens or not gold_tokens:
            return 0.0

        pred_counts = Counter(pred_tokens)
        gold_counts = Counter(gold_tokens)
        overlap = sum(min(count, gold_counts[token]) for token, count in pred_counts.items())
        if overlap <= 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        if precision + recall == 0:
            return 0.0

        return float((2 * precision * recall) / (precision + recall))

    def _calculate_bleu_score(self, predicted: str, gold: str) -> float:
        """
        Compute LoCoMo-Refined unigram BLEU with brevity penalty.
        """
        try:
            pred_tokens = self._simple_tokenize(predicted)
            gold_tokens = self._simple_tokenize(gold)

            if not pred_tokens or not gold_tokens:
                return 0.0

            pred_counts = Counter(pred_tokens)
            gold_counts = Counter(gold_tokens)
            overlap = sum(min(count, gold_counts[token]) for token, count in pred_counts.items())
            precision = overlap / len(pred_tokens)
            if precision <= 0:
                return 0.0

            brevity_penalty = 1.0
            if len(pred_tokens) < len(gold_tokens):
                brevity_penalty = math.exp(1.0 - (len(gold_tokens) / len(pred_tokens)))
            return float(brevity_penalty * precision)
        except Exception as e:
            self.logger.warning(f"BLEU score calculation failed: {e}")
            return 0.0

    def _llm_judge(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        session_context: str = "",
    ):
        """Use an LLM judge to evaluate the answer (returns (score, response_text, reasoning_text) tuple)."""
        if not self.llm_judge_config or not self.llm_judge_client:
            if not self.llm_judge_config:
                self.logger.debug("LLM judge config is None, returning 0")
            if not self.llm_judge_client:
                self.logger.debug("LLM judge client is None, returning 0")
            return 0, "", "", "llm_judge_config or llm_judge_client not set"

        ACCURACY_PROMPT = """Your task is to label an answer as 'CORRECT' or 'WRONG' given:
(1) conversation context,
(2) a question,
(3) a gold (ground truth) answer,
(4) a generated answer.

Core principle - Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold's key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT - even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold's content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; context-grounded calendar reasoning only)
- Granularity must match exactly: HOUR->HOUR, DAY->DAY, MONTH->MONTH, YEAR->YEAR.
  Do not answer a gold at a different time unit - even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] -> WRONG)
- If no conversation context is provided, do NOT convert relative -> absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- If conversation context is provided and explicitly anchors the relative time expression (for example, the session timestamp is known), you MAY resolve relative and absolute forms as equivalent as long as:
  1. the anchor comes directly from the provided context,
  2. the resolved value has the same granularity as the gold answer, and
  3. the generated answer does not contradict the anchored interpretation.
  Example: session date is in May 2023, generated answer is "next month", and gold answer is "June 2023" -> CORRECT.
- Treat harmless modifiers in relative forms (e.g., "the/last/previous/just prior") as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover all of them.
- Extra non-contradictory items generally count as WRONG.
    - Example: gold = A, B, C ; gen = A, B, C -> CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D -> WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C -> C, C'), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include any one of them without contradiction to be CORRECT.

Now it's time for the real question:
Conversation context:
{session_context}

Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""

        attempt = 0
        while True:
            attempt += 1
            base_body = self.llm_judge_config["body"].copy()

            prompt_content = ACCURACY_PROMPT.format(
                session_context=session_context or "(none)",
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer
            )

            body = {
                **base_body,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant designed to output JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                "response_format": {"type": "json_object"},
            }

            # Use the model field from body if present; otherwise fall back to the agent name
            if "model" not in body:
                body["model"] = self.llm_judge_agent

            self.logger.info(
                "Calling LLM judge (attempt=%s): url=%s, model=%s",
                attempt,
                self.llm_judge_config.get("url", "N/A"),
                body.get("model", "N/A"),
            )

            try:
                completion = self.llm_judge_client.chat.completions.create(**body)
                choices = getattr(completion, "choices", None) or []
                if not choices:
                    self.logger.warning("LLM judge returned empty choices; retrying in 5 seconds")
                    time.sleep(5)
                    continue
                message = _message_to_dict(choices[0].message)
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content", "")

                self.logger.info(f"LLM judge response (first 500 chars): {content[:500]}...")

                if not content:
                    self.logger.warning("LLM judge returned empty content; retrying in 5 seconds")
                    time.sleep(5)
                    continue

                # Prefer the shared JSON parser used by memory modules.
                parsed = parse_llm_json_response(content, logger_prefix="LocomoJudge")
                if isinstance(parsed, dict):
                    label = str(parsed.get("label", "")).upper()
                    if label in ("CORRECT", "WRONG"):
                        score = 1 if label == "CORRECT" else 0
                        self.logger.info(f"LLM judge parsed JSON via shared parser: label={label}, score={score}")
                        return score, content, reasoning_content, ""

                # Fall back to keyword search for CORRECT / WRONG
                content_upper = content.upper()
                if "CORRECT" in content_upper:
                    # Make sure it is not part of "INCORRECT"
                    correct_idx = content_upper.find("CORRECT")
                    if correct_idx > 0:
                        prev_word = content_upper[max(0, correct_idx-10):correct_idx].strip()
                        if "IN" not in prev_word[-2:]:  # not "INCORRECT"
                            self.logger.info("LLM judge found CORRECT keyword (no JSON)")
                            return 1, content, reasoning_content, ""
                elif "WRONG" in content_upper:
                    self.logger.info("LLM judge found WRONG keyword (no JSON)")
                    return 0, content, reasoning_content, ""

                self.logger.warning(
                    "LLM judge could not parse response on attempt %s; retrying in 5 seconds. Content preview: %s...",
                    attempt,
                    content[:200],
                )
                time.sleep(5)
            except Exception as e:
                self.logger.warning(
                    "LLM judge evaluation failed on attempt %s: %s. Retrying in 5 seconds",
                    attempt,
                    e,
                    exc_info=True,
                )
                time.sleep(5)

    def _evaluate_answer(
        self,
        question: str,
        predicted: str,
        gold: Any,
        session_context: str = "",
    ) -> Tuple[Dict[str, Any]]:
        """
        Evaluate an answer LoCoMo-Refined style: normalize prediction, compare against
        every acceptable gold answer candidate, then keep the best-scoring candidate.

        Returns:
            metrics dict:
            {
                "f1_score": float,
                "bleu_score": float,
                "llm_score": float (0 or 1),
                "matched_answer": str,
                "response": str,
            }
        """
        answer_candidates = _normalize_answer_candidates(gold)
        normalized_prediction = _normalize_prediction_text(predicted)
        original_prediction = str(predicted or "").strip()

        exact_match = next((candidate for candidate in answer_candidates if normalized_prediction == candidate), None)
        if exact_match is not None:
            metrics = {
                "f1_score": 1.0,
                "bleu_score": 1.0,
                "llm_score": 1.0,
                "llm_judge_response": "",
                "llm_judge_reasoning": "",
                "llm_judge_error": "",
                "llm_reason": "Predicted answer exactly matches the reference answer.",
                "matched_answer": exact_match,
                "response": normalized_prediction,
            }
            if normalized_prediction != original_prediction:
                metrics["ori_response"] = original_prediction
            return metrics

        if normalized_prediction == "" and any(candidate != "" for candidate in answer_candidates):
            best_answer = next((candidate for candidate in answer_candidates if candidate != ""), answer_candidates[0])
            metrics = {
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "llm_score": 0.0,
                "llm_judge_response": "",
                "llm_judge_reasoning": "",
                "llm_judge_error": "",
                "llm_reason": "Predicted answer is empty while the reference answer is not.",
                "matched_answer": best_answer,
                "response": normalized_prediction,
            }
            if normalized_prediction != original_prediction:
                metrics["ori_response"] = original_prediction
            return metrics

        if not answer_candidates or all(candidate == "" for candidate in answer_candidates):
            metrics = {
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "llm_score": 0.0,
                "llm_judge_response": "",
                "llm_judge_reasoning": "",
                "llm_judge_error": "",
                "llm_reason": "",
                "matched_answer": "",
                "response": normalized_prediction,
            }
            if normalized_prediction != original_prediction:
                metrics["ori_response"] = original_prediction
            return metrics

        candidate_metrics_map: Dict[str, Dict[str, Any]] = {}
        for candidate in answer_candidates:
            f1_score = self._calculate_f1_score(normalized_prediction, candidate)
            bleu_score = self._calculate_bleu_score(normalized_prediction, candidate)
            self.logger.info(
                f"Calculating refined LLM judge score (gold='{candidate[:50]}...', predicted='{normalized_prediction[:50]}...')"
            )
            llm_score, llm_judge_response, llm_judge_reasoning, llm_judge_error = self._llm_judge(
                question, candidate, normalized_prediction, session_context=session_context
            )
            candidate_metrics_map[candidate] = {
                "f1": float(f1_score),
                "bleu": float(bleu_score),
                "llm": float(llm_score),
                "llm_judge_response": llm_judge_response,
                "llm_judge_reasoning": llm_judge_reasoning,
                "llm_judge_error": llm_judge_error,
                "llm_reason": "llm_judge_refined",
            }

        def _sort_key(candidate: str) -> Tuple[float, float, float]:
            m = candidate_metrics_map.get(candidate, {})
            return (
                float(m.get("llm", 0.0) or 0.0),
                float(m.get("f1", 0.0) or 0.0),
                float(m.get("bleu", 0.0) or 0.0),
            )

        best_answer = max(answer_candidates, key=_sort_key)
        best_metrics = candidate_metrics_map.get(best_answer, {})

        metrics = {
            "f1_score": float(best_metrics.get("f1", 0.0) or 0.0),
            "bleu_score": float(best_metrics.get("bleu", 0.0) or 0.0),
            "llm_score": float(best_metrics.get("llm", 0.0) or 0.0),
            "llm_judge_response": str(best_metrics.get("llm_judge_response", "") or ""),
            "llm_judge_reasoning": str(best_metrics.get("llm_judge_reasoning", "") or ""),
            "llm_judge_error": str(best_metrics.get("llm_judge_error", "") or ""),
            "llm_reason": str(best_metrics.get("llm_reason", "") or ""),
            "matched_answer": best_answer,
            "response": normalized_prediction,
        }
        if normalized_prediction != original_prediction:
            metrics["ori_response"] = original_prediction

        return metrics

    def _extract_answer_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the LLM's answer from a message history."""
        # Find the last assistant message with non-empty content
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    # Clean up the answer: strip formatting markers
                    content = content.strip()
                    # Remove "Answer:" and similar prefixes
                    content = re.sub(r'^(Answer|The answer|Final answer)[:\s]+', '', content, flags=re.IGNORECASE)
                    return content.strip()
        return None

    def sync_start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """Start a QA sample."""
        self.logger.info(f'Starting sample {index} with session id {session.id}')

        if index >= len(self.qa_list):
            self.logger.error(f"Invalid index {index}, total QA count: {len(self.qa_list)}")
            return TaskSampleExecutionResult(status=SampleStatus.AGENT_VALIDATION_FAILED)

        qa_item = self.qa_list[index]
        question = qa_item.get("question", "")
        gold_answer = qa_item.get("answer", "")
        category = qa_item.get("category", None)

        self.logger.info(f'[session {session.id}] Processing question: {question[:50]}...')
        self.logger.info(f'[session {session.id}] Gold answer: {gold_answer}')
        if category is not None:
            self.logger.info(f'[session {session.id}] Category: {category}')

        training_mode = str(getattr(session, "training_mode", "online") or "online")
        memory_for_enhance = getattr(session, "memory_for_enhance", None)
        is_zero_shot = (
            memory_for_enhance is None
            or memory_for_enhance.__class__.__name__ == "ZeroShotMemory"
        )
        judge_session_context = self._build_memories_text_for_qa(qa_item, training_mode)

        if is_zero_shot:
            user_prompt = f"Memories:\n\n{judge_session_context}\n\nQuestion: {question}\nAnswer:"
        else:
            # Non-zero-shot memory mechanisms should own their own prompt augmentation.
            user_prompt = f"Question: {question}\nAnswer:"

        # Inject system prompt
        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content=SYSTEM_PROMPT
        ))

        # Inject memories + question as a single user message
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=user_prompt
        ))

        # Wait for LLM response via sync_action
        try:
            response = session.sync_action()
            predicted_answer = self._extract_answer_from_messages(response.messages)

            if predicted_answer is None:
                self.logger.warning(f'[session {session.id}] No answer extracted from LLM response')
                # For locomo tasks, reward equals llm_score — 0 when no answer is extracted
                empty_metrics = {
                    "f1_score": 0.0,
                    "bleu_score": 0.0,
                    "llm_score": 0.0,
                    "llm_judge_response": "",
                    "llm_judge_reasoning": "",
                    "llm_judge_error": "No answer extracted from LLM response",
                }
                session.inject(RewardHistoryItem(
                    reward=0.0,
                    metrics=empty_metrics
                ))
                return TaskSampleExecutionResult(
                    status=SampleStatus.COMPLETED,
                    result={
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": None,
                        "category": category,
                        "metrics": empty_metrics,
                    }
                )

            self.logger.info(f'[session {session.id}] Predicted answer: {predicted_answer}')

            # Evaluate the answer (all metrics: BLEU, F1, LLM judge)
            self.logger.info(f'[session {session.id}] Starting answer evaluation...')
            if not self.llm_judge_config:
                self.logger.warning(f'[session {session.id}] LLM judge config is None! This should have been logged during initialization.')
                self.logger.warning(f'[session {session.id}] LLM judge will return 0. Check initialization logs above.')
            metrics = self._evaluate_answer(
                question,
                predicted_answer,
                gold_answer,
                session_context=judge_session_context,
            )

            # Record judge output in history for analysis/debugging.
            judge_response = str(metrics.get("llm_judge_response", "") or "")
            judge_reasoning = str(metrics.get("llm_judge_reasoning", "") or "")
            judge_error = str(metrics.get("llm_judge_error", "") or "")

            if len(judge_response) > 2000:
                judge_response = judge_response[:2000] + "...(truncated)"
            if len(judge_reasoning) > 2000:
                judge_reasoning = judge_reasoning[:2000] + "...(truncated)"

            if judge_response or judge_reasoning or judge_error:
                judge_lines = []
                if judge_response:
                    judge_lines.append(f"response: {judge_response}")
                if judge_reasoning:
                    judge_lines.append(f"reasoning: {judge_reasoning}")
                if judge_error:
                    judge_lines.append(f"error: {judge_error}")
                session.inject({
                    "role": "tool",
                    "name": "llm_judge",
                    "content": "\n".join(judge_lines)
                })

            self.logger.info(
                f'[session {session.id}] Evaluation metrics: '
                f'f1={metrics["f1_score"]:.4f}, bleu={metrics["bleu_score"]:.4f}, llm={metrics["llm_score"]:.0f}'
            )

            # For locomo tasks, reward = llm_score (0 or 1)
            session.inject(RewardHistoryItem(
                reward=metrics["llm_score"],
                metrics=metrics  # stores all metrics: f1_score, bleu_score, llm_score
            ))

            # Return result with all metrics for downstream analysis
            return TaskSampleExecutionResult(
                status=SampleStatus.COMPLETED,
                result={
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "category": category,
                    "metrics": metrics,
                }
            )

        except Exception as e:
            self.logger.error(f'[session {session.id}] Error during answer evaluation: {e}', exc_info=True)
            # For locomo tasks, reward = 0 on exception (llm_score defaults to 0)
            empty_metrics = {
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "llm_score": 0.0,
            }
            session.inject(RewardHistoryItem(
                reward=0.0,
                metrics=empty_metrics
            ))
            return TaskSampleExecutionResult(
                status=SampleStatus.AGENT_VALIDATION_FAILED,
                result={
                    "category": category,
                    "question": question if 'question' in locals() else "",
                    "gold_answer": gold_answer if 'gold_answer' in locals() else "",
                    "predicted_answer": None,
                    "metrics": empty_metrics,
                }
            )

    def get_gold_answer(self, index: SampleIndex) -> Any:
        """Return the gold answer for the given index."""
        if index >= len(self.qa_list):
            return None
        return self.qa_list[index].get("answer")
