import json
import logging
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Python 3.8 compatibility: functools.cache is available from Python 3.9+
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)

try:
    from nltk import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on conversation history.
Given a question, provide a clear and accurate answer based on the information from the conversations."""


def convert_session_to_history(session_dialogues: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert a session dialogue list to a memory-format history.

    Args:
        session_dialogues: list of dicts with format [{"speaker": "...", "dia_id": "...", "text": "..."}, ...]

    Returns:
        history: list of dicts with format [{"role": "user", "content": "'Speaker': 'text'"}, ...]
    """
    history = []
    for i, dialogue in enumerate(session_dialogues):
        role = "user" if i % 2 == 0 else "assistant"
        speaker = dialogue.get("speaker", "")
        text = dialogue.get("text", "")
        content = f"'{speaker}': '{text}'"
        history.append({
            "role": role,
            "content": content
        })
    return history


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

        logger.info(f"Successfully loaded LLM judge config: url={url}, model={body.get('model', 'N/A')}")

        return {
            "url": url,
            "headers": headers,
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
        llm_judge_agent: str = "gpt-4o-mini",  # agent name from evaluate_agent.yaml
        tokenizer_path: Optional[str] = None,   # legacy arg, ignored after switching BLEU to NLTK
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file
        self.llm_judge_agent = llm_judge_agent

        # BLEU is now aligned with mem0's NLTK-based implementation.
        if tokenizer_path:
            self.logger.info("tokenizer_path is ignored because LoCoMo BLEU now uses NLTK tokenization.")
        if not HAS_NLTK:
            self.logger.warning("nltk not installed. BLEU score will be 0. Install with: pip install nltk")

        # Load LLM judge config from configs/llmapi/evaluate_api.yaml and evaluate_agent.yaml
        self.llm_judge_config = None
        if HAS_REQUESTS:
            try:
                self.logger.info(f"Attempting to load LLM judge config for agent: '{llm_judge_agent}'")
                self.llm_judge_config = load_evaluate_agent_config(llm_judge_agent)
                if self.llm_judge_config:
                    self.logger.info(f"✓ Successfully loaded LLM judge config for agent: {llm_judge_agent}")
                    self.logger.info(f"  -> URL: {self.llm_judge_config.get('url', 'N/A')}")
                    self.logger.info(f"  -> Model: {self.llm_judge_config.get('body', {}).get('model', 'N/A')}")
                else:
                    self.logger.error(f"✗ LLM judge agent '{llm_judge_agent}' not found or config invalid. LLM judge will return 0.")
                    self.logger.error(f"  This means _llm_judge() will always return 0. Please check:")
                    self.logger.error(f"  1. Agent name '{llm_judge_agent}' exists in configs/llmapi/evaluate_agent.yaml")
                    self.logger.error(f"  2. The import file (e.g., api.yaml) exists and has correct structure")
                    self.logger.error(f"  3. The config has 'url' field in parameters")
            except Exception as e:
                self.logger.error(f"✗ Failed to load LLM judge config: {e}. LLM judge will return 0.", exc_info=True)
                self.llm_judge_config = None
        else:
            self.logger.warning("requests not installed, LLM judge will return 0")

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
        return convert_session_to_history(session_dialogues)

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
        """Mem0-compatible tokenization for F1 score calculation."""
        text = str(text).lower()
        # Match mem0's token cleanup rules before split.
        text = text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
        return text.split()

    def _calculate_f1_score(self, predicted: str, gold: str) -> float:
        """Compute token-based F1 score."""
        pred_tokens = set(self._simple_tokenize(predicted))
        gold_tokens = set(self._simple_tokenize(gold))

        if not pred_tokens or not gold_tokens:
            return 0.0

        common_tokens = pred_tokens & gold_tokens

        if not common_tokens:
            return 0.0

        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gold_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def _calculate_bleu_score(self, predicted: str, gold: str) -> float:
        """
        Compute mem0-compatible BLEU-1 using NLTK sentence_bleu + method1 smoothing.
        """
        if not HAS_NLTK:
            return 0.0

        try:
            pred_tokens = word_tokenize(str(predicted).lower(), preserve_line=True)
            ref_tokens = [word_tokenize(str(gold).lower(), preserve_line=True)]

            if not pred_tokens:
                return 0.0

            smooth = SmoothingFunction().method1
            bleu1_score = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
            return float(bleu1_score)
        except Exception as e:
            self.logger.warning(f"BLEU score calculation failed: {e}")
            return 0.0

    def _llm_judge(self, question: str, gold_answer: str, predicted_answer: str):
        """Use an LLM judge to evaluate the answer (returns (score, response_text, reasoning_text) tuple)."""
        if not self.llm_judge_config or not HAS_REQUESTS:
            if not self.llm_judge_config:
                self.logger.debug("LLM judge config is None, returning 0")
            if not HAS_REQUESTS:
                self.logger.debug("requests not available, returning 0")
            return 0, "", "", "llm_judge_config not set or requests unavailable"

        ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {predicted_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

        try:
            url = self.llm_judge_config["url"]
            headers = self.llm_judge_config["headers"]
            base_body = self.llm_judge_config["body"].copy()

            prompt_content = ACCURACY_PROMPT.format(
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

            self.logger.info(f"Calling LLM judge: url={url}, model={body.get('model', 'N/A')}")

            response = requests.post(url, headers=headers, json=body, timeout=300)
            response.raise_for_status()

            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")

            self.logger.info(f"LLM judge response (first 500 chars): {content[:500]}...")

            if not content:
                self.logger.warning("LLM judge returned empty content")
                return 0, "", reasoning_content, "LLM judge returned empty content"

            # Try multiple strategies to extract the JSON label
            label = None
            score = 0

            # Strategy 1: look for a complete JSON object containing "label"
            json_patterns = [
                r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}',  # simple JSON
                r'\{[^}]*"label"[^}]*\}',                   # original pattern
                r'\{"label"\s*:\s*"[^"]*"\}',               # minimal pattern
            ]

            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.IGNORECASE)
                if json_match:
                    try:
                        json_str = json_match.group()
                        self.logger.debug(f"Found JSON pattern: {json_str}")
                        label_data = json.loads(json_str)
                        label = label_data.get("label", "").upper()
                        if label in ("CORRECT", "WRONG"):
                            score = 1 if label == "CORRECT" else 0
                            self.logger.info(f"LLM judge parsed JSON: label={label}, score={score}")
                            return score, content, reasoning_content, ""
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"JSON parse failed for pattern '{pattern}': {e}, matched text: {json_match.group()}")
                        continue

            # Strategy 2: search for any JSON object across the entire response (may span multiple lines)
            try:
                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                for json_str in json_objects:
                    try:
                        label_data = json.loads(json_str)
                        if "label" in label_data:
                            label = str(label_data.get("label", "")).upper()
                            if label in ("CORRECT", "WRONG"):
                                score = 1 if label == "CORRECT" else 0
                                self.logger.info(f"LLM judge parsed JSON (strategy 2): label={label}, score={score}")
                                return score, content, reasoning_content, ""
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                self.logger.debug(f"Strategy 2 failed: {e}")

            # Strategy 3: fall back to keyword search for CORRECT / WRONG
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

            # Strategy 4: give up and return 0
            self.logger.warning(f"LLM judge could not parse response. Content preview: {content[:200]}...")
            self.logger.warning("No valid JSON or keyword found, returning 0")
            return 0, content, reasoning_content, "Could not parse CORRECT/WRONG from judge response"
        except Exception as e:
            self.logger.warning(f"LLM judge evaluation failed: {e}", exc_info=True)
            return 0, "", "", str(e)

    def _evaluate_answer(self, question: str, predicted: str, gold: Any) -> Tuple[Dict[str, float]]:
        """
        Evaluate an answer and compute all metrics at once: BLEU, F1, LLM judge.

        Returns:
            metrics dict:
            {
                "f1_score": float,
                "bleu_score": float,
                "llm_score": float (0 or 1),
            }
        """
        gold_str = str(gold) if gold is not None else ""
        predicted_str = str(predicted).strip()

        # Warn if gold_answer is empty, as LLM judge may not evaluate correctly
        if not gold_str:
            self.logger.warning(f"Gold answer is empty for question: {question[:50]}...")

        f1_score = self._calculate_f1_score(predicted_str, gold_str)
        bleu_score = self._calculate_bleu_score(predicted_str, gold_str)
        self.logger.info(f"Calculating LLM judge score (gold='{gold_str[:50]}...', predicted='{predicted_str[:50]}...')")
        llm_score, llm_judge_response, llm_judge_reasoning, llm_judge_error = self._llm_judge(
            question, gold_str, predicted_str
        )
        self.logger.info(f"LLM judge returned: {llm_score}")

        metrics = {
            "f1_score": f1_score,
            "bleu_score": bleu_score,
            "llm_score": float(llm_score),
            "llm_judge_response": llm_judge_response,
            "llm_judge_reasoning": llm_judge_reasoning,
            "llm_judge_error": llm_judge_error,
        }

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

        # Inject system prompt
        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content=SYSTEM_PROMPT
        ))

        # Inject question as a user message
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=question
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
            metrics = self._evaluate_answer(question, predicted_answer, gold_answer)

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
