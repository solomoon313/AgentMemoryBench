from __future__ import annotations

import json
import logging
import re
from typing import Protocol, List, Dict, Any, Optional, Type, TypeVar

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    logging.warning("json_repair not installed. Install with: pip install json-repair")

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    logging.warning("pydantic not installed. Install with: pip install pydantic")


T = TypeVar('T', bound='BaseModel')


def parse_llm_json_response(
    response_text: str,
    schema: Optional[Type[T]] = None,
    logger_prefix: str = "Memory"
) -> Optional[Dict[str, Any] | T]:
    """
    General-purpose LLM JSON response parser with 3-layer progressive fault tolerance:

    1. Output cleaning: strip markdown code blocks, comments, etc., keeping only JSON content
    2. Smart repair: use json_repair to automatically fix common format errors (rescues ~90% of malformed output)
    3. Schema validation: validate types and business logic with Pydantic (optional)

    Args:
        response_text: raw text returned by the LLM
        schema: Pydantic model class (optional) for schema validation and type coercion
        logger_prefix: log prefix (default "Memory")

    Returns:
        - If schema is provided: returns a Pydantic model instance
        - If no schema: returns a Dict
        - On parse failure: returns None

    Example:
        # Basic usage (parse JSON only)
        result = parse_llm_json_response(llm_output)

        # With schema validation
        class MySchema(BaseModel):
            id: str
            event: str
            text: str

        result = parse_llm_json_response(llm_output, schema=MySchema)
    """
    logger = logging.getLogger(logger_prefix)

    # ========== Step 1: Output cleaning ==========
    cleaned = _clean_llm_output(response_text)
    logger.debug(f"[{logger_prefix}] Step 1: Output cleaning completed, length: {len(cleaned)}")

    # ========== Step 2: Smart repair (json_repair) ==========
    repaired = _smart_repair(cleaned, logger_prefix)
    if repaired is None:
        logger.warning(f"[{logger_prefix}] Step 2: Smart repair failed, trying fallback methods")
        # If json_repair fails, fall back to bracket matching
        repaired = _extract_json_by_bracket_matching(cleaned, logger_prefix)

    if repaired is None:
        logger.error(f"[{logger_prefix}] Step 2: All repair methods failed")
        return None

    # ========== Step 3: Schema validation (Pydantic) ==========
    if schema is not None:
        validated = _validate_schema(repaired, schema, logger_prefix)
        if validated is None:
            logger.warning(f"[{logger_prefix}] Step 3: Schema validation failed")
            return None
        logger.debug(f"[{logger_prefix}] Step 3: Schema validation passed")
        return validated

    # No schema — return the dict directly
    logger.debug(f"[{logger_prefix}] Parsing completed (no schema validation)")
    return repaired


def _clean_llm_output(text: str) -> str:
    """Step 1: Output cleaning — strip markdown, comments, etc."""
    # Remove markdown code block markers
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*$', '', cleaned)

    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)

    # Remove JavaScript-style single-line comments (but preserve // in URLs)
    cleaned = re.sub(r'(?<!:)//[^\n]*', '', cleaned)

    # Remove C-style multi-line comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

    return cleaned.strip()


def _smart_repair(text: str, logger_prefix: str) -> Optional[Dict[str, Any]]:
    """Step 2: Smart repair — use json_repair to rescue ~90% of malformed output"""
    logger = logging.getLogger(logger_prefix)

    if not HAS_JSON_REPAIR:
        logger.debug(f"[{logger_prefix}] json_repair not available, skipping smart repair")
        return None

    try:
        repaired_str = repair_json(text)
        result = json.loads(repaired_str)
        logger.debug(f"[{logger_prefix}] Smart repair succeeded")
        return result
    except Exception as e:
        logger.debug(f"[{logger_prefix}] Smart repair failed: {e}")
        return None


def _extract_json_by_bracket_matching(text: str, logger_prefix: str) -> Optional[Dict[str, Any]]:
    """Fallback: extract a JSON object using bracket matching"""
    logger = logging.getLogger(logger_prefix)

    try:
        # Find the position of the first {
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Use bracket matching to find the corresponding closing }
        brace_count = 0
        in_string = False
        escape_next = False
        end_idx = start_idx

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

        if end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            result = json.loads(json_str)
            logger.debug(f"[{logger_prefix}] Bracket matching extraction succeeded")
            return result
    except Exception as e:
        logger.debug(f"[{logger_prefix}] Bracket matching extraction failed: {e}")

    return None


def _validate_schema(data: Dict[str, Any], schema: Type[T], logger_prefix: str) -> Optional[T]:
    """Step 3: Schema validation — use Pydantic to verify types and business logic"""
    logger = logging.getLogger(logger_prefix)

    if not HAS_PYDANTIC:
        logger.warning(f"[{logger_prefix}] pydantic not available, skipping schema validation")
        return data  # return the raw dict

    try:
        validated = schema(**data)
        return validated
    except ValidationError as e:
        logger.debug(f"[{logger_prefix}] Schema validation failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"[{logger_prefix}] Schema validation error: {e}")
        return None


class MemoryMechanism(Protocol):
    """
    Abstract interface for memory mechanisms.

    - use_memory: rewrite messages before calling the LLM (e.g. inject few-shot examples or past experience)
    - update_memory: update the memory store after a sample ends, based on the full history and result

    Note: this interface is task-agnostic (DBBench / OS / KG / ALFWorld, etc.).
    It only operates on the unified OpenAI Chat format: [{role, content, ...}, ...].
    """

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given the current task name and raw messages, return memory-augmented messages.

        For zero_shot this is typically a pass-through.
        """

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """
        Called after a single sample finishes; writes the new trajectory/result into memory.

        For zero_shot this is typically a no-op.
        """


