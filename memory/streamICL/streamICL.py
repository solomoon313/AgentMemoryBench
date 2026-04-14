from __future__ import annotations

from typing import List, Dict, Any, Optional
import yaml
import json
import re
import random
import numpy as np
import logging

from src.utils.message_schema import (
    extract_message_info,
    enhance_messages_with_memory,
    extract_original_question
)

try:
    import torch
    import faiss
    from transformers import AutoTokenizer, AutoModel
    HAS_DEPENDENCIES = True
except ImportError: #"except" wiil not disturb the execution of file
    HAS_DEPENDENCIES = False
    print("Warning: faiss, torch, or transformers not installed. StreamICL will not work.")

from ..base import MemoryMechanism


class RAG:
    """
    RAG (Retrieval-Augmented Generation) vector retrieval system.
    Based on the stream-bench implementation.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
        order: str = "similar_at_top",  # "similar_at_top" | "similar_at_bottom" | "random"
        seed: int = 42,
    ):
        if not HAS_DEPENDENCIES:
            raise ImportError("faiss, torch, or transformers not installed. Please install them to use StreamICL.") #"raise" wiil not disturb the execution of file

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model).eval() #plz remeber to plus ".eval" to shut down dropout or sth else

        self.index = None  # FAISS vector index, latter we will use it to create a VectorDatabse
        self.id2evidence = dict()  # stores raw text values
        # Read embedding dimension directly from model config to avoid unnecessary inference
        self.embed_dim = self.embed_model.config.hidden_size
        self.insert_acc = 0  # insertion counter

        self.seed = seed
        self.top_k = top_k
        self.order = order
        random.seed(self.seed)

        self.create_faiss_index()#creating FAISS vector index

    def create_faiss_index(self):
        """Create FAISS vector index."""
        self.index = faiss.IndexFlatL2(self.embed_dim)

    def encode_data(self, sentence: str) -> np.ndarray:
        """Encode text into an embedding vector."""
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # CLS pooling
            sentence_embeddings = model_output[0][:, 0]
        feature = sentence_embeddings.numpy()[0] #making sure only the first sentence was embedded
        norm = np.linalg.norm(feature)
        return feature / norm #This an vector whose length is 768

    def insert(self, key: str, value: str) -> None:
        """
        Insert an experience into the RAG index.

        Args:
            key: question text (used as the retrieval key)
            value: formatted experience chunk (the stored value)
        """
        embedding = self.encode_data(key).astype('float32')
        self.index.add(np.expand_dims(embedding, axis=0))#FAISS only accept 2-dim data
        self.id2evidence[str(self.insert_acc)] = value
        self.insert_acc += 1

    def retrieve(self, query: str, top_k: int) -> List[str]:
        """
        Retrieve the top_k most similar experiences.

        Args:
            query: current question text
            top_k: number of results to retrieve

        Returns:
            list of retrieved formatted chunks
        """
        if self.insert_acc == 0:#which means there is no data in the Vector Database
            return []

        embedding = self.encode_data(query).astype('float32')
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]

        # Reorder results according to the configured ordering strategy
        if self.order == "similar_at_bottom":
            results = list(reversed(results))
        elif self.order == "random":
            random.shuffle(results)

        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list


class StreamICLMemory(MemoryMechanism):
    """
    StreamICL memory mechanism: RAG-based in-context learning system.

    Based on the stream-bench implementation:
    - Uses a FAISS vector database to store experiences
    - Retrieves top_k most similar experiences by embedding similarity to the current question
    - Optionally filters to only store successful samples
    - Supports two injection positions: appended before or after the user message
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
        order: str = "similar_at_top",  # "similar_at_top" | "similar_at_bottom" | "random"
        success_only: bool = True,  # True: only store samples with status=="completed", False: store all
        reward_bigger_than_zero: bool = False,  # True: only store samples with reward>0, False: store all
        prompt_template: str = "Here are some examples of the task you have completed:\n\n{examples}",
        where: str = "tail",  # "tail": inject after user question | "front": inject before user question
        seed: int = 42,
    ):
        """
        Initialize the StreamICL memory mechanism.

        Args:
            embedding_model: embedding model used to encode questions
            top_k: number of similar experiences to retrieve
            order: ordering strategy for retrieved results
            success_only: if True, only store samples with status=="completed"
            reward_bigger_than_zero: if True, only store samples with reward>0
            prompt_template: template string for formatting the memory block
            where: injection position — "tail" appends after the question, "front" prepends before it
            seed: random seed (used only when order="random")
        """
        if not HAS_DEPENDENCIES:
            raise ImportError("faiss, torch, or transformers not installed. Please install them to use StreamICL.") #"raise" will disturb the execution of the file

        # Save RAG configuration for lazy/re-initialization
        self.rag_config = dict(
            embedding_model=embedding_model,
            top_k=top_k,
            order=order,
            seed=seed,
        )
        self.success_only = success_only
        self.reward_bigger_than_zero = reward_bigger_than_zero
        self.prompt_template = prompt_template
        self.where = where

        # Extract the template title from the prompt_template (used to detect injected memory)
        # e.g. "Here are some examples:\n\n{examples}" -> "Here are some examples:"
        self.template_title = self.prompt_template.split('{examples}')[0].strip()#clearing the " " and "\n"

        # Single global vector store (not partitioned by task)
        self.rag: Optional[RAG] = None
        try:
            self.rag = RAG(**self.rag_config)
        except Exception as e:
            raise ImportError(f"Failed to initialize RAG: {e}")

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given the task name and original messages, return messages enhanced with retrieved memory.
        The memory content is injected into the first user message.
        """
        # Extract the current question text (strips any previously injected memory)
        template_titles = [self.template_title]
        question = extract_original_question(messages, where=self.where, template_titles=template_titles)
        if not question:
            return list(messages) if messages is not None else []  # fallback; rarely triggered

        # Retrieve similar experiences from the global vector store
        if not self.rag:
            return list(messages) if messages is not None else []  # RAG not initialized; rarely triggered

        shots = self.rag.retrieve(query=question, top_k=self.rag.top_k)
        if not shots:
            return list(messages) if messages is not None else []  # no relevant memory found; rarely triggered

        # Format the retrieved experiences into a single memory block
        fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
        memory_content = self.prompt_template.format(examples=fewshot_text)

        # Inject the memory block into the messages
        return enhance_messages_with_memory(messages, memory_content, where=self.where)

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """
        Called after a single sample finishes execution.
        Writes the new trajectory/result into the memory store.
        """
        status = result.get("status", "")
        reward = result.get("reward", 0)
        # is_success is based solely on status, independent of reward
        is_success = status == "completed"

        # Filter: if success_only=True, skip samples that did not complete successfully
        if self.success_only and not is_success:
            print(f"[StreamICL] Skipping sample storage: success_only=True but sample not completed (status={status}, task={task})")
            return

        # Filter: if reward_bigger_than_zero=True, skip samples with non-positive reward
        if self.reward_bigger_than_zero:
            if reward <= 0:
                print(f"[StreamICL] Skipping sample storage: reward_bigger_than_zero=True but reward={reward} (task={task})")
                return

        # Extract the question text from history (used as the retrieval key)
        template_titles = [self.template_title]
        question = extract_original_question(history, where=self.where, template_titles=template_titles)
        if not question:
            print(f"[StreamICL] Skipping sample storage: No question extracted from history (task={task})")
            return

        # Format the trajectory into an experience chunk
        chunk = self._format_experience(history)

        # Insert into the global vector store (not partitioned by task)
        if not self.rag:
            return
        self.rag.insert(key=question, value=chunk)

    def _format_experience(self, history: List[Dict[str, Any]]) -> str:
        """
        Serialize a conversation history into a plain-text experience chunk.

        Output format:
            Question: {question}
            {answer}

        Important: injected few-shot examples from use_memory() must be stripped out;
        only the original question and the agent's actual trajectory are stored.
        """
        template_prefix = self.template_title

        # Step 1: extract the original question (strips any injected memory)
        template_titles = [template_prefix]
        question = extract_original_question(history, where=self.where, template_titles=template_titles)

        if not question:
            return ""

        # Step 2: build the answer lines from the trajectory (skip the first user message)
        answer_lines = []
        skip_first_user = True

        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None or role == "system":#system prompt is useless, so that we don't keep it
                continue

            content = content if content else ""

            # Skip the first user message — it is already captured as the question
            if role == "user":
                if skip_first_user:
                    skip_first_user = False#the first message from user we don't keep it too
                    continue
                # Subsequent user messages: keep only those that don't contain injected memory
                if template_prefix not in content:
                    answer_lines.append(f"User: {content}")
                continue

            # Format assistant messages
            if role == "assistant":
                tool_calls = msg_dict.get("tool_calls", []) if msg_dict else []
                reasoning_content = msg_dict.get("reasoning_content", "") if msg_dict else ""
                reasoning_content = reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content
                think_part = f"<think>{reasoning_content}</think> " if reasoning_content else ""

                if tool_calls:
                    tool_calls_info = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        func_name = func.get("name", "unknown")
                        func_args = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(func_args)  # 1. parse JSON string to Python object  2. decode escape sequences (e.g. \uXXXX -> actual char)
                            args_str = json.dumps(args_dict, ensure_ascii=False)  # 1. serialize Python object back to JSON string  2. re-escape special chars, but preserve non-ASCII (e.g. Chinese) as-is
                        except:
                            args_str = func_args
                        tool_calls_info.append(f"{func_name}({args_str})")
                    tool_calls_str = " ".join(tool_calls_info)
                    answer_lines.append(f"Assistant: {think_part}{content} {tool_calls_str}")
                else:
                    answer_lines.append(f"Assistant: {think_part}{content}")

            # Format tool messages (truncate long outputs)
            elif role == "tool":
                tool_content = content[:500] + "..." if len(content) > 500 else content
                answer_lines.append(f"Tool: {tool_content}")

        answer = "\n".join(answer_lines) if answer_lines else "Completed successfully."
        return f"Question: {question}\n{answer}"


def load_stream_icl_from_yaml(config_path: str) -> StreamICLMemory:
    """
    Load configuration from memory/streamICL/streamICL.yaml and construct a StreamICLMemory.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {} #if safe_loaf gets empty file, it will return None. This situation is very dangerous

    stream_icl_cfg = cfg.get("stream_icl", {})
    rag_cfg = stream_icl_cfg.get("rag", {})

    embedding_model = rag_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
    top_k = rag_cfg.get("top_k", 4)
    order = rag_cfg.get("order", "similar_at_top")
    seed = rag_cfg.get("seed", 42)

    success_only = bool(stream_icl_cfg.get("success_only", True))
    reward_bigger_than_zero = bool(stream_icl_cfg.get("reward_bigger_than_zero", False))
    prompt_template = stream_icl_cfg.get("prompt_template", "Here are some examples of the task you have completed:\n\n{examples}")
    where = stream_icl_cfg.get("where", "tail")

    return StreamICLMemory(
        embedding_model=embedding_model,
        top_k=top_k,
        order=order,
        success_only=success_only,
        reward_bigger_than_zero=reward_bigger_than_zero,
        prompt_template=prompt_template,
        where=where,
        seed=seed,
    )
