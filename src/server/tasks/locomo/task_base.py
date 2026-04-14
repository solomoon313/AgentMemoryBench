import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from .typings import (
    AgentCancelledException,
    AgentOutput,
    AgentOutputStatus,
    ChatHistoryItem,
    FullHistory,
    HistoryItem,
    SampleIndex,
    TaskOutput,
    TaskSampleExecutionResult,
    ToolList
)


class SessionController:
    def __init__(self):
        self.agent_lock = asyncio.Lock()
        self.env_lock = asyncio.Lock()
        self.agent_signal = asyncio.Semaphore(0)
        self.env_signal = asyncio.Semaphore(0)
        self.env_input: Optional[AgentOutput] = None
        self.env_output = TaskOutput()
        self.full_history = False

    async def agent_pull(
        self, env_input: Optional[AgentOutput] = None
    ) -> TaskOutput:
        async with self.agent_lock:
            if env_input is not None:
                self.env_input = env_input
                self.env_signal.release()
            await self.agent_signal.acquire()
            return self.env_output

    async def env_pull(self, history: List[HistoryItem]) -> AgentOutput:
        async with self.env_lock:
            if self.full_history:
                self.env_output.history_ptr = 0
            else:
                self.env_output.history_ptr = len(self.env_output.history)
            self.env_output.history = history.copy()
            self.agent_signal.release()
            await self.env_signal.acquire()
            return self.env_input

    async def env_finish(self, result: TaskOutput = None) -> None:
        async with self.env_lock:
            history = self.env_output.history
            self.env_output = result
            if self.full_history:
                self.env_output.history_ptr = 0
            else:
                self.env_output.history_ptr = len(history)
            if self.env_output.history is None:
                self.env_output.history = history
            self.agent_signal.release()

    def get_status(self):
        waiting_for_env = self.agent_lock.locked()
        waiting_for_agent = self.env_lock.locked()
        return {
            "waiting_for_env": waiting_for_env,
            "waiting_for_agent": waiting_for_agent,
            "env_input": self.env_input,
            "env_output": self.env_output.model_dump(mode='json'),
        }


class Session:
    def __init__(self, session_id: int) -> None:
        self.id: int = session_id
        self.history: List[HistoryItem] = []
        self.controller = SessionController()
        self.tools: Optional[list] = None
        self.loop: Optional[asyncio.EventLoop] = None

    def inject(self, item):
        if not item:
            return
        if isinstance(item, List):
            for sub_item in item:
                self.inject(sub_item)
        else:
            HistoryItem.model_validate(item)
            self.history.append(item)

    def clear(self):
        self.history = []

    def set_full_history(self, full_history: bool = True):
        self.controller.full_history = full_history

    def cover(self, items: list):
        FullHistory.model_validate(items)
        self.history = items.copy()

    def sync_action(self, *injection) -> AgentOutput:
        assert self.loop is not None, "Event loop is not set"
        return asyncio.run_coroutine_threadsafe(self.action(*injection), self.loop).result()

    async def action(self, *injection) -> AgentOutput:
        self.inject(list(injection))
        agent_response = await self.controller.env_pull(self.history)
        if agent_response.status == AgentOutputStatus.CANCELLED:
            raise AgentCancelledException()
        for message in agent_response.messages:
            ChatHistoryItem.model_validate(message)
            self.history.append(message)
        self.controller.env_output.history = self.history.copy()
        return agent_response

    def set_tools(self, tools: Optional[list] = None):
        self.tools = tools
        ToolList.model_validate(self.tools)


class Task:

    def __init__(self,
                 name: str,
                 concurrency: int = 16,
                 tools: Optional[list] = None,
                 full_async: bool = False,
                 *args,
                 **kwargs):
        """
        :param name: Name of the task, will be used as an identifier for the task and is displayed in the dashboard.
        :param concurrency: Max number of concurrent sessions that one worker can handle.
        :param tools: If the task uses function calling for interaction, specify the tools in OpenAI format here.
        :param full_async: If True, the task is considered fully asynchronous,
                           meaning that cancellation of a session will directly `cancel()` the coroutine.
                           This should only be set to True if the task does not call any blocking code,
                           even if through to_thread or run_in_executor.
        """
        self.name = name
        self.concurrency = concurrency
        self.tools = tools
        self.full_async = full_async

    def get_indices(self) -> List[SampleIndex]:
        """
        Return a list of indices for the task. Indices can be str or int.
        """
        raise NotImplementedError()

    def sync_start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """
        Synchronous version of `start_sample`, could use `session.sync_action()` instead of `await session.action()`.
        """
        raise NotImplementedError()

    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        """
        Start a sample with the given index and session.
        The default implementation is to call `sync_start_sample` in a thread pool executor.
        """

        if self.full_async:
            raise NotImplementedError('Full async tasks must implement async start sample')

        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()
        session.loop = loop
        result = await loop.run_in_executor(
            executor,
            self.sync_start_sample,
            index,
            session,
        )
        return result

    async def start_sample_custom(self, task: dict, session: Session) -> TaskSampleExecutionResult:
        """
        For tasks that would like to support custom tasks, include -1 in indices and override this method.
        """
        raise NotImplementedError()

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        raise NotImplementedError()

    def release(self):
        pass

