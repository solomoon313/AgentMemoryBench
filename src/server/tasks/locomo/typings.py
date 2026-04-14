import builtins
from enum import Enum
from typing import List, Optional, Union, Any, Dict

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel, model_validator, RootModel, field_validator

JSONSerializable = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]
SampleIndex = Union[int, str]


class InstanceFactory(BaseModel):
    module: str
    parameters: dict = {}

    @classmethod
    @field_validator('parameters', mode='before')
    def _ensure_dict(cls, v: Optional[dict]) -> dict:
        if v is None:
            return {}
        return v

    def create(self):
        splits = self.module.split(".")
        if len(splits) == 0:
            raise Exception("Invalid module name: {}".format(self.module))
        if len(splits) == 1:
            g = globals()
            if self.module in g:
                class_type = g[self.module]
            else:
                class_type = getattr(builtins, self.module)
            return class_type(**self.parameters)
        else:
            path = ".".join(self.module.split(".")[:-1])
            mod = __import__(path, fromlist=[self.module.split(".")[-1]])
            return getattr(mod, self.module.split(".")[-1])(**self.parameters)


class SampleStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_CONTEXT_LIMIT = "agent context limit"
    AGENT_VALIDATION_FAILED = "agent validation failed"
    AGENT_INVALID_ACTION = "agent invalid action"
    TASK_LIMIT_REACHED = "task limit reached"
    UNKNOWN = "unknown"
    TASK_ERROR = "task error"
    CANCELLED = "cancelled"


class AgentOutputStatus(str, Enum):
    NORMAL = "normal"
    CANCELLED = "cancelled"


ChatHistoryItem = RootModel[ChatCompletionMessageParam]

ChatHistory = RootModel[List[ChatCompletionMessageParam]]

ToolList = RootModel[Optional[List[ChatCompletionToolParam]]]


class RewardHistoryItem(BaseModel):
    reward: float
    score: Optional[float] = None
    metrics: dict = {}

    def __init__(self, **data):
        super().__init__(**data)
        # score is no longer auto-added to metrics to avoid duplication.
        # score and metrics are independent fields:
        # - score: sample quality score used by memory mechanisms (for ranking and selection)
        # - metrics: detailed evaluation metrics (f1_score, bleu_score, llm_score, etc.)


HistoryItem = RootModel[Union[ChatCompletionMessageParam, RewardHistoryItem]]

FullHistory = RootModel[List[Union[ChatCompletionMessageParam, RewardHistoryItem]]]


class TaskOutput(BaseModel):
    index: Optional[SampleIndex] = None
    status: SampleStatus = SampleStatus.RUNNING
    result: JSONSerializable = None
    history: List[HistoryItem] = []
    history_ptr: int = 0


class TaskSampleExecutionResult(BaseModel):
    status: SampleStatus = SampleStatus.COMPLETED
    result: JSONSerializable = None


class AgentOutput(BaseModel):
    status: AgentOutputStatus = AgentOutputStatus.NORMAL
    messages: Optional[List[dict]] = None

    @model_validator(mode='after')
    def post_validate(self):
        # at least one of them should be not None
        assert self.status is not AgentOutputStatus.NORMAL or self.messages, \
            'If status is NORMAL, content should not be None'
        RootModel[Optional[List[ChatCompletionAssistantMessageParam]]].model_validate(self.messages)
        return self


class RegisterRequest(BaseModel):
    name: str
    address: str
    concurrency: int
    indices: list


class StartSampleRequest(BaseModel):
    name: str
    index: SampleIndex
    custom_task: Optional[dict] = None


class InteractRequest(BaseModel):
    session_id: int
    messages: Optional[List[dict]]


class CancelRequest(BaseModel):
    session_id: int


class HeartbeatRequest(BaseModel):
    name: str
    address: str


class CalculateOverallRequest(BaseModel):
    name: str
    results: List[TaskOutput]


class WorkerStartSampleRequest(BaseModel):
    index: SampleIndex
    custom_task: Optional[dict] = None
    session_id: int


class SampleStatusRequest(BaseModel):
    session_id: int


class AgentCancelledException(BaseException):
    def __init__(self, detail: Union[str, None] = None) -> None:
        super().__init__("agent_cancelled", detail)

