"""Agent communication and state models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    LITERATURE_ANALYST = "literature_analyst"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    EXPERIMENT_DESIGNER = "experiment_designer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    SCIENCE_COMMUNICATOR = "science_communicator"
    STATISTICIAN = "statistician"


class AgentState(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Structured message for inter-agent communication."""

    sender: AgentRole
    receiver: AgentRole | None = None  # None = broadcast
    task_type: str
    priority: int = Field(ge=0, le=10, default=5)
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_output_format: str = "json"
    deadline: datetime | None = None
    in_reply_to: str | None = None
    message_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
