"""Base agent class for all research agents."""

from __future__ import annotations

import abc
import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine

import structlog

from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.models.memory import EpisodicEntry, MemoryType, ProceduralEntry
from auto_researcher.utils.llm import LLMClient, LLMResponse
from auto_researcher.utils.logging import get_logger


ToolFunction = Callable[..., Coroutine[Any, Any, Any]]


class BaseAgent(abc.ABC):
    """Abstract base class for all research agents.

    Provides LLM integration, tool use framework, memory access,
    self-reflection, confidence calibration, and structured messaging.
    """

    role: AgentRole

    def __init__(self, config: ResearchConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self.state = AgentState.IDLE
        self.logger: structlog.stdlib.BoundLogger = get_logger(self.role.value)
        self._tools: dict[str, ToolFunction] = {}
        self._tool_descriptions: dict[str, str] = {}
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._outbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._episodic_memory: list[EpisodicEntry] = []
        self._procedural_memory: list[ProceduralEntry] = []
        self._semantic_memory: dict[str, Any] = {}
        self._confidence_history: list[float] = []

    # ── Tool Framework ──────────────────────────────────────────────

    def register_tool(self, name: str, fn: ToolFunction, description: str = "") -> None:
        self._tools[name] = fn
        self._tool_descriptions[name] = description or name

    async def execute_tool(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        self.logger.info("executing_tool", tool=name, args=list(kwargs.keys()))
        return await self._tools[name](**kwargs)

    def available_tools(self) -> dict[str, str]:
        return dict(self._tool_descriptions)

    # ── Memory Access ───────────────────────────────────────────────

    def write_episodic(self, content: str, tags: list[str] | None = None,
                       importance: float = 0.5) -> EpisodicEntry:
        entry = EpisodicEntry(
            id=str(uuid.uuid4()),
            content=content,
            tags=tags or [],
            source=self.role.value,
            importance=importance,
        )
        self._episodic_memory.append(entry)
        return entry

    def read_episodic(self, tags: list[str] | None = None,
                      limit: int = 10) -> list[EpisodicEntry]:
        entries = self._episodic_memory
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set & set(e.tags)]
        entries.sort(key=lambda e: e.importance, reverse=True)
        for entry in entries[:limit]:
            entry.accessed_at = datetime.now(UTC)
            entry.access_count += 1
        return entries[:limit]

    def write_semantic(self, key: str, value: Any) -> None:
        self._semantic_memory[key] = value

    def read_semantic(self, key: str, default: Any = None) -> Any:
        return self._semantic_memory.get(key, default)

    def write_procedural(self, name: str, description: str,
                         tool_sequence: list[str] | None = None,
                         code: str | None = None) -> ProceduralEntry:
        entry = ProceduralEntry(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            tool_sequence=tool_sequence or [],
            code=code,
        )
        self._procedural_memory.append(entry)
        return entry

    def read_procedural(self, name: str | None = None) -> list[ProceduralEntry]:
        if name:
            return [p for p in self._procedural_memory if p.name == name]
        return list(self._procedural_memory)

    # ── Messaging ───────────────────────────────────────────────────

    def create_message(self, receiver: AgentRole | None, task_type: str,
                       payload: dict[str, Any], priority: int = 5,
                       in_reply_to: str | None = None) -> AgentMessage:
        return AgentMessage(
            sender=self.role,
            receiver=receiver,
            task_type=task_type,
            priority=priority,
            payload=payload,
            message_id=str(uuid.uuid4()),
            in_reply_to=in_reply_to,
        )

    async def send_message(self, message: AgentMessage) -> None:
        await self._outbox.put(message)

    async def receive_message(self, timeout: float | None = None) -> AgentMessage | None:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def deliver_message(self, message: AgentMessage) -> None:
        await self._inbox.put(message)

    # ── Self-Reflection & Metacognition ─────────────────────────────

    async def self_reflect(self, context: str) -> dict[str, Any]:
        prompt = (
            f"You are a {self.role.value} agent in an autonomous research system.\n"
            f"Current state: {self.state.value}\n"
            f"Recent episodic memories: {len(self._episodic_memory)}\n"
            f"Context: {context}\n\n"
            "Reflect on:\n"
            "1. What is going well?\n"
            "2. What could be improved?\n"
            "3. What knowledge gaps do you have?\n"
            "4. What should you prioritize next?\n\n"
            "Respond with a JSON object with keys: strengths, improvements, gaps, priorities."
        )
        result = await self.llm.generate_structured(prompt, system="You are a metacognitive AI agent.")
        self.write_episodic(
            f"Self-reflection: {result}",
            tags=["reflection", "metacognition"],
            importance=0.7,
        )
        return result

    # ── Confidence Calibration ──────────────────────────────────────

    async def calibrate_confidence(self, claim: str, evidence: list[str]) -> float:
        prompt = (
            f"Assess the confidence level (0.0 to 1.0) for the following claim "
            f"based on the available evidence.\n\n"
            f"Claim: {claim}\n\n"
            f"Evidence:\n" + "\n".join(f"- {e}" for e in evidence) + "\n\n"
            "Consider:\n"
            "- Strength and quantity of evidence\n"
            "- Potential confounds\n"
            "- How well-established the underlying assumptions are\n\n"
            "Respond with JSON: {\"confidence\": <float>, \"reasoning\": \"<string>\"}"
        )
        result = await self.llm.generate_structured(prompt)
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        self._confidence_history.append(confidence)
        return confidence

    def mean_confidence(self) -> float:
        if not self._confidence_history:
            return 0.5
        return sum(self._confidence_history) / len(self._confidence_history)

    # ── LLM Helpers ─────────────────────────────────────────────────

    async def _ask_llm(self, prompt: str, system: str = "",
                       temperature: float | None = None) -> str:
        response = await self.llm.generate(prompt, system=system, temperature=temperature)
        return response.content

    async def _ask_llm_structured(self, prompt: str, system: str = "",
                                  temperature: float | None = None) -> dict[str, Any]:
        return await self.llm.generate_structured(prompt, system=system, temperature=temperature)

    # ── State Management ────────────────────────────────────────────

    def set_state(self, state: AgentState) -> None:
        self.logger.info("state_change", old=self.state.value, new=state.value)
        self.state = state

    # ── Abstract Interface ──────────────────────────────────────────

    @abc.abstractmethod
    async def execute(self, task: AgentMessage) -> AgentMessage:
        """Execute a task and return a result message."""
        ...

    async def run(self) -> None:
        """Main agent loop: receive tasks and execute them."""
        self.set_state(AgentState.IDLE)
        while True:
            message = await self.receive_message()
            if message is None:
                continue
            self.set_state(AgentState.WORKING)
            try:
                result = await self.execute(message)
                await self.send_message(result)
            except Exception as exc:
                self.logger.error("agent_error", error=str(exc))
                self.set_state(AgentState.ERROR)
                error_msg = self.create_message(
                    receiver=message.sender,
                    task_type="error",
                    payload={"error": str(exc), "original_task": message.task_type},
                    in_reply_to=message.message_id,
                )
                await self.send_message(error_msg)
            finally:
                if self.state != AgentState.ERROR:
                    self.set_state(AgentState.IDLE)
