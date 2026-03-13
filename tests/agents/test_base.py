"""Tests for BaseAgent class."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from auto_researcher.agents.base import BaseAgent
from auto_researcher.config import ResearchConfig
from auto_researcher.models.agent import AgentMessage, AgentRole, AgentState
from auto_researcher.models.memory import EpisodicEntry, ProceduralEntry
from auto_researcher.utils.llm import LLMClient, LLMResponse


# Concrete subclass for testing the abstract BaseAgent
class ConcreteAgent(BaseAgent):
    role = AgentRole.LITERATURE_ANALYST

    async def execute(self, task: AgentMessage) -> AgentMessage:
        return self.create_message(
            receiver=task.sender,
            task_type="result",
            payload={"status": "done"},
            in_reply_to=task.message_id,
        )


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig()


@pytest.fixture
def mock_llm(config: ResearchConfig) -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock(return_value=LLMResponse(content="test", model="test"))
    llm.generate_structured = AsyncMock(return_value={
        "confidence": 0.8,
        "reasoning": "test",
        "strengths": [],
        "improvements": [],
        "gaps": [],
        "priorities": [],
    })
    return llm


@pytest.fixture
def agent(config: ResearchConfig, mock_llm: LLMClient) -> ConcreteAgent:
    return ConcreteAgent(config, mock_llm)


class TestBaseAgentInit:
    def test_initial_state_is_idle(self, agent: ConcreteAgent) -> None:
        assert agent.state == AgentState.IDLE

    def test_role_is_set(self, agent: ConcreteAgent) -> None:
        assert agent.role == AgentRole.LITERATURE_ANALYST

    def test_empty_tools(self, agent: ConcreteAgent) -> None:
        assert agent._tools == {}
        assert agent._tool_descriptions == {}

    def test_empty_memories(self, agent: ConcreteAgent) -> None:
        assert agent._episodic_memory == []
        assert agent._procedural_memory == []
        assert agent._semantic_memory == {}

    def test_empty_confidence_history(self, agent: ConcreteAgent) -> None:
        assert agent._confidence_history == []


class TestToolFramework:
    @pytest.mark.asyncio
    async def test_register_and_execute_tool(self, agent: ConcreteAgent) -> None:
        async def my_tool(x: int) -> int:
            return x * 2

        agent.register_tool("double", my_tool, "Doubles a number")
        result = await agent.execute_tool("double", x=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self, agent: ConcreteAgent) -> None:
        with pytest.raises(ValueError, match="Unknown tool: nonexistent"):
            await agent.execute_tool("nonexistent")

    def test_register_tool_without_description(self, agent: ConcreteAgent) -> None:
        async def noop() -> None:
            pass

        agent.register_tool("noop", noop)
        assert agent._tool_descriptions["noop"] == "noop"

    def test_available_tools(self, agent: ConcreteAgent) -> None:
        async def tool_a() -> None:
            pass

        async def tool_b() -> None:
            pass

        agent.register_tool("a", tool_a, "Tool A")
        agent.register_tool("b", tool_b, "Tool B")
        tools = agent.available_tools()
        assert tools == {"a": "Tool A", "b": "Tool B"}


class TestMemory:
    def test_write_and_read_episodic(self, agent: ConcreteAgent) -> None:
        entry = agent.write_episodic("test event", tags=["test"], importance=0.9)
        assert isinstance(entry, EpisodicEntry)
        assert entry.content == "test event"
        assert entry.importance == 0.9
        assert entry.tags == ["test"]
        assert entry.source == "literature_analyst"

        entries = agent.read_episodic()
        assert len(entries) == 1
        assert entries[0].content == "test event"
        assert entries[0].access_count == 1

    def test_read_episodic_filters_by_tags(self, agent: ConcreteAgent) -> None:
        agent.write_episodic("a", tags=["alpha"])
        agent.write_episodic("b", tags=["beta"])
        agent.write_episodic("c", tags=["alpha", "gamma"])

        results = agent.read_episodic(tags=["alpha"])
        assert len(results) == 2

    def test_read_episodic_sorts_by_importance(self, agent: ConcreteAgent) -> None:
        agent.write_episodic("low", importance=0.1)
        agent.write_episodic("high", importance=0.9)
        agent.write_episodic("mid", importance=0.5)

        results = agent.read_episodic()
        assert results[0].content == "high"
        assert results[1].content == "mid"
        assert results[2].content == "low"

    def test_read_episodic_limit(self, agent: ConcreteAgent) -> None:
        for i in range(20):
            agent.write_episodic(f"event {i}", importance=i / 20.0)

        results = agent.read_episodic(limit=5)
        assert len(results) == 5

    def test_write_and_read_semantic(self, agent: ConcreteAgent) -> None:
        agent.write_semantic("key1", {"data": 42})
        assert agent.read_semantic("key1") == {"data": 42}
        assert agent.read_semantic("missing") is None
        assert agent.read_semantic("missing", "default") == "default"

    def test_write_and_read_procedural(self, agent: ConcreteAgent) -> None:
        entry = agent.write_procedural(
            "search_papers",
            "Search for relevant papers",
            tool_sequence=["arxiv_search", "filter", "rank"],
            code="print('hello')",
        )
        assert isinstance(entry, ProceduralEntry)
        assert entry.name == "search_papers"
        assert entry.tool_sequence == ["arxiv_search", "filter", "rank"]
        assert entry.code == "print('hello')"

        results = agent.read_procedural(name="search_papers")
        assert len(results) == 1

        results = agent.read_procedural(name="nonexistent")
        assert len(results) == 0

        all_results = agent.read_procedural()
        assert len(all_results) == 1


class TestMessaging:
    def test_create_message(self, agent: ConcreteAgent) -> None:
        msg = agent.create_message(
            receiver=AgentRole.CRITIC,
            task_type="review",
            payload={"text": "hello"},
            priority=3,
        )
        assert msg.sender == AgentRole.LITERATURE_ANALYST
        assert msg.receiver == AgentRole.CRITIC
        assert msg.task_type == "review"
        assert msg.priority == 3
        assert msg.message_id != ""

    def test_create_message_broadcast(self, agent: ConcreteAgent) -> None:
        msg = agent.create_message(
            receiver=None,
            task_type="announcement",
            payload={},
        )
        assert msg.receiver is None

    @pytest.mark.asyncio
    async def test_send_and_receive_message(self, agent: ConcreteAgent) -> None:
        msg = agent.create_message(
            receiver=AgentRole.CRITIC,
            task_type="test",
            payload={},
        )
        await agent.deliver_message(msg)
        received = await agent.receive_message(timeout=1.0)
        assert received is not None
        assert received.task_type == "test"

    @pytest.mark.asyncio
    async def test_receive_message_timeout(self, agent: ConcreteAgent) -> None:
        result = await agent.receive_message(timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_message_goes_to_outbox(self, agent: ConcreteAgent) -> None:
        msg = agent.create_message(
            receiver=AgentRole.CRITIC,
            task_type="test",
            payload={},
        )
        await agent.send_message(msg)
        outbox_msg = await agent._outbox.get()
        assert outbox_msg.task_type == "test"


class TestStateManagement:
    def test_set_state(self, agent: ConcreteAgent) -> None:
        agent.set_state(AgentState.WORKING)
        assert agent.state == AgentState.WORKING

        agent.set_state(AgentState.ERROR)
        assert agent.state == AgentState.ERROR


class TestSelfReflection:
    @pytest.mark.asyncio
    async def test_self_reflect(self, agent: ConcreteAgent) -> None:
        result = await agent.self_reflect("testing context")
        assert isinstance(result, dict)
        # Verify LLM was called
        agent.llm.generate_structured.assert_called_once()
        # Verify episodic memory was written
        assert len(agent._episodic_memory) == 1
        assert "reflection" in agent._episodic_memory[0].tags


class TestConfidenceCalibration:
    @pytest.mark.asyncio
    async def test_calibrate_confidence(self, agent: ConcreteAgent) -> None:
        confidence = await agent.calibrate_confidence(
            "Test claim",
            ["evidence 1", "evidence 2"],
        )
        assert 0.0 <= confidence <= 1.0
        assert len(agent._confidence_history) == 1

    @pytest.mark.asyncio
    async def test_calibrate_confidence_clamps(self, agent: ConcreteAgent) -> None:
        agent.llm.generate_structured = AsyncMock(
            return_value={"confidence": 1.5, "reasoning": "test"}
        )
        confidence = await agent.calibrate_confidence("claim", ["ev"])
        assert confidence == 1.0

        agent.llm.generate_structured = AsyncMock(
            return_value={"confidence": -0.5, "reasoning": "test"}
        )
        confidence = await agent.calibrate_confidence("claim", ["ev"])
        assert confidence == 0.0

    def test_mean_confidence_empty(self, agent: ConcreteAgent) -> None:
        assert agent.mean_confidence() == 0.5

    def test_mean_confidence(self, agent: ConcreteAgent) -> None:
        agent._confidence_history = [0.6, 0.8, 1.0]
        assert abs(agent.mean_confidence() - 0.8) < 1e-9


class TestLLMHelpers:
    @pytest.mark.asyncio
    async def test_ask_llm(self, agent: ConcreteAgent) -> None:
        result = await agent._ask_llm("prompt", system="sys")
        assert result == "test"
        agent.llm.generate.assert_called_once_with(
            "prompt", system="sys", temperature=None
        )

    @pytest.mark.asyncio
    async def test_ask_llm_structured(self, agent: ConcreteAgent) -> None:
        result = await agent._ask_llm_structured("prompt", system="sys")
        assert isinstance(result, dict)
        agent.llm.generate_structured.assert_called_once_with(
            "prompt", system="sys", temperature=None
        )


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_run_executes_task(self, agent: ConcreteAgent) -> None:
        task = AgentMessage(
            sender=AgentRole.ORCHESTRATOR,
            receiver=AgentRole.LITERATURE_ANALYST,
            task_type="test_task",
            payload={},
            message_id="msg-1",
        )
        await agent.deliver_message(task)

        # Run agent loop with a timeout to avoid infinite loop
        async def run_with_timeout() -> None:
            agent.set_state(AgentState.IDLE)
            msg = await agent.receive_message(timeout=1.0)
            if msg:
                agent.set_state(AgentState.WORKING)
                result = await agent.execute(msg)
                await agent.send_message(result)
                agent.set_state(AgentState.IDLE)

        await asyncio.wait_for(run_with_timeout(), timeout=5.0)
        outbox_msg = await agent._outbox.get()
        assert outbox_msg.task_type == "result"
        assert outbox_msg.in_reply_to == "msg-1"

    @pytest.mark.asyncio
    async def test_run_handles_error(self, agent: ConcreteAgent) -> None:
        class ErrorAgent(BaseAgent):
            role = AgentRole.CRITIC

            async def execute(self, task: AgentMessage) -> AgentMessage:
                raise RuntimeError("test error")

        error_agent = ErrorAgent(agent.config, agent.llm)
        task = AgentMessage(
            sender=AgentRole.ORCHESTRATOR,
            receiver=AgentRole.CRITIC,
            task_type="fail_task",
            payload={},
            message_id="msg-2",
        )
        await error_agent.deliver_message(task)

        # Simulate one iteration of the run loop
        msg = await error_agent.receive_message(timeout=1.0)
        assert msg is not None
        error_agent.set_state(AgentState.WORKING)
        try:
            await error_agent.execute(msg)
        except RuntimeError:
            error_agent.set_state(AgentState.ERROR)
            error_msg = error_agent.create_message(
                receiver=msg.sender,
                task_type="error",
                payload={"error": "test error"},
                in_reply_to=msg.message_id,
            )
            await error_agent.send_message(error_msg)

        assert error_agent.state == AgentState.ERROR
        outbox_msg = await error_agent._outbox.get()
        assert outbox_msg.task_type == "error"
