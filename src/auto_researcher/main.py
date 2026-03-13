"""CLI entry point for the autonomous AI research system."""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

import yaml

from auto_researcher.config import ResearchConfig
from auto_researcher.evaluation.iwpg import IWPGScorer
from auto_researcher.evaluation.peer_review import SimulatedPeerReview
from auto_researcher.evaluation.sunfire import SUNFIREEvaluator
from auto_researcher.orchestrator.orchestrator import ResearchOrchestrator
from auto_researcher.orchestrator.resource_manager import ResourceManager
from auto_researcher.orchestrator.task_router import TaskRouter
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def load_config(config_path: str | None = None) -> ResearchConfig:
    """Load configuration from YAML file or environment defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return ResearchConfig(**data)
    return ResearchConfig()


def build_system(config: ResearchConfig) -> ResearchOrchestrator:
    """Initialize all components and wire them together."""
    llm = LLMClient(config.llm)
    task_router = TaskRouter()
    resource_manager = ResourceManager(config.orchestrator)
    sunfire = SUNFIREEvaluator(llm, config.sunfire)
    iwpg = IWPGScorer(llm, config.iwpg)
    peer_review = SimulatedPeerReview(llm, config.peer_review)

    # Register rate limits for external services
    resource_manager.register_rate_limit("arxiv", max_requests=3, window_seconds=1.0)
    resource_manager.register_rate_limit("llm", max_requests=10, window_seconds=1.0)

    orchestrator = ResearchOrchestrator(
        config=config,
        llm=llm,
        task_router=task_router,
        resource_manager=resource_manager,
        sunfire=sunfire,
        iwpg=iwpg,
        peer_review=peer_review,
    )

    return orchestrator


async def run(config_path: str | None = None) -> None:
    """Main async entry point."""
    config = load_config(config_path)

    # Ensure data and log directories exist
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging("INFO")
    logger.info("system_starting", config_path=config_path)

    orchestrator = build_system(config)

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Start orchestrator in background
    orchestrator_task = asyncio.create_task(orchestrator.start())

    # Wait for shutdown signal
    await shutdown_event.wait()
    logger.info("initiating_shutdown")

    await orchestrator.stop()
    orchestrator_task.cancel()
    try:
        await orchestrator_task
    except asyncio.CancelledError:
        pass

    logger.info("system_stopped")


def main() -> None:
    """CLI entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(config_path))


if __name__ == "__main__":
    main()
