# CLAUDE.md — Project Instructions for AI Assistants

## Project Overview

picoclaw-auto-ai-researcher is an autonomous AI research system that directs agent teams to explore the AI research space, identify gaps, and produce novel contributions. The goal is a self-improving AI research loop.

## Repository Layout

- `docs/` — 12 research documents covering foundations, architecture, algorithms, costs, and analysis
- `experiments/` — Active experiments with hypotheses, methodology, and measurement plans
- `experiments-completed/` — Completed experiments with full write-ups (paper format with citations, graphs, reproduction instructions)
- `src/auto_researcher/` — Python implementation (agents, ingestion, infrastructure, evaluation, orchestrator, learning, verification, models)
- `config.example.yaml` — System configuration reference

## Critical Instructions

### Experiment Lifecycle

1. **Before starting any experiment:** Read the experiment's markdown file in `experiments/` to understand the hypothesis, methodology, metrics, and success criteria.

2. **During experiments:** Update the experiment file with:
   - Status changes (planned → in-progress → completed/failed)
   - Intermediate results and observations
   - Any methodology adjustments and why they were made
   - Links to code, data, and outputs produced

3. **On experiment completion:** Move the experiment file to `experiments-completed/` and replace it with a full paper-format write-up including:
   - Abstract, introduction, methodology, results, discussion, conclusion
   - All statistical tests with effect sizes and confidence intervals
   - Graphs/visualizations of key results
   - Full reproduction instructions (commands, data, environment)
   - Citations for all referenced work
   - What was learned and how it affects downstream experiments

4. **After each experiment:** Update `experiments/README.md` with:
   - Results summary and key findings
   - Which downstream experiments are affected and how
   - Any new experiments suggested by the findings
   - Updated priority ordering if findings change assumptions

### Documentation Updates

- **Keep docs/ current:** As experiments produce results, update the relevant research documents in `docs/` with empirical findings. Theory should be revised when evidence contradicts it.
- **Cross-reference:** When an experiment validates or invalidates a claim in a doc, add a note in the doc pointing to the experiment write-up.
- **New research:** When you discover relevant new papers or techniques during experiments, add them to the appropriate doc with citations.

### Experiment Design Principles

- Every experiment starts with a falsifiable hypothesis
- Use the simplest design that tests the hypothesis — avoid over-engineering
- Always define success/failure criteria before running
- Use appropriate statistical tests (see individual experiment files for specifics)
- Record negative results — they're as informative as positive ones
- Each experiment should produce data that informs at least one downstream experiment

### Code Standards

- Python 3.11+, use existing project structure under `src/auto_researcher/`
- Dependencies managed via `pyproject.toml` (use `uv` for package management)
- Type hints required, `mypy --strict` must pass
- Format with `ruff`, test with `pytest`
- Experiment scripts go in `experiments/<experiment-id>/` subdirectories

### Commit Practices

- Do not include Co-Authored-By lines for Claude
- Commit and push after completing meaningful units of work
- Use descriptive commit messages that reference experiment IDs where applicable
