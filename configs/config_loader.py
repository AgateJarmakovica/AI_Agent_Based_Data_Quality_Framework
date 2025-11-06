"""
config_loader.py
----------------
Configuration loader and validator for healthdq-ai v2.2

Loads YAML-based configs:
- rules.yml  → Data quality rules and FAIR/FHIR policies
- agents.yml → Multi-agent configuration and behavior
- prompts.yml → LLM and agent communication templates

Author: Agate Jarmakoviča
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# Pydantic models for config validation
# ============================================================

class Rule(BaseModel):
    description: str
    severity: str = Field(default="medium", regex="^(low|medium|high|critical)$")
    action: str
    enabled: bool = True
    threshold: float | None = None


class RulesConfig(BaseModel):
    rules: Dict[str, Dict[str, Rule]]
    settings: Dict[str, Any] | None = None


class AgentConfig(BaseModel):
    name: str
    type: str
    role: str
    enabled: bool = True
    model: str | None = None
    temperature: float | None = None
    tools: list[str] | None = None


class AgentsConfig(BaseModel):
    agents: list[AgentConfig]


class PromptConfig(BaseModel):
    name: str
    template: str
    description: str | None = None


class PromptsConfig(BaseModel):
    prompts: list[PromptConfig]


# ============================================================
# Loader utilities
# ============================================================

def _load_yaml(file_path: Path) -> dict:
    """Safely load YAML file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rules_config(base_dir: str = "configs") -> RulesConfig:
    """Load and validate data quality rules."""
    path = Path(base_dir) / "rules.yml"
    data = _load_yaml(path)
    try:
        return RulesConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid rules.yml configuration: {e}")


def load_agents_config(base_dir: str = "configs") -> AgentsConfig:
    """Load and validate agent definitions."""
    path = Path(base_dir) / "agents.yml"
    data = _load_yaml(path)
    try:
        return AgentsConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid agents.yml configuration: {e}")


def load_prompts_config(base_dir: str = "configs") -> PromptsConfig:
    """Load and validate prompt templates."""
    path = Path(base_dir) / "prompts.yml"
    data = _load_yaml(path)
    try:
        return PromptsConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid prompts.yml configuration: {e}")


# ============================================================
# High-level helper
# ============================================================

def load_all_configs(base_dir: str = "configs") -> dict:
    """
    Load all configurations (rules, agents, prompts)
    into a unified dictionary structure.
    """
    return {
        "rules": load_rules_config(base_dir),
        "agents": load_agents_config(base_dir),
        "prompts": load_prompts_config(base_dir),
    }


# ============================================================
# Example usage (manual test)
# ============================================================
if __name__ == "__main__":
    configs = load_all_configs()
    print("✅ Rules loaded:", len(configs["rules"].rules))
    print("✅ Agents loaded:", len(configs["agents"].agents))
    print("✅ Prompts loaded:", len(configs["prompts"].prompts))
