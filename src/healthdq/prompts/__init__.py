"""
Prompt generation module for healthdq-ai
Author: Agate Jarmakoviča

Provides both programmatic prompt generation and LLM prompt templates for
healthcare data quality analysis, including FHIR, schema learning, and
semantic analysis prompts.
"""

from pathlib import Path
from typing import Optional

from .base_prompt import BasePromptGenerator
from .prompt_templates import (
    COORDINATOR_SYSTEM_PROMPT,
    PRECISION_SYSTEM_PROMPT,
    COMPLETENESS_SYSTEM_PROMPT,
    REUSABILITY_SYSTEM_PROMPT,
    get_prompt_template,
)


def load_prompt(prompt_name: str, version: Optional[str] = None) -> str:
    """
    Load a prompt template from file.

    Args:
        prompt_name: Name of the prompt (without .md extension)
                    Available: fhir_analysis, schema_learning, semantic_analysis
        version: Specific version to load (e.g., "v1"), None for latest

    Returns:
        Prompt content as string

    Example:
        >>> prompt = load_prompt("fhir_analysis")
        >>> prompt = load_prompt("schema_learning", version="v1")
    """
    prompts_dir = Path(__file__).parent

    if version:
        filename = f"{prompt_name}.{version}.md"
    else:
        filename = f"{prompt_name}.md"

    prompt_path = prompts_dir / filename

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def list_available_prompts() -> list:
    """
    List all available prompt templates.

    Returns:
        List of prompt names (without .md extension)

    Example:
        >>> prompts = list_available_prompts()
        >>> print(prompts)
        ['fhir_analysis', 'schema_learning', 'semantic_analysis']
    """
    prompts_dir = Path(__file__).parent
    prompts = [
        p.stem for p in prompts_dir.glob("*.md")
        if p.stem != "README" and not p.stem.endswith((".v1", ".v2"))
    ]
    return sorted(prompts)


__all__ = [
    "BasePromptGenerator",
    "COORDINATOR_SYSTEM_PROMPT",
    "PRECISION_SYSTEM_PROMPT",
    "COMPLETENESS_SYSTEM_PROMPT",
    "REUSABILITY_SYSTEM_PROMPT",
    "get_prompt_template",
    "load_prompt",
    "list_available_prompts",
]
