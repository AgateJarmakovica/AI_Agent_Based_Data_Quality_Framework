"""
Prompt generation module for healthdq-ai
Author: Agate Jarmakoviča
"""

from .base_prompt import BasePromptGenerator
from .prompt_templates import (
    COORDINATOR_SYSTEM_PROMPT,
    PRECISION_SYSTEM_PROMPT,
    COMPLETENESS_SYSTEM_PROMPT,
    REUSABILITY_SYSTEM_PROMPT,
    get_prompt_template,
)

__all__ = [
    "BasePromptGenerator",
    "COORDINATOR_SYSTEM_PROMPT",
    "PRECISION_SYSTEM_PROMPT",
    "COMPLETENESS_SYSTEM_PROMPT",
    "REUSABILITY_SYSTEM_PROMPT",
    "get_prompt_template",
]
