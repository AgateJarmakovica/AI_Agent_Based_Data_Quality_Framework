"""
Base Prompt Generator for AI Agents
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path


class BasePromptGenerator:
    """
    Bāzes klase promptu ģenerēšanai AI aģentiem.

    Ielādē promptu templates no YAML un aizvieto placeholders.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize prompt generator.

        Args:
            config_path: Path to prompts.yml config file
        """
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "prompts.yml"

        self.config_path = Path(config_path)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Load prompt templates from YAML."""
        if not self.config_path.exists():
            # Return empty templates if config doesn't exist
            return {}

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config or {}

    def get_system_prompt(self, agent_type: str) -> str:
        """
        Get system prompt for specific agent type.

        Args:
            agent_type: Type of agent (coordinator, precision, completeness, reusability)

        Returns:
            System prompt string
        """
        system_prompts = self.templates.get("system_prompts", {})
        agent_config = system_prompts.get(agent_type, {})

        role = agent_config.get("role", "You are a data quality agent.")
        instructions = agent_config.get("instructions", "")

        prompt = f"{role}\n\n"
        if instructions:
            prompt += f"Instructions:\n{instructions}\n"

        return prompt

    def generate_analysis_prompt(
        self,
        prompt_type: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generate analysis prompt from template.

        Args:
            prompt_type: Type of analysis (precision_analysis, completeness_analysis, etc.)
            variables: Variables to fill in template

        Returns:
            Generated prompt
        """
        analysis_prompts = self.templates.get("analysis_prompts", {})
        template = analysis_prompts.get(prompt_type, {}).get("template", "")

        if not template:
            return "Analyze the following data for quality issues."

        # Replace variables
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

    def generate_feedback_prompt(
        self,
        prompt_type: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generate feedback-related prompt.

        Args:
            prompt_type: Type of feedback prompt
            variables: Variables to fill in

        Returns:
            Generated prompt
        """
        feedback_prompts = self.templates.get("feedback_prompts", {})
        template = feedback_prompts.get(prompt_type, {}).get("template", "")

        if not template:
            return "Please provide feedback on this action."

        # Replace variables
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

    def generate_collaboration_prompt(
        self,
        prompt_type: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generate collaboration prompt for multi-agent communication.

        Args:
            prompt_type: Type of collaboration prompt
            variables: Variables to fill in

        Returns:
            Generated prompt
        """
        collab_prompts = self.templates.get("collaboration_prompts", {})
        template = collab_prompts.get(prompt_type, {}).get("template", "")

        if not template:
            return "Please collaborate on this task."

        # Replace variables
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

    def generate_rule_generation_prompt(
        self,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for automatic rule generation.

        Args:
            variables: Variables including pattern description, context, etc.

        Returns:
            Generated prompt
        """
        rule_prompts = self.templates.get("rule_generation_prompts", {})
        template = rule_prompts.get("generate_rule", {}).get("template", "")

        if not template:
            return "Generate a data quality rule based on the observed pattern."

        # Replace variables
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

    def generate_reporting_prompt(
        self,
        report_type: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generate reporting prompt.

        Args:
            report_type: Type of report (executive_summary, technical_report)
            variables: Variables to fill in

        Returns:
            Generated prompt
        """
        reporting_prompts = self.templates.get("reporting_prompts", {})
        template = reporting_prompts.get(report_type, {}).get("template", "")

        if not template:
            return "Generate a report on data quality."

        # Replace variables
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt

    def create_custom_prompt(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Create a custom prompt from a template string.

        Args:
            template: Template string with {variable} placeholders
            variables: Variables to fill in

        Returns:
            Generated prompt
        """
        prompt = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt


__all__ = ["BasePromptGenerator"]
