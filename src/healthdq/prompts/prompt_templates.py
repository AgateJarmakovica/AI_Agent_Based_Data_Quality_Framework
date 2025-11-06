"""
Prompt Templates Library
Author: Agate Jarmakoviča

Kolekcija ar gataviem prompt templates.
"""

from typing import Dict, Any
import pandas as pd


# System Prompts
COORDINATOR_SYSTEM_PROMPT = """
You are a Coordinator Agent in a multi-agent data quality system.

Your role is to:
1. Orchestrate analysis across specialized agents (Precision, Completeness, Reusability)
2. Aggregate and synthesize results from multiple agents
3. Resolve conflicts between agent recommendations
4. Generate comprehensive quality reports
5. Prioritize issues by severity and impact

Always consider the business context and data domain when making decisions.
"""

PRECISION_SYSTEM_PROMPT = """
You are a Precision Agent specializing in data accuracy and correctness.

Your focus areas:
1. Data type consistency and validation
2. Format standardization and compliance
3. Outlier detection and anomaly identification
4. Pattern matching and validation
5. Value range verification

Provide specific, actionable recommendations for improving data precision.
"""

COMPLETENESS_SYSTEM_PROMPT = """
You are a Completeness Agent specializing in missing data analysis.

Your focus areas:
1. Identify missing values across all columns
2. Analyze missing data patterns (MCAR, MAR, MNAR)
3. Recommend appropriate imputation methods
4. Assess impact of missing data on analysis
5. Prioritize columns for completion

Consider the domain context when suggesting imputation strategies.
"""

REUSABILITY_SYSTEM_PROMPT = """
You are a Reusability Agent specializing in FAIR principles and data standardization.

Your focus areas:
1. Assess metadata completeness and quality
2. Check naming convention compliance
3. Evaluate documentation adequacy
4. Verify standard format adherence (FHIR, etc.)
5. Ensure data is findable, accessible, interoperable, and reusable

Promote best practices for data sharing and reusability.
"""


# Analysis Prompts
PRECISION_ANALYSIS_TEMPLATE = """
Analyze the following dataset for precision and accuracy issues:

Dataset Information:
- Name: {dataset_name}
- Rows: {total_rows}
- Columns: {column_list}

Sample Data:
{sample_data}

Please identify:
1. Data type inconsistencies (e.g., mixed types in a column)
2. Format violations (e.g., inconsistent date formats)
3. Outliers (values beyond {outlier_threshold} IQR)
4. Invalid patterns (e.g., malformed emails, phone numbers)
5. Range violations (values outside expected ranges)

For each issue found, provide:
- Issue type
- Affected column(s)
- Severity (critical/high/medium/low)
- Number of affected rows
- Suggested fix with specific method
- Confidence score (0-1)

Focus on actionable issues that can be automatically corrected.
"""

COMPLETENESS_ANALYSIS_TEMPLATE = """
Analyze missing data in the following dataset:

Dataset: {dataset_name}
Total rows: {total_rows}

Missing Data Summary by Column:
{missing_summary}

Column Data Types:
{column_types}

For each column with missing data (>5%), determine:

1. Missing percentage and count
2. Missing data pattern:
   - MCAR (Missing Completely At Random)
   - MAR (Missing At Random)
   - MNAR (Missing Not At Random)

3. Recommended imputation method:
   - Mean/Median/Mode (for simple cases)
   - Forward/Backward fill (for time series)
   - ML prediction (for complex patterns)
   - Domain-specific rules
   - Leave as-is (if missingness is informative)

4. Potential risks:
   - Bias introduction
   - Variance changes
   - Relationship distortion

5. Business importance score (1-10)

Prioritize columns by: (importance × missing_pct × imputation_feasibility)

Provide specific imputation recommendations for top 5 columns.
"""

REUSABILITY_ANALYSIS_TEMPLATE = """
Assess the reusability and FAIR compliance of this dataset:

Dataset: {dataset_name}
Columns: {columns}
Current metadata: {metadata}

Evaluate against FAIR principles:

**F (Findable):**
1. Does it have descriptive metadata? (title, description, keywords)
2. Does it have persistent identifiers? (DOI, URN)
3. Is it registered in searchable resources?
4. Does metadata include data identifier?

**A (Accessible):**
1. Is it retrievable via standard protocol? (HTTP, FTP)
2. Is the format clearly documented?
3. Are access restrictions clear?
4. Is metadata accessible even if data is not?

**I (Interoperable):**
1. Does it use standard formats? (CSV, JSON, FHIR)
2. Does it use common vocabularies? (SNOMED, ICD-10)
3. Does it have clear schemas?
4. Does it reference other datasets appropriately?

**R (Reusable):**
1. Does it have clear licensing? (MIT, CC-BY)
2. Is provenance documented? (source, transformations)
3. Are quality metrics provided?
4. Does it meet community standards?

For each dimension, provide:
- Score (0-1)
- Issues found
- Specific recommendations
- Implementation priority

Generate actionable steps to improve FAIR compliance.
"""


# Feedback Prompts
HUMAN_FEEDBACK_REQUEST_TEMPLATE = """
The AI system has identified the following issue and proposes a solution.
Your review is needed:

**Issue Detected:**
{issue_description}

**Affected Data:**
- Column: {column_name}
- Rows affected: {affected_rows}
- Example values: {example_values}

**AI Proposed Solution:**
Method: {proposed_method}
Action: {proposed_action}
Expected outcome: {expected_outcome}

**AI Confidence:** {confidence_score} (0-1)

**Your Decision:**
Please review and decide:
1. ✅ APPROVE - Solution is appropriate
2. ❌ REJECT - Solution is not appropriate
3. ✏️ MODIFY - Suggest alternative approach

If rejecting or modifying, please provide:
- Reason for your decision
- Alternative approach (if applicable)
- Domain-specific considerations
- Your confidence in this decision (0-1)

Your feedback helps the AI learn and improve!
"""


# Collaboration Prompts
COLLABORATION_REQUEST_TEMPLATE = """
Agent {sender_agent} requests your collaboration on a task:

**Task Description:**
{task_description}

**Required Capabilities:**
{required_capabilities}

**Data Reference:** {data_reference}
**Deadline:** {deadline}

**Your Participation:**
Can you contribute to this task?

If YES, please provide:
1. Your relevant capabilities for this task
2. Estimated time needed (in minutes)
3. Any constraints or requirements
4. Your confidence in successful completion (0-1)
5. Specific contributions you can make

If NO, please provide:
1. Reason for not participating
2. Alternative suggestions (if any)
3. Recommended other agents

Your response will help coordinate the multi-agent workflow.
"""


# Rule Generation Prompts
RULE_GENERATION_TEMPLATE = """
Based on observed data patterns, generate a data quality rule:

**Observed Pattern:**
{pattern_description}

**Pattern Frequency:** {pattern_frequency} occurrences
**Data Context:** {data_context}

**Similar Approved Rules:**
{similar_rules}

**Task:**
Generate a rule specification in the following format:

```yaml
rule_name: descriptive_name_in_snake_case
rule_type: validation | transformation | imputation
description: Clear description of what this rule does
condition:
  when: When should this rule apply?
  columns: [list of columns]
  constraints: Specific conditions
action:
  type: Action to take (validate, transform, impute, flag)
  method: Specific method to use
  parameters: Required parameters
severity: critical | high | medium | low
confidence: 0.0-1.0
rationale: Why is this rule needed?
examples:
  - input: Example input
    output: Expected output
  - input: Another example
    output: Expected output
requires_human_approval: true | false
```

Ensure the rule is:
1. Specific and testable
2. Generalizable to similar cases
3. Non-destructive (preserves data when possible)
4. Well-documented with examples
"""


# Reporting Prompts
EXECUTIVE_SUMMARY_TEMPLATE = """
Generate an executive summary of data quality assessment:

**Dataset:** {dataset_name}
**Records Analyzed:** {record_count:,}
**Analysis Date:** {analysis_date}
**Analysis Duration:** {analysis_duration}

**Quality Scores:**
{quality_scores}

**Key Findings:**
{key_findings}

**Critical Issues:**
{critical_issues}

**Task:**
Create a concise executive summary (200-300 words) that:

1. **Overall Status** (2-3 sentences)
   - Current data quality level
   - Fitness for intended use
   - Risk assessment

2. **Critical Issues** (3-5 bullet points)
   - Most urgent problems requiring immediate attention
   - Potential business impact
   - Estimated scope

3. **Recommended Actions** (Priority order)
   - Top 3 actions to take
   - Expected impact of each
   - Resource requirements

4. **Timeline & Resources**
   - Estimated time to resolve
   - Required expertise
   - Budget implications (if applicable)

5. **Expected Outcomes**
   - Quality improvement percentage
   - Business benefits
   - Risk reduction

Use business-friendly language. Avoid technical jargon.
Focus on actionable insights and ROI.
"""


def get_prompt_template(template_name: str) -> str:
    """
    Get a prompt template by name.

    Args:
        template_name: Name of the template

    Returns:
        Template string
    """
    templates = {
        "coordinator_system": COORDINATOR_SYSTEM_PROMPT,
        "precision_system": PRECISION_SYSTEM_PROMPT,
        "completeness_system": COMPLETENESS_SYSTEM_PROMPT,
        "reusability_system": REUSABILITY_SYSTEM_PROMPT,
        "precision_analysis": PRECISION_ANALYSIS_TEMPLATE,
        "completeness_analysis": COMPLETENESS_ANALYSIS_TEMPLATE,
        "reusability_analysis": REUSABILITY_ANALYSIS_TEMPLATE,
        "feedback_request": HUMAN_FEEDBACK_REQUEST_TEMPLATE,
        "collaboration_request": COLLABORATION_REQUEST_TEMPLATE,
        "rule_generation": RULE_GENERATION_TEMPLATE,
        "executive_summary": EXECUTIVE_SUMMARY_TEMPLATE,
    }

    return templates.get(template_name, "")


__all__ = [
    "COORDINATOR_SYSTEM_PROMPT",
    "PRECISION_SYSTEM_PROMPT",
    "COMPLETENESS_SYSTEM_PROMPT",
    "REUSABILITY_SYSTEM_PROMPT",
    "PRECISION_ANALYSIS_TEMPLATE",
    "COMPLETENESS_ANALYSIS_TEMPLATE",
    "REUSABILITY_ANALYSIS_TEMPLATE",
    "HUMAN_FEEDBACK_REQUEST_TEMPLATE",
    "COLLABORATION_REQUEST_TEMPLATE",
    "RULE_GENERATION_TEMPLATE",
    "EXECUTIVE_SUMMARY_TEMPLATE",
    "get_prompt_template",
]
