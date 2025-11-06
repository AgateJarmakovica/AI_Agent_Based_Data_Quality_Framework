#!/usr/bin/env python3
"""
Setup script for healthdq-ai package
Author: Agate Jarmakoviča
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("=")
        ]

# Read dev requirements
dev_requirements_file = Path(__file__).parent / "requirements-dev.txt"
dev_requirements = []
if dev_requirements_file.exists():
    with open(dev_requirements_file, "r", encoding="utf-8") as f:
        dev_requirements = [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("=")
            and not line.startswith("-r")
        ]

setup(
    name="healthdq-ai",
    version="2.0.0",
    author="Agate Jarmakoviča",
    author_email="",
    description="AI Agent-Based Data Quality Framework for Healthcare Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework",
    project_urls={
        "Bug Tracker": "https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework/issues",
        "Documentation": "https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework/docs",
        "Source Code": "https://github.com/AgateJarmakovica/AI_Agent_Based_Data_Quality_Framework",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=8.1.1",
            "pytest-cov>=5.0.0",
            "pytest-asyncio>=0.23.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "healthdq=healthdq.cli:main",
            "healthdq-api=healthdq.api.server:run",
            "healthdq-ui=healthdq.ui.streamlit_app:run",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "data-quality",
        "healthcare",
        "ai-agents",
        "machine-learning",
        "data-centric-ai",
        "fair-principles",
        "human-in-the-loop",
        "langchain",
        "llm",
    ],
)
