"""
Human-in-the-Loop Review System
Author: Agate Jarmakoviča

Cilvēks pārskata un novērtē datu kvalitāti PIRMS izmaiņu piemērošanas.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime
import uuid

from healthdq.utils.logger import get_logger
from healthdq.utils.helpers import calculate_data_statistics

logger = get_logger(__name__)


class DataReview:
    """
    Datu kopas pārskatīšana un novērtēšana pirms transformācijas.
    """

    def __init__(self):
        """Initialize review system."""
        self.reviews: Dict[str, Dict[str, Any]] = {}

    def create_review_session(
        self,
        data: pd.DataFrame,
        quality_results: Dict[str, Any],
        improvement_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Izveidot pārskatīšanas sesiju ar pilnu informāciju.

        Args:
            data: Oriģinālie dati
            quality_results: Kvalitātes analīzes rezultāti
            improvement_plan: Uzlabošanas plāns

        Returns:
            Review session ar visu nepieciešamo informāciju
        """
        session_id = str(uuid.uuid4())

        # Aprēķināt statistiku
        statistics = calculate_data_statistics(data)

        # Sagatavot pārskatu
        review_session = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending_review",

            # Datu informācija
            "data_info": {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            },

            # Kvalitātes novērtējums
            "quality_assessment": {
                "overall_score": quality_results.get("overall_score", 0.0),
                "dimension_scores": {
                    dim: results.get("score", 0.0)
                    for dim, results in quality_results.get("dimension_results", {}).items()
                },
                "total_issues": len(improvement_plan.get("actions", [])),
            },

            # Problēmu saraksts
            "issues": self._categorize_issues(improvement_plan),

            # Ieteiktās izmaiņas
            "proposed_changes": self._summarize_changes(improvement_plan),

            # Statistika
            "statistics": statistics,

            # Sample data (pirmās 10 rindas)
            "sample_data": data.head(10).to_dict(orient="records"),

            # Requires human decision
            "requires_approval": True,
            "approved": None,
            "reviewed_by": None,
            "review_comments": None,
        }

        # Saglabāt sesiju
        self.reviews[session_id] = review_session

        logger.info(f"Review session created: {session_id}")
        return review_session

    def _categorize_issues(self, improvement_plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Kategorizēt problēmas pēc svarīguma."""
        issues_by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }

        for action in improvement_plan.get("actions", []):
            severity = action.get("severity", "medium")
            if severity in issues_by_severity:
                issues_by_severity[severity].append({
                    "dimension": action.get("dimension"),
                    "column": action.get("column"),
                    "type": action.get("type"),
                    "description": f"{action.get('type')} in column {action.get('column')}",
                    "recommended_action": action.get("recommended_action"),
                })

        return issues_by_severity

    def _summarize_changes(self, improvement_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apkopot visas ieteiktās izmaiņas."""
        changes = []

        for action in improvement_plan.get("actions", []):
            changes.append({
                "action_type": action.get("recommended_action"),
                "target": action.get("column", "Dataset"),
                "severity": action.get("severity"),
                "description": self._get_action_description(action),
                "estimated_impact": self._estimate_impact(action),
            })

        return changes

    def _get_action_description(self, action: Dict[str, Any]) -> str:
        """Iegūt cilvēkam saprotamu darbības aprakstu."""
        action_type = action.get("recommended_action", "unknown")
        column = action.get("column", "N/A")

        descriptions = {
            "impute_missing_values": f"Aizpildīt trūkstošās vērtības kolonnā '{column}'",
            "handle_outliers": f"Apstrādāt izskaņotās vērtības kolonnā '{column}'",
            "standardize_data_types": f"Standartizēt datu tipus kolonnā '{column}'",
            "normalize_column_names": "Normalizēt kolonnu nosaukumus",
            "add_metadata": "Pievienot metadatus",
            "manual_review": f"Nepieciešama manuāla pārbaude kolonnai '{column}'",
        }

        return descriptions.get(action_type, f"Veikt {action_type} darbību")

    def _estimate_impact(self, action: Dict[str, Any]) -> str:
        """Novērtēt darbības ietekmi."""
        severity = action.get("severity", "medium")

        impact_map = {
            "critical": "Ļoti liela ietekme - ieteicams apstiprināt",
            "high": "Liela ietekme - pārskatīt rūpīgi",
            "medium": "Vidēja ietekme",
            "low": "Zema ietekme",
        }

        return impact_map.get(severity, "Nezināma ietekme")

    def get_review_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Iegūt pārskatīšanas kopsavilkumu.

        Args:
            session_id: Review session ID

        Returns:
            Summary ar galveno informāciju
        """
        if session_id not in self.reviews:
            raise ValueError(f"Review session not found: {session_id}")

        session = self.reviews[session_id]

        summary = {
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"],

            # Datu info
            "dataset_rows": session["data_info"]["shape"][0],
            "dataset_columns": session["data_info"]["shape"][1],

            # Kvalitāte
            "overall_quality": session["quality_assessment"]["overall_score"],
            "total_issues": session["quality_assessment"]["total_issues"],

            # Problēmu sadalījums
            "issues_by_severity": {
                severity: len(issues)
                for severity, issues in session["issues"].items()
            },

            # Statuss
            "requires_approval": session["requires_approval"],
            "approved": session["approved"],
        }

        return summary

    def get_detailed_issues(self, session_id: str, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Iegūt detalizētu problēmu sarakstu.

        Args:
            session_id: Review session ID
            severity: Filtrēt pēc svarīguma (critical, high, medium, low)

        Returns:
            Saraksts ar problēmām
        """
        if session_id not in self.reviews:
            raise ValueError(f"Review session not found: {session_id}")

        session = self.reviews[session_id]

        if severity:
            return session["issues"].get(severity, [])
        else:
            # Atgriest visas problēmas, sakārtoti pēc svarīguma
            all_issues = []
            for sev in ["critical", "high", "medium", "low"]:
                for issue in session["issues"].get(sev, []):
                    issue["severity"] = sev
                    all_issues.append(issue)
            return all_issues

    def get_proposed_changes_preview(self, session_id: str) -> pd.DataFrame:
        """
        Iegūt vizuālu priekšskatījumu par izmaiņām.

        Returns:
            DataFrame ar izmaiņu pārskatu
        """
        if session_id not in self.reviews:
            raise ValueError(f"Review session not found: {session_id}")

        session = self.reviews[session_id]
        changes = session["proposed_changes"]

        preview_data = []
        for i, change in enumerate(changes, 1):
            preview_data.append({
                "#": i,
                "Darbība": change["action_type"],
                "Mērķis": change["target"],
                "Svarīgums": change["severity"],
                "Apraksts": change["description"],
                "Ietekme": change["estimated_impact"],
            })

        return pd.DataFrame(preview_data)


class ImprovementReview:
    """
    Uzlabojumu pārskatīšana - salīdzina oriģinālos un uzlabotus datus.
    """

    def __init__(self):
        """Initialize improvement review."""
        pass

    def create_comparison(
        self,
        original_data: pd.DataFrame,
        improved_data: pd.DataFrame,
        original_metrics: Dict[str, Any],
        improved_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Izveidot salīdzinājumu starp oriģināliem un uzlabotiem datiem.

        Returns:
            Comparison report
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),

            # Shape izmaiņas
            "shape_changes": {
                "original": original_data.shape,
                "improved": improved_data.shape,
                "rows_changed": improved_data.shape[0] - original_data.shape[0],
                "columns_changed": improved_data.shape[1] - original_data.shape[1],
            },

            # Metriku uzlabojumi
            "metric_improvements": self._calculate_improvements(original_metrics, improved_metrics),

            # Kolonnu izmaiņas
            "column_changes": self._compare_columns(original_data, improved_data),

            # Iztrūkstošo vērtību izmaiņas
            "missing_value_changes": self._compare_missing_values(original_data, improved_data),
        }

        return comparison

    def _calculate_improvements(
        self,
        original_metrics: Dict[str, Any],
        improved_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aprēķināt metriku uzlabojumus."""
        improvements = {}

        # Overall score
        original_overall = original_metrics.get("overall_score", 0.0)
        improved_overall = improved_metrics.get("overall_score", 0.0)
        improvements["overall_score"] = {
            "original": original_overall,
            "improved": improved_overall,
            "change": improved_overall - original_overall,
            "improvement_percent": ((improved_overall - original_overall) / max(original_overall, 0.01)) * 100,
        }

        # Dimensiju scores
        for dimension in ["completeness", "accuracy", "consistency", "uniqueness", "validity"]:
            if dimension in original_metrics and dimension in improved_metrics:
                orig_score = original_metrics[dimension].get("overall_score", 0.0)
                imp_score = improved_metrics[dimension].get("overall_score", 0.0)

                improvements[dimension] = {
                    "original": orig_score,
                    "improved": imp_score,
                    "change": imp_score - orig_score,
                    "improvement_percent": ((imp_score - orig_score) / max(orig_score, 0.01)) * 100,
                }

        return improvements

    def _compare_columns(self, original: pd.DataFrame, improved: pd.DataFrame) -> Dict[str, Any]:
        """Salīdzināt kolonnas."""
        original_cols = set(original.columns)
        improved_cols = set(improved.columns)

        return {
            "added_columns": list(improved_cols - original_cols),
            "removed_columns": list(original_cols - improved_cols),
            "renamed_columns": [],  # TODO: implement column matching
        }

    def _compare_missing_values(self, original: pd.DataFrame, improved: pd.DataFrame) -> Dict[str, Any]:
        """Salīdzināt iztrūkstošās vērtības."""
        original_missing = original.isna().sum().sum()
        improved_missing = improved.isna().sum().sum()

        # Per column
        column_comparison = {}
        for col in original.columns:
            if col in improved.columns:
                orig_miss = original[col].isna().sum()
                imp_miss = improved[col].isna().sum()

                if orig_miss != imp_miss:
                    column_comparison[col] = {
                        "original_missing": int(orig_miss),
                        "improved_missing": int(imp_miss),
                        "filled": int(orig_miss - imp_miss),
                    }

        return {
            "total_original": int(original_missing),
            "total_improved": int(improved_missing),
            "total_filled": int(original_missing - improved_missing),
            "by_column": column_comparison,
        }


__all__ = ["DataReview", "ImprovementReview"]
