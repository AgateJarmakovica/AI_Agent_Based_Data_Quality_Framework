"""
Human-in-the-Loop Approval System
Author: Agate Jarmakoviča

Cilvēks apstiprina vai noraida AI ieteiktos uzlabojumus.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
from enum import Enum

from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class ApprovalStatus(str, Enum):
    """Apstiprināšanas statusi."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PARTIALLY_APPROVED = "partially_approved"
    NEEDS_MODIFICATION = "needs_modification"


class ApprovalDecision(str, Enum):
    """Lēmumu tipi."""
    APPROVE_ALL = "approve_all"
    REJECT_ALL = "reject_all"
    SELECTIVE = "selective"


class ApprovalManager:
    """
    Pārvalda apstiprināšanas procesu un lēmumus.
    """

    def __init__(self):
        """Initialize approval manager."""
        self.approvals: Dict[str, Dict[str, Any]] = {}
        self.approval_history: List[Dict[str, Any]] = []

    def create_approval_request(
        self,
        session_id: str,
        proposed_changes: List[Dict[str, Any]],
        reviewer_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Izveidot apstiprināšanas pieprasījumu.

        Args:
            session_id: Review session ID
            proposed_changes: Ieteiktas izmaiņas
            reviewer_info: Recenzenta informācija

        Returns:
            Approval request
        """
        approval_id = str(uuid.uuid4())

        approval_request = {
            "approval_id": approval_id,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": ApprovalStatus.PENDING,

            # Izmaiņas, kas jāapstiprina
            "proposed_changes": proposed_changes,
            "total_changes": len(proposed_changes),

            # Apstiprināšanas statuss katrai izmaiņai
            "change_approvals": {
                i: {
                    "approved": None,
                    "comment": None,
                    "modified": False,
                    "modification": None,
                }
                for i in range(len(proposed_changes))
            },

            # Recenzents
            "reviewer": reviewer_info or {},
            "reviewed_at": None,

            # Galīgais lēmums
            "final_decision": None,
            "decision_rationale": None,

            # Statistika
            "approved_count": 0,
            "rejected_count": 0,
            "modified_count": 0,
        }

        self.approvals[approval_id] = approval_request
        logger.info(f"Approval request created: {approval_id} for session {session_id}")

        return approval_request

    def approve_change(
        self,
        approval_id: str,
        change_index: int,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Apstiprināt konkrētu izmaiņu.

        Args:
            approval_id: Approval ID
            change_index: Izmaiņas indekss
            comment: Komentārs

        Returns:
            Success status
        """
        if approval_id not in self.approvals:
            logger.error(f"Approval not found: {approval_id}")
            return False

        approval = self.approvals[approval_id]

        if change_index not in approval["change_approvals"]:
            logger.error(f"Invalid change index: {change_index}")
            return False

        # Mark as approved
        approval["change_approvals"][change_index]["approved"] = True
        approval["change_approvals"][change_index]["comment"] = comment

        # Update counts
        approval["approved_count"] = sum(
            1 for ca in approval["change_approvals"].values() if ca["approved"] is True
        )

        logger.info(f"Change {change_index} approved in approval {approval_id}")
        return True

    def reject_change(
        self,
        approval_id: str,
        change_index: int,
        reason: str,
    ) -> bool:
        """
        Noraidīt konkrētu izmaiņu.

        Args:
            approval_id: Approval ID
            change_index: Izmaiņas indekss
            reason: Noraidīšanas iemesls

        Returns:
            Success status
        """
        if approval_id not in self.approvals:
            logger.error(f"Approval not found: {approval_id}")
            return False

        approval = self.approvals[approval_id]

        if change_index not in approval["change_approvals"]:
            logger.error(f"Invalid change index: {change_index}")
            return False

        # Mark as rejected
        approval["change_approvals"][change_index]["approved"] = False
        approval["change_approvals"][change_index]["comment"] = reason

        # Update counts
        approval["rejected_count"] = sum(
            1 for ca in approval["change_approvals"].values() if ca["approved"] is False
        )

        logger.info(f"Change {change_index} rejected in approval {approval_id}")
        return True

    def modify_change(
        self,
        approval_id: str,
        change_index: int,
        modification: Dict[str, Any],
        comment: Optional[str] = None,
    ) -> bool:
        """
        Modificēt ieteikto izmaiņu.

        Args:
            approval_id: Approval ID
            change_index: Izmaiņas indekss
            modification: Modificētā versija
            comment: Komentārs

        Returns:
            Success status
        """
        if approval_id not in self.approvals:
            logger.error(f"Approval not found: {approval_id}")
            return False

        approval = self.approvals[approval_id]

        if change_index not in approval["change_approvals"]:
            logger.error(f"Invalid change index: {change_index}")
            return False

        # Mark as modified and approved
        approval["change_approvals"][change_index]["approved"] = True
        approval["change_approvals"][change_index]["modified"] = True
        approval["change_approvals"][change_index]["modification"] = modification
        approval["change_approvals"][change_index]["comment"] = comment

        # Update counts
        approval["modified_count"] = sum(
            1 for ca in approval["change_approvals"].values() if ca["modified"] is True
        )

        logger.info(f"Change {change_index} modified in approval {approval_id}")
        return True

    def bulk_approve(
        self,
        approval_id: str,
        change_indices: List[int],
        comment: Optional[str] = None,
    ) -> int:
        """
        Apstiprināt vairākas izmaiņas vienlaicīgi.

        Returns:
            Skaits, cik izmaiņas apstiprinātās
        """
        approved_count = 0

        for index in change_indices:
            if self.approve_change(approval_id, index, comment):
                approved_count += 1

        logger.info(f"Bulk approved {approved_count} changes")
        return approved_count

    def bulk_reject(
        self,
        approval_id: str,
        change_indices: List[int],
        reason: str,
    ) -> int:
        """
        Noraidīt vairākas izmaiņas vienlaicīgi.

        Returns:
            Skaits, cik izmaiņas noraidītas
        """
        rejected_count = 0

        for index in change_indices:
            if self.reject_change(approval_id, index, reason):
                rejected_count += 1

        logger.info(f"Bulk rejected {rejected_count} changes")
        return rejected_count

    def finalize_approval(
        self,
        approval_id: str,
        reviewer_info: Dict[str, Any],
        decision_rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pabeigt apstiprināšanas procesu un pieņemt galīgo lēmumu.

        Args:
            approval_id: Approval ID
            reviewer_info: Recenzenta informācija
            decision_rationale: Lēmuma pamatojums

        Returns:
            Final decision
        """
        if approval_id not in self.approvals:
            raise ValueError(f"Approval not found: {approval_id}")

        approval = self.approvals[approval_id]

        # Determine final status
        total = approval["total_changes"]
        approved = approval["approved_count"]
        rejected = approval["rejected_count"]

        if approved == total:
            final_status = ApprovalStatus.APPROVED
            final_decision = ApprovalDecision.APPROVE_ALL
        elif rejected == total:
            final_status = ApprovalStatus.REJECTED
            final_decision = ApprovalDecision.REJECT_ALL
        elif approved > 0:
            final_status = ApprovalStatus.PARTIALLY_APPROVED
            final_decision = ApprovalDecision.SELECTIVE
        else:
            final_status = ApprovalStatus.PENDING
            final_decision = None

        # Update approval
        approval["status"] = final_status
        approval["final_decision"] = final_decision
        approval["decision_rationale"] = decision_rationale
        approval["reviewer"] = reviewer_info
        approval["reviewed_at"] = datetime.now().isoformat()

        # Add to history
        self.approval_history.append({
            "approval_id": approval_id,
            "session_id": approval["session_id"],
            "final_status": final_status,
            "final_decision": final_decision,
            "reviewed_at": approval["reviewed_at"],
            "reviewer": reviewer_info,
            "statistics": {
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "modified": approval["modified_count"],
            },
        })

        logger.info(f"Approval finalized: {approval_id} with status {final_status}")

        return {
            "approval_id": approval_id,
            "status": final_status,
            "decision": final_decision,
            "approved_changes": self.get_approved_changes(approval_id),
            "rejected_changes": self.get_rejected_changes(approval_id),
            "statistics": {
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "modified": approval["modified_count"],
            },
        }

    def get_approved_changes(self, approval_id: str) -> List[Dict[str, Any]]:
        """Iegūt visas apstiprinātās izmaiņas."""
        if approval_id not in self.approvals:
            return []

        approval = self.approvals[approval_id]
        approved = []

        for index, change_approval in approval["change_approvals"].items():
            if change_approval["approved"] is True:
                change = approval["proposed_changes"][index].copy()

                # Ja modificēts, izmanto modificēto versiju
                if change_approval["modified"]:
                    change.update(change_approval["modification"])

                change["approval_comment"] = change_approval["comment"]
                approved.append(change)

        return approved

    def get_rejected_changes(self, approval_id: str) -> List[Dict[str, Any]]:
        """Iegūt visas noraidītās izmaiņas."""
        if approval_id not in self.approvals:
            return []

        approval = self.approvals[approval_id]
        rejected = []

        for index, change_approval in approval["change_approvals"].items():
            if change_approval["approved"] is False:
                change = approval["proposed_changes"][index].copy()
                change["rejection_reason"] = change_approval["comment"]
                rejected.append(change)

        return rejected

    def get_approval_summary(self, approval_id: str) -> Dict[str, Any]:
        """Iegūt apstiprināšanas kopsavilkumu."""
        if approval_id not in self.approvals:
            raise ValueError(f"Approval not found: {approval_id}")

        approval = self.approvals[approval_id]

        return {
            "approval_id": approval_id,
            "session_id": approval["session_id"],
            "status": approval["status"],
            "created_at": approval["created_at"],
            "reviewed_at": approval["reviewed_at"],
            "total_changes": approval["total_changes"],
            "approved_count": approval["approved_count"],
            "rejected_count": approval["rejected_count"],
            "modified_count": approval["modified_count"],
            "pending_count": approval["total_changes"] - approval["approved_count"] - approval["rejected_count"],
            "final_decision": approval["final_decision"],
        }

    def get_approval_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Iegūt apstiprināšanas vēsturi.

        Args:
            session_id: Filtrēt pēc session ID
            limit: Maksimālais ierakstu skaits

        Returns:
            History records
        """
        history = self.approval_history

        if session_id:
            history = [h for h in history if h["session_id"] == session_id]

        # Sort by date (newest first)
        history = sorted(history, key=lambda x: x["reviewed_at"], reverse=True)

        return history[:limit]


__all__ = ["ApprovalManager", "ApprovalStatus", "ApprovalDecision"]
