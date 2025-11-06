"""
Message router for intelligent message routing between agents
Author: Agate Jarmakoviča
"""

from typing import Any, Dict, List, Optional
import asyncio

from healthdq.communication.protocol import get_protocol
from healthdq.communication.message import (
    AgentMessage,
    MessageType,
    MessagePriority,
    create_request_message,
)
from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class MessageRouter:
    """
    Intelligent message router for multi-agent system.

    Handles:
    - Smart routing based on agent capabilities
    - Load balancing
    - Priority-based routing
    - Failover mechanisms
    """

    def __init__(self):
        """Initialize the router."""
        self.protocol = get_protocol()
        self.routing_rules: Dict[str, Any] = {}
        self.agent_loads: Dict[str, int] = {}

    def add_routing_rule(self, action: str, required_capability: str, priority: MessagePriority = MessagePriority.MEDIUM) -> None:
        """
        Add a routing rule.

        Args:
            action: Message action
            required_capability: Required agent capability
            priority: Default priority
        """
        self.routing_rules[action] = {
            "capability": required_capability,
            "priority": priority,
        }
        logger.debug(f"Routing rule added: {action} -> {required_capability}")

    async def route_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Route a message to appropriate agent.

        Args:
            message: Message to route

        Returns:
            Response message or None
        """
        # If receiver is specified, use direct routing
        if message.receiver:
            return await self._route_direct(message)

        # Otherwise, find appropriate agent
        if message.action in self.routing_rules:
            rule = self.routing_rules[message.action]
            agents = self.protocol.find_agents_by_capability(rule["capability"])

            if not agents:
                logger.error(f"No agents found with capability: {rule['capability']}")
                return None

            # Select agent based on load balancing
            selected_agent = self._select_agent(agents)
            message.receiver = selected_agent

            return await self._route_direct(message)
        else:
            logger.warning(f"No routing rule for action: {message.action}")
            return None

    async def _route_direct(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Route message directly to specified agent."""
        if message.requires_response:
            response = await self.protocol.request_response(
                message, timeout=message.timeout_seconds or 60.0
            )
            return response
        else:
            await self.protocol.send_message(message)
            return None

    def _select_agent(self, agents: List[str]) -> str:
        """
        Select an agent based on load balancing.

        Args:
            agents: List of eligible agents

        Returns:
            Selected agent ID
        """
        # Initialize load tracking if needed
        for agent in agents:
            if agent not in self.agent_loads:
                self.agent_loads[agent] = 0

        # Select agent with lowest load
        selected_agent = min(agents, key=lambda a: self.agent_loads.get(a, 0))

        # Increment load
        self.agent_loads[selected_agent] = self.agent_loads.get(selected_agent, 0) + 1

        return selected_agent

    def decrease_load(self, agent_id: str) -> None:
        """Decrease load count for an agent."""
        if agent_id in self.agent_loads:
            self.agent_loads[agent_id] = max(0, self.agent_loads[agent_id] - 1)

    async def broadcast(self, sender: str, action: str, payload: Dict[str, Any]) -> None:
        """
        Broadcast message to all agents.

        Args:
            sender: Sender agent ID
            action: Action name
            payload: Message payload
        """
        from healthdq.communication.message import create_broadcast_message

        message = create_broadcast_message(sender, action, payload)
        await self.protocol.send_message(message)

    async def query_agents(
        self, sender: str, action: str, payload: Dict[str, Any], capability: Optional[str] = None, timeout: float = 30.0
    ) -> List[AgentMessage]:
        """
        Query multiple agents and collect responses.

        Args:
            sender: Sender agent ID
            action: Action name
            payload: Message payload
            capability: Filter agents by capability (if None, query all)
            timeout: Timeout in seconds

        Returns:
            List of responses
        """
        # Get target agents
        if capability:
            agents = self.protocol.find_agents_by_capability(capability)
        else:
            agents = self.protocol.list_active_agents()
            # Remove sender
            agents = [a for a in agents if a != sender]

        if not agents:
            logger.warning("No agents to query")
            return []

        # Create messages for each agent
        messages = []
        for agent in agents:
            msg = create_request_message(
                sender=sender,
                receiver=agent,
                action=action,
                payload=payload,
                requires_response=True,
            )
            messages.append(msg)

        # Send all messages and wait for responses
        tasks = [self.protocol.request_response(msg, timeout=timeout) for msg in messages]

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out None and exceptions
            valid_responses = [r for r in responses if isinstance(r, AgentMessage)]

            logger.info(f"Query completed: {len(valid_responses)}/{len(agents)} responses received")
            return valid_responses

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []

    async def coordinate_task(
        self, sender: str, task_description: str, required_capabilities: List[str], data_reference: str
    ) -> Dict[str, Any]:
        """
        Coordinate a multi-agent task.

        Args:
            sender: Coordinator agent ID
            task_description: Task description
            required_capabilities: Required capabilities
            data_reference: Reference to data

        Returns:
            Task coordination results
        """
        from healthdq.communication.message import CollaborationRequest

        # Find agents for each capability
        participating_agents = {}
        for capability in required_capabilities:
            agents = self.protocol.find_agents_by_capability(capability)
            if agents:
                # Select first available agent (can be improved with more logic)
                participating_agents[capability] = agents[0]
            else:
                logger.error(f"No agent found with capability: {capability}")
                return {"status": "failed", "error": f"Missing capability: {capability}"}

        # Send collaboration requests
        import uuid

        task_id = str(uuid.uuid4())
        responses = {}

        for capability, agent_id in participating_agents.items():
            request = CollaborationRequest(
                task_id=task_id,
                task_description=task_description,
                required_capabilities=[capability],
                data_reference=data_reference,
            )

            message = create_request_message(
                sender=sender,
                receiver=agent_id,
                action="collaboration_request",
                payload=request.dict(),
                requires_response=True,
            )

            response = await self.protocol.request_response(message, timeout=30.0)
            if response:
                responses[agent_id] = response.payload
            else:
                logger.warning(f"No response from {agent_id} for collaboration")

        return {
            "status": "coordinated",
            "task_id": task_id,
            "participating_agents": participating_agents,
            "responses": responses,
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_rules": len(self.routing_rules),
            "agent_loads": self.agent_loads.copy(),
            "active_agents": len(self.protocol.list_active_agents()),
        }


# Global router instance
_router: Optional[MessageRouter] = None


def get_router() -> MessageRouter:
    """Get the global router instance."""
    global _router
    if _router is None:
        _router = MessageRouter()
    return _router


__all__ = ["MessageRouter", "get_router"]
