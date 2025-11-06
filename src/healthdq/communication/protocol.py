"""
Agent Communication Protocol (ACP) implementation
Author: Agate Jarmakoviča
"""

from typing import Any, Callable, Dict, List, Optional, Set
from asyncio import Queue, create_task, wait_for, TimeoutError
from datetime import datetime
import asyncio

from healthdq.communication.message import (
    AgentMessage,
    MessageType,
    MessagePriority,
    create_response_message,
)
from healthdq.utils.logger import get_logger

logger = get_logger(__name__)


class AgentCommunicationProtocol:
    """
    Implements the Agent Communication Protocol (ACP).

    Handles:
    - Message routing between agents
    - Request/response patterns
    - Broadcast messaging
    - Message queuing and prioritization
    - Timeout handling
    """

    def __init__(self):
        """Initialize the protocol."""
        self.message_queues: Dict[str, Queue] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, Dict[str, Callable]] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.conversation_history: Dict[str, List[AgentMessage]] = {}

        logger.info("Agent Communication Protocol initialized")

    def register_agent(self, agent_id: str, capabilities: List[str], metadata: Optional[Dict] = None) -> None:
        """
        Register an agent in the system.

        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            metadata: Additional agent metadata
        """
        self.message_queues[agent_id] = Queue()
        self.agent_registry[agent_id] = {
            "agent_id": agent_id,
            "capabilities": capabilities,
            "registered_at": datetime.now().isoformat(),
            "status": "active",
            "metadata": metadata or {},
        }
        self.message_handlers[agent_id] = {}

        logger.info(f"Agent registered: {agent_id} with capabilities: {capabilities}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            del self.message_queues[agent_id]
            del self.message_handlers[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")

    def register_handler(self, agent_id: str, action: str, handler: Callable) -> None:
        """
        Register a message handler for an agent.

        Args:
            agent_id: Agent identifier
            action: Action name to handle
            handler: Async callable to handle the message
        """
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}

        self.message_handlers[agent_id][action] = handler
        logger.debug(f"Handler registered for {agent_id}: {action}")

    async def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to an agent or broadcast.

        Args:
            message: Message to send
        """
        # Store in conversation history
        if message.conversation_id:
            if message.conversation_id not in self.conversation_history:
                self.conversation_history[message.conversation_id] = []
            self.conversation_history[message.conversation_id].append(message)

        # Handle broadcast
        if message.receiver is None:
            await self._broadcast_message(message)
        else:
            # Send to specific agent
            if message.receiver not in self.message_queues:
                logger.error(f"Agent not found: {message.receiver}")
                await self._send_error_response(message, f"Agent not found: {message.receiver}")
                return

            await self.message_queues[message.receiver].put(message)
            logger.debug(f"Message sent: {message.sender} -> {message.receiver} [{message.action}]")

    async def _broadcast_message(self, message: AgentMessage) -> None:
        """Broadcast message to all registered agents except sender."""
        for agent_id in self.agent_registry.keys():
            if agent_id != message.sender:
                await self.message_queues[agent_id].put(message)

        logger.debug(f"Message broadcast from {message.sender}: {message.action}")

    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """
        Receive a message for an agent.

        Args:
            agent_id: Agent identifier
            timeout: Timeout in seconds (None for blocking)

        Returns:
            Received message or None if timeout
        """
        if agent_id not in self.message_queues:
            logger.error(f"Agent not found: {agent_id}")
            return None

        try:
            if timeout:
                message = await wait_for(self.message_queues[agent_id].get(), timeout=timeout)
            else:
                message = await self.message_queues[agent_id].get()

            logger.debug(f"Message received by {agent_id}: {message.action}")
            return message

        except TimeoutError:
            logger.debug(f"Receive timeout for agent: {agent_id}")
            return None

    async def request_response(
        self, message: AgentMessage, timeout: float = 60.0
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response.

        Args:
            message: Request message
            timeout: Timeout in seconds

        Returns:
            Response message or None if timeout
        """
        if not message.requires_response:
            await self.send_message(message)
            return None

        # Create a future for the response
        response_future = asyncio.Future()
        self.pending_responses[message.message_id] = response_future

        # Send the message
        await self.send_message(message)

        try:
            # Wait for response
            response = await wait_for(response_future, timeout=timeout)
            return response

        except TimeoutError:
            logger.warning(
                f"Request timeout: {message.sender} -> {message.receiver} [{message.action}]"
            )
            # Clean up
            del self.pending_responses[message.message_id]
            return None

    async def send_response(self, response: AgentMessage) -> None:
        """
        Send a response to a request.

        Args:
            response: Response message
        """
        await self.send_message(response)

        # Check if someone is waiting for this response
        if response.reply_to and response.reply_to in self.pending_responses:
            self.pending_responses[response.reply_to].set_result(response)
            del self.pending_responses[response.reply_to]

    async def process_messages(self, agent_id: str) -> None:
        """
        Process messages for an agent (runs in background).

        Args:
            agent_id: Agent identifier
        """
        logger.info(f"Starting message processor for: {agent_id}")

        while agent_id in self.agent_registry:
            message = await self.receive_message(agent_id, timeout=1.0)

            if message is None:
                continue

            # Handle message
            if message.action in self.message_handlers.get(agent_id, {}):
                handler = self.message_handlers[agent_id][message.action]
                try:
                    # Call handler
                    result = await handler(message)

                    # Send response if required
                    if message.requires_response:
                        response = create_response_message(
                            sender=agent_id,
                            original_message=message,
                            payload={"result": result},
                            success=True,
                        )
                        await self.send_response(response)

                except Exception as e:
                    logger.error(f"Handler error for {agent_id}.{message.action}: {str(e)}")

                    # Send error response
                    if message.requires_response:
                        await self._send_error_response(message, str(e))
            else:
                logger.warning(f"No handler for {agent_id}.{message.action}")

                if message.requires_response:
                    await self._send_error_response(message, f"No handler for action: {message.action}")

    async def _send_error_response(self, original_message: AgentMessage, error_msg: str) -> None:
        """Send an error response."""
        response = create_response_message(
            sender="system",
            original_message=original_message,
            payload={"error": error_msg},
            success=False,
        )
        await self.send_response(response)

    def find_agents_by_capability(self, capability: str) -> List[str]:
        """
        Find agents with a specific capability.

        Args:
            capability: Capability to search for

        Returns:
            List of agent IDs
        """
        agents = []
        for agent_id, info in self.agent_registry.items():
            if capability in info.get("capabilities", []):
                agents.append(agent_id)

        return agents

    def get_conversation_history(self, conversation_id: str) -> List[AgentMessage]:
        """Get conversation history."""
        return self.conversation_history.get(conversation_id, [])

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information."""
        return self.agent_registry.get(agent_id)

    def list_active_agents(self) -> List[str]:
        """List all active agents."""
        return list(self.agent_registry.keys())


# Global protocol instance
_protocol: Optional[AgentCommunicationProtocol] = None


def get_protocol() -> AgentCommunicationProtocol:
    """Get the global protocol instance."""
    global _protocol
    if _protocol is None:
        _protocol = AgentCommunicationProtocol()
    return _protocol


__all__ = ["AgentCommunicationProtocol", "get_protocol"]
