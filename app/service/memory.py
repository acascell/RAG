from collections import deque, defaultdict


class MemoryStore:
    """In-memory conversation history store with per-session sliding window.

    Maintains a fixed-size buffer of recent conversation turns for each session,
    automatically evicting the oldest messages when the limit is reached.
    """

    def __init__(self, max_turns=5):
        """Initialize the memory store.

        Args:
            max_turns: Maximum number of conversation turns to retain per session.
        """
        self.store = defaultdict(lambda: deque(maxlen=max_turns))

    def add(self, session_id: str, role: str, content: str):
        """Add a conversation turn to the session history.

        Args:
            session_id: Unique identifier for the conversation session.
            role: The speaker role (e.g., 'user' or 'assistant').
            content: The message content.
        """
        self.store[session_id].append({"role": role, "content": content})

    def get(self, session_id: str):
        """Retrieve the conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            A list of conversation turn dicts with 'role' and 'content' keys.
        """
        return list(self.store[session_id])


memory = MemoryStore()