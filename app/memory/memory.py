from collections import deque, defaultdict

class MemoryStore:
    def __init__(self, max_turns=5):
        self.store = defaultdict(lambda: deque(maxlen=max_turns))

    def add(self, session_id: str, role: str, content: str):
        self.store[session_id].append({"role": role, "content": content})

    def get(self, session_id: str):
        return list(self.store[session_id])


short_term_memory = MemoryStore()