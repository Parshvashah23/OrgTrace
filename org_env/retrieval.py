"""
OrgMemory-Env: Retrieval Engine
BM25-based search with filtering and thread tracing.
"""

import re
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from rank_bm25 import BM25Okapi

from .models import Message


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25 indexing."""
    text = text.lower()
    # Remove common punctuation but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


class RetrievalEngine:
    """
    Retrieval engine for the message corpus.
    Provides BM25 search with filtering and thread tracing.
    """

    def __init__(self, messages: List[Dict[str, Any]]):
        """
        Initialize the retrieval engine.

        Args:
            messages: List of message dicts from the corpus
        """
        self.messages = messages
        self.message_by_id: Dict[str, Dict[str, Any]] = {
            m["message_id"]: m for m in messages
        }
        self.thread_index: Dict[str, List[Dict[str, Any]]] = {}

        # Build thread index
        for m in messages:
            tid = m.get("thread_id")
            if tid:
                if tid not in self.thread_index:
                    self.thread_index[tid] = []
                self.thread_index[tid].append(m)

        # Sort thread messages by timestamp
        for tid in self.thread_index:
            self.thread_index[tid].sort(key=lambda m: m["timestamp"])

        # Build BM25 index
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build the BM25 index from message bodies and subjects."""
        corpus = []
        for m in self.messages:
            # Combine subject and body for indexing
            text = ""
            if m.get("subject"):
                text += m["subject"] + " "
            text += m.get("body", "")
            corpus.append(tokenize(text))

        self.bm25 = BM25Okapi(corpus)
        self.corpus_tokens = corpus

    def search(
        self,
        query: str,
        person_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        project_id: Optional[str] = None,
        channel: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for messages using BM25 with optional filters.

        Args:
            query: Search query string
            person_id: Filter by sender ID
            date_from: Filter messages on or after this date
            date_to: Filter messages on or before this date
            project_id: Filter by project tag
            channel: Filter by channel (exact match or prefix)
            top_k: Maximum number of results to return

        Returns:
            List of matching message dicts, sorted by BM25 score
        """
        # Tokenize query
        query_tokens = tokenize(query)

        if not query_tokens:
            # If empty query, return filtered results by recency
            filtered = self._apply_filters(
                self.messages, person_id, date_from, date_to, project_id, channel
            )
            # Sort by timestamp descending
            filtered.sort(key=lambda m: m["timestamp"], reverse=True)
            return filtered[:top_k]

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Create list of (message, score) tuples
        scored_messages = list(zip(self.messages, scores))

        # Apply filters
        filtered_scored = []
        for msg, score in scored_messages:
            if self._passes_filters(msg, person_id, date_from, date_to, project_id, channel):
                filtered_scored.append((msg, score))

        # Sort by score descending
        filtered_scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return [msg for msg, score in filtered_scored[:top_k] if score > 0]

    def _passes_filters(
        self,
        msg: Dict[str, Any],
        person_id: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        project_id: Optional[str],
        channel: Optional[str],
    ) -> bool:
        """Check if a message passes all filters."""
        # Person filter
        if person_id:
            if msg.get("sender_id") != person_id:
                return False

        # Date filters
        msg_ts = msg.get("timestamp")
        if msg_ts:
            if isinstance(msg_ts, str):
                msg_ts = datetime.fromisoformat(msg_ts)
            if date_from and msg_ts < date_from:
                return False
            if date_to and msg_ts > date_to:
                return False

        # Project filter
        if project_id:
            if msg.get("project_tag") != project_id:
                return False

        # Channel filter (supports prefix matching like "slack" matching "slack:#channel")
        if channel:
            msg_channel = msg.get("channel", "")
            if not msg_channel.startswith(channel):
                return False

        return True

    def _apply_filters(
        self,
        messages: List[Dict[str, Any]],
        person_id: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        project_id: Optional[str],
        channel: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Apply filters to a list of messages."""
        return [
            m for m in messages
            if self._passes_filters(m, person_id, date_from, date_to, project_id, channel)
        ]

    def trace_thread(
        self,
        thread_id: str,
        direction: Literal["forward", "backward", "both"] = "both",
        reference_message_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all messages in a thread.

        Args:
            thread_id: The thread ID to trace
            direction: Which direction to trace:
                - "forward": Messages after the reference (or all if no reference)
                - "backward": Messages before the reference (or all if no reference)
                - "both": All messages in the thread
            reference_message_id: Optional message ID to use as reference point

        Returns:
            List of messages in the thread, sorted by timestamp
        """
        thread_messages = self.thread_index.get(thread_id, [])

        if not thread_messages:
            return []

        if direction == "both" or reference_message_id is None:
            return thread_messages.copy()

        # Find reference message index
        ref_idx = None
        for i, m in enumerate(thread_messages):
            if m["message_id"] == reference_message_id:
                ref_idx = i
                break

        if ref_idx is None:
            return thread_messages.copy()

        if direction == "forward":
            return thread_messages[ref_idx:]
        elif direction == "backward":
            return thread_messages[:ref_idx + 1]

        return thread_messages.copy()

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a single message by ID."""
        return self.message_by_id.get(message_id)

    def get_messages_by_sender(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all messages from a specific sender."""
        return [m for m in self.messages if m.get("sender_id") == person_id]

    def get_messages_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all messages tagged with a specific project."""
        return [m for m in self.messages if m.get("project_tag") == project_id]

    def get_messages_by_day(self, day: int) -> List[Dict[str, Any]]:
        """Get all messages from a specific simulation day."""
        return [m for m in self.messages if m.get("day") == day]

    def get_messages_in_range(self, start_day: int, end_day: int) -> List[Dict[str, Any]]:
        """Get all messages within a day range (inclusive)."""
        return [
            m for m in self.messages
            if start_day <= m.get("day", 0) <= end_day
        ]
