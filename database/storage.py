import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

class ConversationStorage:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokenized_content TEXT,
                has_pii BOOLEAN DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # PII events table (for auditing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pii_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                entities_detected INTEGER DEFAULT 0,
                entity_details TEXT,
                token_map TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            )
        """)

        conn.commit()
        conn.close()

    def create_conversation(self, user_id: str = "default") -> int:
        """Create a new conversation and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO conversations (user_id) VALUES (?)",
            (user_id,)
        )
        conversation_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return conversation_id

    def add_message( self, conversation_id: int, role: str, content: str, tokenized_content: Optional[str] = None, has_pii: bool = False, pii_data: Optional[Dict] = None ) -> int:
        """
        Add a message to a conversation

        Args:
            conversation_id: The conversation ID
            role: 'user' or 'assistant'
            content: The message content (de-tokenized)
            tokenized_content: The tokenized version (if PII was detected)
            has_pii: Whether the message contained PII
            pii_data: Dictionary with 'entities', 'token_map', etc.

        Returns:
            message_id: The ID of the inserted message
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert message
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, role, content, tokenized_content, has_pii)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, tokenized_content, has_pii)
        )
        message_id = cursor.lastrowid

        # Update conversation updated_at
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )

        # If PII was detected, log it
        if has_pii and pii_data:
            entities = pii_data.get('entities', [])
            token_map = pii_data.get('token_map', {})

            cursor.execute(
                """
                INSERT INTO pii_events (message_id, entities_detected, entity_details, token_map)
                VALUES (?, ?, ?, ?)
                """,
                (
                    message_id,
                    len(entities),
                    json.dumps(entities),
                    json.dumps(token_map)
                )
            )

        conn.commit()
        conn.close()

        return message_id

    def get_conversation(self, conversation_id: int) -> List[Dict]:
        """Get all messages in a conversation"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, role, content, tokenized_content, has_pii, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            """,
            (conversation_id,)
        )

        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return messages

    def get_recent_conversations(self, user_id: str = "default", limit: int = 10) -> List[Dict]:
        """Get recent conversations for a user"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT c.id, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        )

        conversations = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return conversations

    def delete_conversation(self, conversation_id: int):
        """Delete a conversation and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete PII events first (due to foreign key)
        cursor.execute(
            "DELETE FROM pii_events WHERE message_id IN (SELECT id FROM messages WHERE conversation_id = ?)",
            (conversation_id,)
        )

        # Delete messages
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

        # Delete conversation
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

        conn.commit()
        conn.close()

    def get_pii_audit_log(self, conversation_id: Optional[int] = None) -> List[Dict]:
        """Get PII detection audit log"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if conversation_id:
            cursor.execute(
                """
                SELECT p.*, m.conversation_id, m.role, m.content
                FROM pii_events p
                JOIN messages m ON p.message_id = m.id
                WHERE m.conversation_id = ?
                ORDER BY p.timestamp DESC
                """,
                (conversation_id,)
            )
        else:
            cursor.execute(
                """
                SELECT p.*, m.conversation_id, m.role
                FROM pii_events p
                JOIN messages m ON p.message_id = m.id
                ORDER BY p.timestamp DESC
                LIMIT 100
                """
            )

        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return logs
