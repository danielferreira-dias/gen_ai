import sqlite3
from datetime import datetime
from typing import Optional

class EvaluationStorage:
    def __init__(self, db_path: str = "./db/evaluation.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_judge_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                message_id INTEGER,
                user_query TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                relevance INTEGER CHECK(relevance >= 0 AND relevance <= 5) NOT NULL,
                accuracy INTEGER CHECK(accuracy >= 0 AND accuracy <= 5) NOT NULL,
                pii_violation INTEGER CHECK(pii_violation IN (0, 1)) NOT NULL DEFAULT 0,
                safety_violation INTEGER CHECK(safety_violation IN (0, 1)) NOT NULL DEFAULT 0,
                clarity INTEGER CHECK(clarity >= 0 AND clarity <= 5) NOT NULL,
                overall_score INTEGER CHECK(overall_score >= 0 AND overall_score <= 5) NOT NULL,
                rationale TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_id
            ON llm_judge_evaluations(conversation_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON llm_judge_evaluations(created_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_overall_score
            ON llm_judge_evaluations(overall_score)
        """)

        # Daily metrics table for aggregated statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                total_evaluations INTEGER DEFAULT 0,
                avg_score REAL,
                high_relevance_count INTEGER DEFAULT 0,
                medium_relevance_count INTEGER DEFAULT 0,
                low_relevance_count INTEGER DEFAULT 0,
                pii_violations INTEGER DEFAULT 0,
                safety_violations INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_date
            ON daily_metrics(date)
        """)

        # User feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                message_id INTEGER,
                evaluation_id INTEGER,
                feedback INTEGER CHECK(feedback IN (0, 1)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evaluation_id) REFERENCES llm_judge_evaluations(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_message
            ON user_feedback(message_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_evaluation
            ON user_feedback(evaluation_id)
        """)

        conn.commit()
        conn.close()

    def add_evaluation(
        self,
        user_query: str,
        assistant_response: str,
        relevance: int,
        accuracy: int,
        pii_violation: int,
        safety_violation: int,
        clarity: int,
        overall_score: int,
        rationale: str,
        conversation_id: Optional[int] = None,
        message_id: Optional[int] = None
    ) -> int:
        """
        Add an evaluation record to the database

        Args:
            user_query: The user's query that was evaluated
            assistant_response: The assistant's response that was evaluated
            relevance: Relevance score (0-5)
            accuracy: Accuracy score (0-5)
            pii_violation: Whether PII violation was detected (0 or 1)
            safety_violation: Whether safety violation was detected (0 or 1)
            clarity: Clarity score (0-5)
            overall_score: Overall quality score (0-5)
            rationale: Explanation of the evaluation
            conversation_id: Optional reference to conversation
            message_id: Optional reference to specific message

        Returns:
            evaluation_id: The ID of the inserted evaluation
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO llm_judge_evaluations (
                conversation_id, message_id, user_query, assistant_response,
                relevance, accuracy, pii_violation, safety_violation,
                clarity, overall_score, rationale
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id, message_id, user_query, assistant_response,
                relevance, accuracy, pii_violation, safety_violation,
                clarity, overall_score, rationale
            )
        )
        evaluation_id = cursor.lastrowid

        # Update daily metrics
        self._update_daily_metrics(cursor, datetime.now().date())

        conn.commit()
        conn.close()

        return evaluation_id

    def delete_evaluation(self, evaluation_id: int):
        """Delete a specific evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM llm_judge_evaluations WHERE id = ?",
            (evaluation_id,)
        )

        conn.commit()
        conn.close()

    def save_user_feedback(
        self,
        conversation_id: int,
        message_id: int,
        feedback: int,
        evaluation_id: Optional[int] = None
    ) -> int:
        """
        Save user feedback (thumbs up/down) for a message

        Args:
            conversation_id: The conversation ID
            message_id: The message ID being rated
            feedback: 0 for thumbs down, 1 for thumbs up
            evaluation_id: Optional link to LLM judge evaluation

        Returns:
            feedback_id: The ID of the inserted feedback record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_feedback (conversation_id, message_id, evaluation_id, feedback)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, message_id, evaluation_id, feedback)
        )
        feedback_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return feedback_id

    def _update_daily_metrics(self, cursor, date):
        """
        Update daily metrics aggregation table

        Args:
            cursor: Database cursor
            date: Date to update metrics for
        """
        # Get all evaluations for the date
        cursor.execute(
            """
            SELECT
                COUNT(*) as total,
                AVG(overall_score) as avg_score,
                SUM(CASE WHEN relevance >= 4 THEN 1 ELSE 0 END) as high_relevance,
                SUM(CASE WHEN relevance = 3 THEN 1 ELSE 0 END) as medium_relevance,
                SUM(CASE WHEN relevance <= 2 THEN 1 ELSE 0 END) as low_relevance,
                SUM(pii_violation) as pii_violations,
                SUM(safety_violation) as safety_violations
            FROM llm_judge_evaluations
            WHERE DATE(created_at) = ?
            """,
            (date,)
        )
        result = cursor.fetchone()

        if result:
            total, avg_score, high_rel, med_rel, low_rel, pii_viol, safety_viol = result

            # Insert or update daily metrics
            cursor.execute(
                """
                INSERT INTO daily_metrics (
                    date, total_evaluations, avg_score,
                    high_relevance_count, medium_relevance_count, low_relevance_count,
                    pii_violations, safety_violations, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(date) DO UPDATE SET
                    total_evaluations = excluded.total_evaluations,
                    avg_score = excluded.avg_score,
                    high_relevance_count = excluded.high_relevance_count,
                    medium_relevance_count = excluded.medium_relevance_count,
                    low_relevance_count = excluded.low_relevance_count,
                    pii_violations = excluded.pii_violations,
                    safety_violations = excluded.safety_violations,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (date, total or 0, avg_score or 0.0, high_rel or 0, med_rel or 0, low_rel or 0, pii_viol or 0, safety_viol or 0)
            )
