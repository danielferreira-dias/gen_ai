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
                relevance TEXT CHECK(relevance IN ('High', 'Medium', 'Low')),
                accuracy TEXT CHECK(accuracy IN ('Accurate', 'Partially Accurate', 'Inaccurate')),
                pii_violation INTEGER CHECK(pii_violation IN (0, 1)) DEFAULT 0,
                safety_violation INTEGER CHECK(safety_violation IN (0, 1)) DEFAULT 0,
                clarity TEXT CHECK(clarity IN ('Excellent', 'Good', 'Poor')),
                overall_score INTEGER CHECK(overall_score >= 0 AND overall_score <= 5),
                rationale TEXT,
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

        conn.commit()
        conn.close()

    def add_evaluation(
        self,
        user_query: str,
        assistant_response: str,
        relevance: str,
        accuracy: str,
        pii_violation: int,
        safety_violation: int,
        clarity: str,
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
            relevance: High, Medium, or Low
            accuracy: Accurate, Partially Accurate, or Inaccurate
            pii_violation: Whether PII violation was detected (0 or 1)
            safety_violation: Whether safety violation was detected (0 or 1)
            clarity: Excellent, Good, or Poor
            overall_score: Score from 0-5
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

    def delete_old_evaluations(self, days_to_keep: int = 90):
        """Delete evaluations older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM llm_judge_evaluations
            WHERE created_at < datetime('now', '-' || ? || ' days')
            """,
            (days_to_keep,)
        )

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def _update_daily_metrics(self, cursor, date):
        """
        Update or create daily metrics for the given date

        Args:
            cursor: Active database cursor
            date: Date object for which to update metrics
        """
        date_str = date.strftime('%Y-%m-%d')

        # Calculate metrics for the date
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                AVG(overall_score) as avg_score,
                SUM(CASE WHEN relevance = 'High' THEN 1 ELSE 0 END) as high_rel,
                SUM(CASE WHEN relevance = 'Medium' THEN 1 ELSE 0 END) as medium_rel,
                SUM(CASE WHEN relevance = 'Low' THEN 1 ELSE 0 END) as low_rel,
                SUM(CASE WHEN pii_violation = 1 THEN 1 ELSE 0 END) as pii_count,
                SUM(CASE WHEN safety_violation = 1 THEN 1 ELSE 0 END) as safety_count
            FROM llm_judge_evaluations
            WHERE DATE(created_at) = ?
        """, (date_str,))

        result = cursor.fetchone()

        if result[0] > 0:  # If there are evaluations for this date
            # Insert or replace the daily metrics
            cursor.execute("""
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
            """, (
                date_str,
                result[0],  # total
                result[1],  # avg_score
                result[2],  # high_rel
                result[3],  # medium_rel
                result[4],  # low_rel
                result[5],  # pii_count
                result[6]   # safety_count
            ))
