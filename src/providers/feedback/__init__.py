"""Feedback persistence providers (user ratings on analysis results).

SQLiteFeedbackProvider stores thumbs-up/down ratings in data/feedback.db.
Ratings are used to:
    1. Let users mark inaccurate results (thumbs down) to improve future runs
    2. Track overall accuracy metrics across all sessions
    3. Filter out previously-downvoted items in cross-session recommendations
"""
