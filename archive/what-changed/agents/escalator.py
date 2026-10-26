"""Escalator: tells a sighted helper when something isn't safe for the user to handle alone."""
import logging
import os

import boto3

log = logging.getLogger(__name__)


def notify(home: str, message: str) -> dict:
    topic = os.environ.get("ALERT_TOPIC_ARN")
    if not topic:
        log.info("escalation for %s (no ALERT_TOPIC_ARN configured): %s", home, message)
        return {"sent": False, "channel": "log"}
    boto3.client("sns").publish(TopicArn=topic, Subject=f"What Changed: {home}"[:100], Message=message)
    return {"sent": True, "channel": "sns"}
