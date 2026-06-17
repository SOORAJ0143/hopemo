# app/engines/response/validator.py
import re
import asyncio
import logging
from typing import Dict, Any

from detoxify import Detoxify
from app.engines.safety.detector import SafetyDetector  # your existing class

logger = logging.getLogger(__name__)

class ResponseValidator:
    def __init__(self, use_moderation: bool = True):
        """
        :param use_moderation: If True, also runs OpenAI Moderation via SafetyDetector.
        """
        # Load Detoxify (the model loads once at startup)
        self.detoxify = Detoxify('original')
        self.use_moderation = use_moderation
        if use_moderation:
            self.safety_detector = SafetyDetector()  # reuses OpenAI client

        # Expanded harmful patterns (with word boundaries to avoid accidental matches)
        self.harmful_patterns = [
            r"\bkill\s+yourself\b",
            r"\bcommit\s+suicide\b",
            r"\bend\s+your\s+life\b",
            r"\bwant\s+to\s+die\b",
            r"\bharm\s+yourself\b",
            r"\bself\s*[-]?\s*harm\b",
            r"\bdeath\s+is\s+the\s+only\s+way\b",
            r"\byou\s+should\s+just\s+give\s+up\b",
            r"\bno\s+one\s+would\s+miss\s+you\b",
            r"\byou're\s+a\s+failure\b",    # be careful – may flag a lot
            r"\bdo\s+it\s+now\b",          # subtle but possible in context
            r"\btake\s+the\s+easy\s+way\s+out\b",
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.harmful_patterns]

    async def validate(self, response: str) -> Dict[str, Any]:
        """
        Validates the assistant's response for safety.
        Returns a dict with:
            - safe: bool
            - toxicity: float
            - harmful_detected: bool
            - moderation_flagged: bool (if use_moderation=True)
            - issues: list of strings describing why it failed (if any)
        """
        issues = []
        toxicity = 0.0
        harmful_detected = False
        moderation_flagged = False

        # 1. Run Detoxify (synchronous -> offload to thread)
        try:
            detox_scores = await asyncio.to_thread(self.detoxify.predict, response)
            toxicity = detox_scores.get('toxicity', 0.0)
            # Also check severe_toxicity for higher confidence
            severe_tox = detox_scores.get('severe_toxicity', 0.0)
        except Exception as e:
            logger.error(f"Detoxify prediction failed: {e}")
            # In case of failure, we treat as unsafe to be conservative
            issues.append("Detoxify failed; defaulting to unsafe")
            return {
                "safe": False,
                "toxicity": 1.0,
                "harmful_detected": True,
                "moderation_flagged": True,
                "issues": issues,
            }

        # 2. Check against harmful patterns
        for pattern in self.compiled_patterns:
            if pattern.search(response):
                harmful_detected = True
                issues.append(f"Matched pattern: {pattern.pattern}")
                break

        # 3. Optionally run OpenAI Moderation via SafetyDetector
        if self.use_moderation:
            try:
                # SafetyDetector.check() expects a text and returns a dict with 'flagged_categories' etc.
                moderation_result = await self.safety_detector.check(response)
                moderation_flagged = moderation_result.get('risk_level') in ['high', 'critical']
                if moderation_flagged:
                    issues.append(f"Moderation flagged: {moderation_result.get('flagged_categories', [])}")
            except Exception as e:
                logger.error(f"Moderation check failed: {e}")
                # Fail open? Better to flag as unsafe.
                moderation_flagged = True
                issues.append("Moderation service failed; flagged as unsafe")

        # 4. Combine safety decision
        # Consider unsafe if: toxicity >= 0.5 OR severe_tox >= 0.4 OR harmful pattern found OR moderation flagged
        is_safe = not (
            toxicity >= 0.5 or severe_tox >= 0.4 or harmful_detected or moderation_flagged
        )

        # If unsafe, log the full response (but be careful with PII)
        if not is_safe:
            logger.warning(f"Unsafe response detected. Toxicity={toxicity}, harmful={harmful_detected}, moderation={moderation_flagged}. Response snippet: {response[:200]}")

        return {
            "safe": is_safe,
            "toxicity": toxicity,
            "harmful_detected": harmful_detected,
            "moderation_flagged": moderation_flagged,
            "issues": issues,
        }