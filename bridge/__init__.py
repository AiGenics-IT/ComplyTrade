"""Bridge between the 8082 main server and the 8083 standalone classifier.

The 8083 classifier (https://github.com/azharnaeem76/classifier-server) owns:
  • OCR (GLM-OCR on RTX 5090)
  • Per-page classification (SWIFT pattern, title regex, VLM image+text)
  • SWIFT message grouping (LC + amendments)
  • Per-printout splitting via Page-of-N markers + T&C absorption
  • Deep field extraction per doc type

The 8082 main server owns:
  • Final LC creation (step06) — the mature, battle-tested FLC builder
  • Clause extraction (step07)
  • All verification + report steps (10–20)
  • The UI

This bridge package provides:
  • EightThreeClient: HTTP client to talk to the 8083 server
  • Adapter: maps 8083's classification.json into the step-result shapes
    that step06 onwards consume, so downstream code does not need to know
    where the classification came from.
"""
from .eight_three_client import EightThreeClient, EightThreeError
from .adapter import adapt_8083_to_step_results

__all__ = ['EightThreeClient', 'EightThreeError', 'adapt_8083_to_step_results']
