"""HTTP client for the 8083 classifier server.

Design goals:
  • FAST — no unnecessary polling overhead; default poll interval 2 s
  • DOES NOT TIME OUT on long-running classifications (large bundles can
    take 10+ minutes). We use the async pattern: POST /classify with a
    pre-generated job_id, then poll /api/progress until DONE appears,
    then fetch /api/job/{job_id} for the full classification.json.
  • RESILIENT — retries transient network failures, surfaces hard errors
    fast (8083 down, malformed PDF, etc.).
  • OBSERVABLE — relays 8083's progress events to a callback so the 8082
    UI can show what 8083 is doing in real-time.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import requests


class EightThreeError(RuntimeError):
    """Raised when the 8083 server is unreachable, returns a non-2xx, or
    reports a classification failure."""


# Defaults — overridable via env or constructor args.
DEFAULT_BASE_URL = os.environ.get('CLASSIFIER_8083_URL', 'http://localhost:8083')
DEFAULT_POLL_INTERVAL = float(os.environ.get('CLASSIFIER_8083_POLL', '2.0'))
DEFAULT_TIMEOUT_TOTAL = int(os.environ.get('CLASSIFIER_8083_TIMEOUT', '1800'))  # 30 min
DEFAULT_HTTP_TIMEOUT = 60  # individual request timeout


class EightThreeClient:
    """Thin client around the 8083 /classify + /api/progress + /api/job
    endpoints. Single-threaded; thread-safe to create one per job."""

    def __init__(self,
                 base_url: str = DEFAULT_BASE_URL,
                 poll_interval: float = DEFAULT_POLL_INTERVAL,
                 total_timeout: int = DEFAULT_TIMEOUT_TOTAL,
                 http_timeout: int = DEFAULT_HTTP_TIMEOUT):
        self.base_url = base_url.rstrip('/')
        self.poll_interval = poll_interval
        self.total_timeout = total_timeout
        self.http_timeout = http_timeout

    # ── Health / readiness ──────────────────────────────────────────
    def health(self) -> Dict:
        """Returns the 8083 /health payload. Raises EightThreeError if
        the server is down or unreachable."""
        try:
            r = requests.get(f'{self.base_url}/health',
                             timeout=self.http_timeout)
        except requests.RequestException as e:
            raise EightThreeError(f'8083 health unreachable: {e}') from e
        if r.status_code != 200:
            raise EightThreeError(f'8083 health returned {r.status_code}')
        return r.json()

    def is_ready(self) -> bool:
        """True if the 8083 server is reachable AND healthy."""
        try:
            h = self.health()
            return h.get('status') == 'ok'
        except EightThreeError:
            return False

    # ── End-to-end classification ───────────────────────────────────
    def classify(self,
                 pdf_path: str,
                 job_id: Optional[str] = None,
                 progress_cb: Optional[Callable[[Dict], None]] = None
                 ) -> Tuple[str, Dict]:
        """Classify a PDF end-to-end.

        Args:
            pdf_path: filesystem path to the PDF to classify
            job_id: optional pre-chosen job id. When supplied, 8083 will
                store its results under RESULTS_DIR/<job_id>/. Useful so
                the 8082 server can reuse 8083's per-page artifacts
                (source.pdf, page images) without re-uploading.
            progress_cb: optional callback receiving each new progress
                event from 8083. Called with a dict like
                {'ts': '18:10:11', 'level': 'STEP1', 'msg': 'pg5: ...'}.

        Returns:
            (job_id, classification_dict) — the full classification.json
            that 8083 produces, with `pages`, `logical_documents`,
            `physical_packets`, `final_lc`, `lc_requirements`, etc.

        Raises:
            EightThreeError if anything goes wrong (8083 down,
            classification failed, timeout, etc.).
        """
        if not os.path.exists(pdf_path):
            raise EightThreeError(f'PDF not found: {pdf_path}')

        # Confirm 8083 is up before we even bother uploading — fail fast
        if not self.is_ready():
            raise EightThreeError(
                '8083 classifier server is not reachable at '
                f'{self.base_url}. Start it with: python server.py '
                'from the classifier_server directory.'
            )

        job_id = job_id or uuid.uuid4().hex

        # POST PDF to /classify in a background thread so we can poll
        # progress concurrently. 8083 itself runs the classify_pdf in a
        # worker thread on its side, so it serves /api/progress in
        # parallel.
        import threading
        result_holder: Dict[str, object] = {}

        def _do_classify():
            try:
                with open(pdf_path, 'rb') as f:
                    files = {'file': (os.path.basename(pdf_path), f,
                                       'application/pdf')}
                    # Long timeout — 8083 streams nothing until done
                    r = requests.post(
                        f'{self.base_url}/classify',
                        params={'job_id': job_id},
                        files=files,
                        timeout=self.total_timeout,
                    )
                if r.status_code != 200:
                    result_holder['error'] = EightThreeError(
                        f'8083 /classify returned {r.status_code}: '
                        f'{r.text[:200]}'
                    )
                else:
                    result_holder['result'] = r.json()
            except requests.RequestException as e:
                result_holder['error'] = EightThreeError(
                    f'8083 /classify network error: {e}'
                )
            except Exception as e:
                result_holder['error'] = EightThreeError(
                    f'8083 /classify unexpected error: {e}'
                )

        t = threading.Thread(target=_do_classify, daemon=True,
                             name=f'8083-classify-{job_id[:8]}')
        t.start()

        # Poll progress while the classify thread runs
        seen_count = 0
        deadline = time.time() + self.total_timeout
        last_progress_change = time.time()
        while t.is_alive():
            if time.time() > deadline:
                raise EightThreeError(
                    f'8083 classification timed out after '
                    f'{self.total_timeout}s (job_id={job_id})'
                )
            try:
                pr = requests.get(
                    f'{self.base_url}/api/progress/{job_id}',
                    params={'since': seen_count},
                    timeout=self.http_timeout,
                )
                if pr.status_code == 200:
                    body = pr.json()
                    events = body.get('events') or []
                    if events:
                        last_progress_change = time.time()
                        for ev in events:
                            if progress_cb:
                                try:
                                    progress_cb(ev)
                                except Exception:
                                    pass  # progress cb errors must not
                                          # break classification
                        seen_count += len(events)
            except requests.RequestException:
                # Transient network error — keep polling, the classify
                # thread is still running. Don't fail until total deadline.
                pass
            t.join(timeout=self.poll_interval)

        # Classify thread is done — propagate error or result
        if 'error' in result_holder:
            raise result_holder['error']
        result = result_holder.get('result')
        if not isinstance(result, dict):
            # Fall back to /api/job/{job_id} in case the POST returned
            # weird (rare with the worker-thread setup, but defensive)
            try:
                r = requests.get(f'{self.base_url}/api/job/{job_id}',
                                 timeout=self.http_timeout)
                if r.status_code == 200:
                    result = r.json()
            except requests.RequestException:
                pass
        if not isinstance(result, dict):
            raise EightThreeError(
                f'8083 returned no classification for job {job_id}'
            )
        return job_id, result

    # ── Page-image proxy ────────────────────────────────────────────
    def page_image_url(self, job_id: str, page_num: int,
                       dpi: int = 100) -> str:
        """Build the URL for a rendered page image, hosted by 8083.
        Useful for the UI to embed images without re-rendering on 8082."""
        return (f'{self.base_url}/api/page-image/{job_id}/{page_num}'
                f'?dpi={dpi}')

    def get_classification(self, job_id: str) -> Dict:
        """Fetch a previously-computed classification by job_id (for
        resume / re-fetch on the 8082 side without re-classifying)."""
        try:
            r = requests.get(f'{self.base_url}/api/job/{job_id}',
                             timeout=self.http_timeout)
        except requests.RequestException as e:
            raise EightThreeError(
                f'8083 get_classification network error: {e}'
            ) from e
        if r.status_code == 404:
            raise EightThreeError(f'Job {job_id} not found on 8083')
        if r.status_code != 200:
            raise EightThreeError(
                f'8083 /api/job/{job_id} returned {r.status_code}'
            )
        return r.json()

    def page_positions_url(self, job_id: str, page_num: int) -> str:
        """URL for the per-page text-position overlay 8083 hosts. The UI
        fetches this to draw bbox-anchored text on the rendered image."""
        return f'{self.base_url}/api/page-positions/{job_id}/{page_num}'

    def cancel_job(self, job_id: str) -> bool:
        """Tell 8083 to cancel an in-flight job. Idempotent — already-
        completed jobs return ok. Returns True on 2xx, False otherwise."""
        try:
            r = requests.post(f'{self.base_url}/api/job/{job_id}/cancel',
                              timeout=self.http_timeout)
            return r.status_code < 300
        except requests.RequestException:
            return False

    def delete_job(self, job_id: str) -> bool:
        """Tell 8083 to delete a job's results (source.pdf, page cache,
        classification.json). Used by the 8082 delete endpoint to keep
        the two servers in sync. Returns True if 8083 deleted it (or it
        wasn't there)."""
        try:
            r = requests.delete(f'{self.base_url}/api/job/{job_id}',
                                timeout=self.http_timeout)
            return r.status_code < 300 or r.status_code == 404
        except requests.RequestException:
            return False
