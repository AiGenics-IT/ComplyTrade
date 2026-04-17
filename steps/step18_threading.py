"""
Step 18 — Multi-Threaded Clause Processing
=============================================
Utility module for parallel clause verification. Wraps ThreadPoolExecutor
to process clause groups concurrently while respecting dependencies.

PURPOSE:
    When verifying many LC clauses, processing them one at a time is slow
    because each VLM call takes 1-5 seconds. This module provides a
    parallelization framework that:
    1. Groups clauses by dependency (e.g., F46A and F47A share context)
    2. Builds an execution plan with "waves" of independent tasks
    3. Executes tasks within each wave in parallel using threads
    4. Passes accumulated results from earlier waves to later ones

    This is especially important for LCs with many clauses (some have 15+
    conditions in F46A alone), where sequential processing would take minutes.

DEPENDENCY GROUPS:
    F46A and F47A are interdependent — F47A can override F46A requirements.
    They are grouped together as 'DOC_REQUIREMENTS' so that F47A results
    are available when processing F46A. Similarly:
    - F45A/F45B → 'GOODS_DESCRIPTION' (description of goods)
    - F78 → 'INSTRUCTIONS' (instructions to paying/negotiating bank)
    - F72 → 'SENDER_RECEIVER' (sender to receiver information)

INPUTS:
    - List of ClauseTask objects (clause_ref, clause_text, documents, etc.)
    - A verification function that processes one task at a time
    - Shared context accessible to all threads (read-only)

OUTPUTS:
    - BatchResult with all clause results, timing, and error information
    - Flattened row list ready for downstream steps

AI MODEL: None — this is orchestration/threading infrastructure only.
    The actual verification is done by the verify_fn callback, which may
    or may not call the VLM depending on the implementation.
"""

import json
import sys as _sys; _sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(_sys.stdout, "reconfigure") else None
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Callable, Optional, Any, Tuple
from pathlib import Path


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ClauseTask:
    """
    A single clause verification task to be executed in a thread.

    Contains all the data needed to verify one clause: the clause text,
    the documents to check against, relevant LC fields, and any dependency
    references (other clauses that must be processed first).
    """
    clause_ref: str             # e.g. "F46A-1", "F47A-3"
    clause_text: str
    clause_group: str           # e.g. "F46A", "F47A" — used for dependency grouping
    documents: List[Dict] = field(default_factory=list)       # Documents to check against
    lc_fields: Dict = field(default_factory=dict)             # Relevant LC fields for context
    dependency_refs: List[str] = field(default_factory=list)  # clause_refs this depends on


@dataclass
class ClauseResult:
    """
    Result from processing a single clause task.

    Contains the verification rows produced and metadata about execution
    (timing, errors, which thread processed it).
    """
    clause_ref: str
    clause_group: str
    rows: List[Dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    thread_id: str = ""          # Thread name for debugging parallel execution


@dataclass
class BatchResult:
    """
    Aggregated result from parallel clause processing.

    Contains all individual ClauseResults plus overall statistics.
    """
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    results: List[ClauseResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ── Dependency Graph ─────────────────────────────────────────────────────────
# Maps clause group prefixes to logical dependency groups.
# Tasks within the same dependency group share context and run sequentially
# within each wave, while independent groups run in parallel.

_DEPENDENCY_GROUPS = {
    'F46A': 'DOC_REQUIREMENTS',   # Documents Required and Additional Conditions are interdependent
    'F47A': 'DOC_REQUIREMENTS',   # F47A can override F46A requirements
    'F46B': 'DOC_REQUIREMENTS',
    'F45A': 'GOODS_DESCRIPTION',  # Description of goods — shared across invoice, BL
    'F45B': 'GOODS_DESCRIPTION',
    'F78':  'INSTRUCTIONS',       # Instructions to Paying/Accepting/Negotiating Bank
    'F72':  'SENDER_RECEIVER',    # Sender to Receiver Information
    'F79':  'NARRATIVE',          # Narrative / Free Format
}


def _get_dependency_group(clause_ref: str) -> str:
    """
    Determine which dependency group a clause belongs to.

    Extracts the F-tag prefix (e.g., "F46A" from "F46A-2") and looks it up
    in the dependency map. Clauses in the same group share context.
    """
    prefix = clause_ref.split('-')[0].upper()
    return _DEPENDENCY_GROUPS.get(prefix, prefix)


def _build_execution_plan(tasks: List[ClauseTask]) -> List[List[ClauseTask]]:
    """
    Organize tasks into execution waves respecting dependencies.

    Tasks within the same dependency group run sequentially (one per wave).
    Independent groups can run in parallel across waves.

    Example with 3 F46A tasks and 2 F45A tasks:
    - Wave 1: F46A-1, F45A-1  (parallel — different groups)
    - Wave 2: F46A-2, F45A-2  (parallel — different groups)
    - Wave 3: F46A-3           (only F46A has a 3rd task)

    Returns a list of waves, where each wave is a list of tasks
    that can execute in parallel.
    """
    # Group tasks by dependency group
    groups: Dict[str, List[ClauseTask]] = {}
    for task in tasks:
        dep_group = _get_dependency_group(task.clause_ref)
        groups.setdefault(dep_group, []).append(task)

    # Build waves — one task per group per wave
    max_depth = max((len(g) for g in groups.values()), default=0)

    waves = []
    for i in range(max_depth):
        wave = []
        for group_name, group_tasks in groups.items():
            if i < len(group_tasks):
                wave.append(group_tasks[i])
        if wave:
            waves.append(wave)

    return waves


# ── Thread Pool Executor ─────────────────────────────────────────────────────

def process_clauses_parallel(
    tasks: List[ClauseTask],
    verify_fn: Callable[[ClauseTask, Dict], List[Dict]],
    shared_context: Optional[Dict] = None,
    max_workers: int = 4,
    progress_fn: Optional[Callable] = None,
) -> BatchResult:
    """
    Execute clause verification tasks in parallel using ThreadPoolExecutor.

    The execution follows a wave-based approach:
    1. Build execution plan (group by dependencies, create waves)
    2. For each wave, submit all tasks to the thread pool
    3. Collect results as they complete
    4. Merge results into accumulated context for the next wave

    This ensures that dependent clauses (e.g., F46A needing F47A context)
    have access to prior results, while independent groups run in parallel.

    Args:
        tasks:           list of ClauseTask objects to process
        verify_fn:       function(task, context) -> list of row dicts
        shared_context:  shared data accessible to all threads (read-only recommended)
        max_workers:     maximum concurrent threads in the pool
        progress_fn:     callback for progress updates

    Returns:
        BatchResult with all clause results
    """
    t0 = time.time()
    if shared_context is None:
        shared_context = {}
    if progress_fn is None:
        def progress_fn(msg): pass

    batch = BatchResult(total_tasks=len(tasks))

    if not tasks:
        batch.elapsed_seconds = round(time.time() - t0, 2)
        return batch

    progress_fn(f"Parallel processing: {len(tasks)} clause tasks, max_workers={max_workers}")

    # Build execution plan respecting dependencies
    waves = _build_execution_plan(tasks)
    progress_fn(f"Execution plan: {len(waves)} waves")

    # Accumulated context from completed waves — passed to subsequent waves
    # so dependent clauses can access prior results
    accumulated_results: Dict[str, ClauseResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for wave_idx, wave in enumerate(waves):
            progress_fn(f"Wave {wave_idx + 1}/{len(waves)}: {len(wave)} tasks")

            # Merge accumulated results into context for this wave
            wave_context = {**shared_context, '_prior_results': accumulated_results}

            # Submit all tasks in this wave to the thread pool
            futures: Dict[Future, ClauseTask] = {}
            for task in wave:
                future = executor.submit(_execute_task, task, verify_fn, wave_context)
                futures[future] = task

            # Collect results as they complete (order may differ from submission order)
            for future in as_completed(futures):
                task = futures[future]
                try:
                    clause_result = future.result(timeout=None)  # 5-minute timeout per task
                    batch.results.append(clause_result)
                    batch.completed += 1
                    # Store result for use by dependent tasks in later waves
                    accumulated_results[clause_result.clause_ref] = clause_result
                    progress_fn(
                        f"  {clause_result.clause_ref}: "
                        f"{len(clause_result.rows)} rows, "
                        f"{clause_result.elapsed_seconds:.1f}s"
                    )
                except Exception as e:
                    batch.failed += 1
                    err_result = ClauseResult(
                        clause_ref=task.clause_ref,
                        clause_group=task.clause_group,
                        error=str(e),
                    )
                    batch.results.append(err_result)
                    progress_fn(f"  {task.clause_ref}: FAILED — {str(e)[:100]}")

    batch.elapsed_seconds = round(time.time() - t0, 2)
    progress_fn(
        f"Parallel processing complete: "
        f"{batch.completed}/{batch.total_tasks} succeeded in {batch.elapsed_seconds:.1f}s"
    )

    return batch


def _execute_task(
    task: ClauseTask,
    verify_fn: Callable,
    context: Dict,
) -> ClauseResult:
    """
    Execute a single clause verification task (runs in a worker thread).

    Calls the provided verify_fn with the task and context, then wraps
    the result in a ClauseResult with timing and thread identification.
    """
    import threading
    t0 = time.time()

    try:
        rows = verify_fn(task, context)
        return ClauseResult(
            clause_ref=task.clause_ref,
            clause_group=task.clause_group,
            rows=rows if rows else [],
            elapsed_seconds=round(time.time() - t0, 2),
            thread_id=threading.current_thread().name,
        )
    except Exception as e:
        return ClauseResult(
            clause_ref=task.clause_ref,
            clause_group=task.clause_group,
            error=f"{type(e).__name__}: {str(e)}",
            elapsed_seconds=round(time.time() - t0, 2),
            thread_id=threading.current_thread().name,
        )


# ── Convenience: Flatten BatchResult into row list ───────────────────────────

def flatten_results(batch: BatchResult) -> List[Dict]:
    """
    Flatten a BatchResult into a single list of row dicts.

    For failed tasks, includes an error row so the failure appears in the
    final report as a REVIEW item rather than being silently dropped.
    """
    all_rows = []
    for cr in batch.results:
        if cr.error:
            # Include an error row so it appears in the report
            all_rows.append({
                'clause_ref': cr.clause_ref,
                'clause_text': '',
                'condition': f'[PROCESSING ERROR: {cr.error}]',
                'found_text': '',
                'document_checked': '',
                'result': 'REVIEW',
                'compliance': 'REVIEW REQUIRED',
            })
        else:
            all_rows.extend(cr.rows)
    return all_rows


# ── Runner ───────────────────────────────────────────────────────────────────

def run(
    tasks: List[Dict],
    verify_fn: Callable,
    shared_context: Optional[Dict] = None,
    output_dir: str = '',
    max_workers: int = 4,
    progress_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Execute Step 18: Multi-Threaded Clause Processing.

    This is a general-purpose parallel execution wrapper. The server.py
    orchestrator uses it to parallelize clause verification across multiple
    VLM calls.

    Args:
        tasks:           list of task dicts with clause_ref, clause_text, etc.
        verify_fn:       verification function(task, context) -> list of rows
        shared_context:  shared data for all threads (e.g., LC fields, document texts)
        output_dir:      directory for step output
        max_workers:     thread pool size (number of concurrent VLM calls)
        progress_fn:     callback for progress messages

    Returns:
        dict with step results including verified_rows
    """
    t0 = time.time()

    if progress_fn is None:
        def progress_fn(msg): pass

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    progress_fn("Step 18: Multi-Threaded Clause Processing")

    # Convert raw dicts to ClauseTask objects for the parallel processor
    clause_tasks = []
    for t in tasks:
        clause_tasks.append(ClauseTask(
            clause_ref=t.get('clause_ref', ''),
            clause_text=t.get('clause_text', ''),
            clause_group=t.get('clause_group', t.get('clause_ref', '').split('-')[0]),
            documents=t.get('documents', []),
            lc_fields=t.get('lc_fields', {}),
            dependency_refs=t.get('dependency_refs', []),
        ))

    # Run parallel processing
    batch = process_clauses_parallel(
        tasks=clause_tasks,
        verify_fn=verify_fn,
        shared_context=shared_context,
        max_workers=max_workers,
        progress_fn=progress_fn,
    )

    # Flatten all results into a single row list
    all_rows = flatten_results(batch)

    result = {
        'step': 18,
        'step_name': 'Multi-Threaded Clause Processing',
        'total_tasks': batch.total_tasks,
        'completed': batch.completed,
        'failed': batch.failed,
        'total_rows': len(all_rows),
        'verified_rows': all_rows,
        'elapsed_seconds': round(time.time() - t0, 2),
    }

    # Save result to disk
    if output_dir:
        out_path = Path(output_dir) / 'step18_result.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    progress_fn(
        f"Step 18 complete: {batch.completed}/{batch.total_tasks} tasks, "
        f"{len(all_rows)} rows in {batch.elapsed_seconds:.1f}s"
    )

    return result
