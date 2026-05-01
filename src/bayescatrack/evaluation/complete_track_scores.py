"""Track-level scoring helpers for longitudinal ROI identity matrices."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_MISSING = object()
_MISSING_STRINGS = {"", "none", "nan", "null"}

__all__ = (
    "complete_track_set",
    "identity_switch_events",
    "normalize_track_matrix",
    "pairwise_track_set",
    "score_complete_tracks",
    "score_identity_switches",
    "score_pairwise_tracks",
    "score_track_matrices",
    "summarize_tracks",
    "track_lengths",
)


def normalize_track_matrix(track_matrix: Any) -> np.ndarray:
    """Return an object matrix containing integer ROI indices or ``None``."""

    matrix = np.asarray(track_matrix, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("track_matrix must have shape (n_tracks, n_sessions)")

    normalized = np.empty(matrix.shape, dtype=object)
    for index, value in np.ndenumerate(matrix):
        normalized[index] = _parse_optional_int(value)
    return normalized


def track_lengths(track_matrix: Any) -> np.ndarray:
    """Return the number of present sessions for each predicted track."""

    matrix = normalize_track_matrix(track_matrix)
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    present = np.vectorize(lambda value: value is not None, otypes=[bool])(matrix)
    return np.sum(present, axis=1, dtype=int)


def complete_track_set(
    track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> set[tuple[int, ...]]:
    """Return exact full-track tuples for tracks present in every selected session."""

    matrix = normalize_track_matrix(track_matrix)
    selected_sessions = _selected_sessions(matrix, session_indices)
    complete_tracks: set[tuple[int, ...]] = set()
    for row in matrix:
        values = [row[session_idx] for session_idx in selected_sessions]
        if all(value is not None for value in values):
            complete_tracks.add(tuple(int(value) for value in values))
    return complete_tracks


def pairwise_track_set(
    track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> set[tuple[int, int, int, int]]:
    """Return pairwise identity links encoded as ``(session_a, session_b, roi_a, roi_b)``."""

    matrix = normalize_track_matrix(track_matrix)
    pairs = list(_adjacent_session_pairs(matrix) if session_pairs is None else session_pairs)
    normalized_pairs: set[tuple[int, int, int, int]] = set()
    for session_a, session_b in pairs:
        _validate_session_index(matrix, session_a)
        _validate_session_index(matrix, session_b)
        if session_a >= session_b:
            raise ValueError("session_pairs must point forward in time")
        for row in matrix:
            roi_a = row[session_a]
            roi_b = row[session_b]
            if roi_a is None or roi_b is None:
                continue
            normalized_pairs.add((int(session_a), int(session_b), int(roi_a), int(roi_b)))
    return normalized_pairs


def score_complete_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score exact complete-track recovery with precision, recall, and F1."""

    predicted = complete_track_set(predicted_track_matrix, session_indices=session_indices)
    reference = complete_track_set(reference_track_matrix, session_indices=session_indices)
    return _score_identity_sets(
        predicted,
        reference,
        prefix="complete_track",
        predicted_total_name="complete_tracks",
        reference_total_name="reference_complete_tracks",
    )


def score_pairwise_tracks(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Score pairwise links induced by track matrices."""

    predicted = pairwise_track_set(predicted_track_matrix, session_pairs=session_pairs)
    reference = pairwise_track_set(reference_track_matrix, session_pairs=session_pairs)
    return _score_identity_sets(
        predicted,
        reference,
        prefix="pairwise",
        predicted_total_name="pairwise_links",
        reference_total_name="reference_pairwise_links",
    )


def identity_switch_events(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> list[dict[str, int]]:
    """Return reference-trajectory identity-switch events.

    An identity switch is counted when the same reference neuron is matched to
    one predicted track in an earlier selected session and to a different
    predicted track in a later selected session. Missed detections do not break
    the diagnostic state; the next matched prediction is compared with the last
    matched prediction for that reference trajectory.
    """

    reference, selected_sessions, prediction_lookup = _identity_switch_inputs(
        predicted_track_matrix,
        reference_track_matrix,
        session_indices=session_indices,
    )
    return _identity_switch_events_from_lookup(reference, selected_sessions, prediction_lookup)


def score_identity_switches(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Score identity-switch diagnostics induced by two track matrices."""

    reference, selected_sessions, prediction_lookup = _identity_switch_inputs(
        predicted_track_matrix,
        reference_track_matrix,
        session_indices=session_indices,
    )
    events = _identity_switch_events_from_lookup(reference, selected_sessions, prediction_lookup)
    switched_reference_tracks = {event["reference_track"] for event in events}
    matched_reference_tracks = _reference_tracks_with_predictions(reference, selected_sessions, prediction_lookup)
    reference_tracks = int(reference.shape[0])
    identity_switch_count = len(events)
    return {
        "identity_switches": identity_switch_count,
        "reference_tracks": reference_tracks,
        "reference_tracks_with_predictions": matched_reference_tracks,
        "reference_tracks_with_identity_switches": len(switched_reference_tracks),
        "identity_switches_per_reference_track": _zero_safe_ratio(identity_switch_count, reference_tracks),
        "identity_switches_per_matched_reference_track": _zero_safe_ratio(
            identity_switch_count,
            matched_reference_tracks,
        ),
    }


def summarize_tracks(track_matrix: Any) -> dict[str, float | int]:
    """Summarize the number and length of predicted tracks."""

    matrix = normalize_track_matrix(track_matrix)
    lengths = track_lengths(matrix)
    n_tracks = int(matrix.shape[0])
    mean_track_length = float(np.mean(lengths)) if lengths.size else 0.0
    return {
        "tracks": n_tracks,
        "mean_track_length": mean_track_length,
        "max_track_length": int(np.max(lengths)) if lengths.size else 0,
    }


def score_track_matrices(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_pairs: Iterable[tuple[int, int]] | None = None,
    complete_session_indices: Sequence[int] | None = None,
    identity_switch_session_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    """Return pairwise, complete-track, identity-switch, and length metrics."""

    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")

    scores: dict[str, float | int] = {}
    scores.update(score_pairwise_tracks(predicted, reference, session_pairs=session_pairs))
    scores.update(score_complete_tracks(predicted, reference, session_indices=complete_session_indices))
    scores.update(score_identity_switches(predicted, reference, session_indices=identity_switch_session_indices))
    scores.update(summarize_tracks(predicted))
    return scores


def _score_identity_sets(
    predicted: set[Any],
    reference: set[Any],
    *,
    prefix: str,
    predicted_total_name: str,
    reference_total_name: str,
) -> dict[str, float | int]:
    true_positives, false_positives, false_negatives = _confusion_counts(predicted, reference)
    precision, recall, f1 = _precision_recall_f1(true_positives, false_positives, false_negatives)
    return {
        f"{prefix}_true_positives": true_positives,
        f"{prefix}_false_positives": false_positives,
        f"{prefix}_false_negatives": false_negatives,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        predicted_total_name: len(predicted),
        reference_total_name: len(reference),
    }


def _confusion_counts(predicted: set[Any], reference: set[Any]) -> tuple[int, int, int]:
    return (
        len(predicted.intersection(reference)),
        len(predicted.difference(reference)),
        len(reference.difference(predicted)),
    )


def _precision_recall_f1(true_positives: int, false_positives: int, false_negatives: int) -> tuple[float, float, float]:
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _parse_optional_int(value: Any) -> int | None:
    candidate = _optional_int_candidate(value)
    if candidate is _MISSING:
        return None
    try:
        parsed = int(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_int_candidate(value: Any) -> Any:
    if value is None:
        return _MISSING
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        stripped = value.strip()
        return _MISSING if stripped.casefold() in _MISSING_STRINGS else stripped
    if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
        return _MISSING
    return value


def _selected_sessions(matrix: np.ndarray, session_indices: Sequence[int] | None) -> list[int]:
    if session_indices is None:
        return list(range(matrix.shape[1]))
    selected = [int(session_idx) for session_idx in session_indices]
    for session_idx in selected:
        _validate_session_index(matrix, session_idx)
    return selected


def _adjacent_session_pairs(matrix: np.ndarray) -> list[tuple[int, int]]:
    return [(session_idx, session_idx + 1) for session_idx in range(max(0, matrix.shape[1] - 1))]


def _validate_session_index(matrix: np.ndarray, session_idx: int) -> None:
    if session_idx < 0 or session_idx >= matrix.shape[1]:
        raise IndexError(f"session index {session_idx} out of bounds for {matrix.shape[1]} sessions")


def _identity_switch_inputs(
    predicted_track_matrix: Any,
    reference_track_matrix: Any,
    *,
    session_indices: Sequence[int] | None,
) -> tuple[np.ndarray, list[int], list[dict[int, int]]]:
    predicted = normalize_track_matrix(predicted_track_matrix)
    reference = normalize_track_matrix(reference_track_matrix)
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError("Predicted and reference matrices must have the same number of sessions")
    selected_sessions = _selected_sessions(reference, session_indices)
    prediction_lookup = _session_roi_to_track_map(predicted, matrix_name="predicted")
    return reference, selected_sessions, prediction_lookup


def _identity_switch_events_from_lookup(
    reference: np.ndarray,
    selected_sessions: Sequence[int],
    prediction_lookup: Sequence[dict[int, int]],
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    for reference_track_idx, reference_row in enumerate(reference):
        previous_session: int | None = None
        previous_predicted_track: int | None = None
        previous_roi: int | None = None
        for session_idx in selected_sessions:
            reference_roi = reference_row[session_idx]
            if reference_roi is None:
                continue
            roi = int(reference_roi)
            predicted_track_idx = prediction_lookup[session_idx].get(roi)
            if predicted_track_idx is None:
                continue

            current_session = int(session_idx)
            current_predicted_track = int(predicted_track_idx)
            if previous_predicted_track is not None and previous_predicted_track != current_predicted_track:
                assert previous_session is not None
                assert previous_roi is not None
                events.append(
                    {
                        "reference_track": int(reference_track_idx),
                        "previous_session": previous_session,
                        "session": current_session,
                        "previous_predicted_track": previous_predicted_track,
                        "predicted_track": current_predicted_track,
                        "previous_roi": previous_roi,
                        "roi": roi,
                    }
                )
            previous_session = current_session
            previous_predicted_track = current_predicted_track
            previous_roi = roi
    return events


def _reference_tracks_with_predictions(
    reference: np.ndarray,
    selected_sessions: Sequence[int],
    prediction_lookup: Sequence[dict[int, int]],
) -> int:
    matched = 0
    for reference_row in reference:
        if any(
            reference_row[session_idx] is not None
            and int(reference_row[session_idx]) in prediction_lookup[session_idx]
            for session_idx in selected_sessions
        ):
            matched += 1
    return matched


def _session_roi_to_track_map(matrix: np.ndarray, *, matrix_name: str) -> list[dict[int, int]]:
    lookup: list[dict[int, int]] = [{} for _ in range(matrix.shape[1])]
    for track_idx, row in enumerate(matrix):
        for session_idx, roi in enumerate(row):
            if roi is None:
                continue
            roi_int = int(roi)
            if roi_int in lookup[session_idx]:
                raise ValueError(
                    f"{matrix_name} track matrix contains ROI {roi_int} "
                    f"more than once in session {session_idx}"
                )
            lookup[session_idx][roi_int] = int(track_idx)
    return lookup


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


def _zero_safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
