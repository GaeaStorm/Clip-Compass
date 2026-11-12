"""Score each judge on control hypotheses whose right answer is known.

Judges:

* ``agent``: believes every hypothesis (the unverified baseline; accepts all).
* ``text_check``: ``verification baseline`` accepts the hypothesis.
* ``masking``: ``verification verdicts`` labels it ``supported``.
* ``masking_lenient``: also counts ``detected_not_explanatory`` as accepted.

For each judge we report the hit rate (accepted controls that should be accepted)
and the false-alarm rate (accepted controls that should be rejected), overall and
per control kind. Masking verdicts that could not be decided
(``insufficient_evidence`` or ``untestable``) count as not accepted and are also
tallied separately, so a judge cannot look good by abstaining unnoticed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .controls import EXPECTED, load_control_key
from .types import VerdictArtifact, _atomic_csv, _atomic_json

EVALUATION_SCHEMA_VERSION = 1
JUDGES = ("agent", "text_check", "masking", "masking_lenient")
UNDECIDED = ("insufficient_evidence", "untestable")


@dataclass(frozen=True)
class JudgeScore:
    judge: str
    kind: str
    expected: str
    total: int
    accepted: int
    undecided: int

    @property
    def correct(self) -> int:
        return self.accepted if self.expected == "accept" else self.total - self.accepted

    def as_dict(self) -> Dict[str, object]:
        return {
            "judge": self.judge,
            "kind": self.kind,
            "expected": self.expected,
            "total": self.total,
            "accepted": self.accepted,
            "undecided": self.undecided,
            "correct": self.correct,
            "accept_rate": self.accepted / self.total if self.total else None,
        }


def _decisions(
    key_ids: List[str],
    verdicts: Optional[VerdictArtifact],
    baseline: Optional[Mapping[str, object]],
) -> Dict[str, Dict[str, Tuple[bool, bool]]]:
    """judge -> hypothesis_id -> (accepted, undecided)."""

    decisions: Dict[str, Dict[str, Tuple[bool, bool]]] = {
        "agent": {h: (True, False) for h in key_ids}
    }
    if baseline is not None:
        accepted = {
            f"{score['node_id']}-h{int(score['hypothesis_index']):02d}": bool(score["accepted"])
            for score in baseline["scores"]
        }
        decisions["text_check"] = {h: (accepted[h], False) for h in key_ids if h in accepted}
    if verdicts is not None:
        labels = {verdict.hypothesis_id: verdict.label for verdict in verdicts.hypotheses}
        decisions["masking"] = {
            h: (labels[h] == "supported", labels[h] in UNDECIDED) for h in key_ids if h in labels
        }
        decisions["masking_lenient"] = {
            h: (labels[h] in ("supported", "detected_not_explanatory"), labels[h] in UNDECIDED)
            for h in key_ids
            if h in labels
        }
    return decisions


def evaluate_judges(
    key_path: Path,
    *,
    verdicts_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Compare judges against the control answer key; inputs must share one controls file."""

    explanations_sha256, keys = load_control_key(key_path)
    verdicts = VerdictArtifact.load(verdicts_path) if verdicts_path else None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path else None
    if verdicts is not None and verdicts.explanations_sha256 != explanations_sha256:
        raise ValueError("verdicts were computed for a different controls file")
    if baseline is not None and baseline["explanations_sha256"] != explanations_sha256:
        raise ValueError("baseline was scored for a different controls file")

    key_ids = [key.hypothesis_id for key in keys]
    decisions = _decisions(key_ids, verdicts, baseline)
    scores: List[JudgeScore] = []
    rows: List[Dict[str, object]] = []
    for judge in JUDGES:
        if judge not in decisions:
            continue
        judged = decisions[judge]
        missing = [h for h in key_ids if h not in judged]
        if missing:
            raise ValueError(f"{judge} has no decision for control {missing[0]}")
        for kind, expected in EXPECTED.items():
            ids = [key.hypothesis_id for key in keys if key.kind == kind]
            scores.append(
                JudgeScore(
                    judge,
                    kind,
                    expected,
                    len(ids),
                    sum(judged[h][0] for h in ids),
                    sum(judged[h][1] for h in ids),
                )
            )
    labels = (
        {verdict.hypothesis_id: verdict.label for verdict in verdicts.hypotheses}
        if verdicts
        else {}
    )
    for key in keys:
        row: Dict[str, object] = dict(key.as_dict())
        row["categories"] = ";".join(key.categories)
        for judge in JUDGES:
            if judge in decisions:
                row[judge] = "accept" if decisions[judge][key.hypothesis_id][0] else "reject"
        row["masking_label"] = labels.get(key.hypothesis_id)
        rows.append(row)

    summary = {}
    for judge in JUDGES:
        if judge not in decisions:
            continue
        mine = [score for score in scores if score.judge == judge]
        positives = [score for score in mine if score.expected == "accept"]
        negatives = [score for score in mine if score.expected == "reject"]
        pos_total, neg_total = sum(s.total for s in positives), sum(s.total for s in negatives)
        pos_hits, neg_hits = sum(s.accepted for s in positives), sum(s.accepted for s in negatives)
        # Undecided controls are never accepted, so abstaining can look like correctly
        # rejecting a negative. The *_decided rates leave undecided controls out.
        pos_decided = pos_total - sum(s.undecided for s in positives)
        neg_decided = neg_total - sum(s.undecided for s in negatives)
        summary[judge] = {
            "hit_rate": _ratio(pos_hits, pos_total),
            "false_alarm_rate": _ratio(neg_hits, neg_total),
            "hit_rate_decided": _ratio(pos_hits, pos_decided),
            "false_alarm_rate_decided": _ratio(neg_hits, neg_decided),
            "accuracy": _ratio(sum(s.correct for s in mine), pos_total + neg_total),
            "undecided": sum(s.undecided for s in mine),
            "positives": pos_total,
            "negatives": neg_total,
        }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "explanations_sha256": explanations_sha256,
        "summary": summary,
        "scores": [score.as_dict() for score in scores],
        "controls": rows,
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def save_evaluation(evaluation: Mapping[str, object], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": _atomic_json(output_dir / "evaluation.json", evaluation),
        "scores_csv": _atomic_csv(output_dir / "judge_scores.csv", list(evaluation["scores"])),
        "controls_csv": _atomic_csv(output_dir / "controls.csv", list(evaluation["controls"])),
    }


def format_evaluation(evaluation: Mapping[str, object]) -> str:
    """A small plain-text table: one row per judge."""

    def pct(value: Optional[float]) -> str:
        return "-" if value is None else f"{value:.0%}"

    lines = [
        f"{'judge':<16}{'hit rate':>10}{'false alarms':>14}{'accuracy':>10}{'undecided':>11}"
        f"{'decided: hits':>15}{'false alarms':>14}"
    ]
    for judge, row in evaluation["summary"].items():
        lines.append(
            f"{judge:<16}{pct(row['hit_rate']):>10}{pct(row['false_alarm_rate']):>14}"
            f"{pct(row['accuracy']):>10}{row['undecided']:>11}"
            f"{pct(row['hit_rate_decided']):>15}{pct(row['false_alarm_rate_decided']):>14}"
        )
    return "\n".join(lines)
