"""Consolidate durable Phase E2.2 pytest evidence without double counting."""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"


def command_from(path: Path) -> str:
    first = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    return first.removeprefix("COMMAND: ")


def public_log(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    text = re.sub(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\r\n]+[\\/]Documents[\\/]dataset generate v1", "<REPOSITORY_ROOT>", text)
    text = re.sub(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\r\n]+", "<LOCAL_USER_ROOT>", text)
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="\n")


def junit_run(name: str, xml_name: str, log_name: str, exit_name: str | None, classification: str) -> tuple[dict[str, object], set[str]]:
    xml_path = LOGS / xml_name
    root = ET.parse(xml_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"MISSING_TESTSUITE:{name}")
    tests = int(suite.attrib.get("tests", 0)); failures = int(suite.attrib.get("failures", 0)); errors = int(suite.attrib.get("errors", 0)); skipped = int(suite.attrib.get("skipped", 0)); duration = float(suite.attrib.get("time", 0.0))
    started = dt.datetime.fromisoformat(suite.attrib["timestamp"]); finished = started + dt.timedelta(seconds=duration)
    source_log = LOGS / log_name; safe_log = LOGS / "public" / log_name
    suite.attrib.pop("hostname", None)
    safe_xml = LOGS / "public" / xml_name; safe_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(safe_xml, encoding="utf-8", xml_declaration=True)
    if name == "training-system-historical":
        safe_log.parent.mkdir(parents=True, exist_ok=True)
        safe_log.write_text(f"Historical wrapper was interrupted after pytest wrote complete JUnit.\nJUnit result: {tests} passed, {failures} failed, {errors} errors, {skipped} skipped in {duration:.3f}s.\nWrapper numeric exit marker: unavailable.\n", encoding="utf-8", newline="\n")
        command = "python -m pytest training/tests/test_training_system.py"
    else:
        public_log(source_log, safe_log); command = command_from(source_log)
    exit_code = int((LOGS / exit_name).read_text().strip()) if exit_name else None
    ids = {f"{case.attrib.get('classname','')}::{case.attrib.get('name','')}" for case in suite.iter("testcase") if case.find("error") is None and case.find("failure") is None}
    return ({
        "suite": name, "classification": classification, "command": command, "working_directory": "repository root",
        "python_executable": "python.exe (absolute local path withheld)", "python_version": sys.version.split()[0],
        "started_utc": started.astimezone(dt.timezone.utc).isoformat(), "finished_utc": finished.astimezone(dt.timezone.utc).isoformat(),
        "duration_seconds": duration, "test_count": tests, "passed": tests - failures - errors - skipped,
        "failed": failures, "errors": errors, "skipped": skipped, "numeric_process_exit_code": exit_code,
        "pytest_process_completion": "confirmed through complete JUnit" if exit_code is None else "confirmed through JUnit and wrapper exit marker",
        "junit_path": f"training/evaluation/e2/reason-code-remediation/logs/public/{xml_name}",
        "console_log_path": f"training/evaluation/e2/reason-code-remediation/logs/public/{log_name}",
    }, ids)


def historical_failure(name: str, log_name: str, exit_name: str) -> dict[str, object]:
    path = LOGS / log_name; text = path.read_text(encoding="utf-8-sig", errors="replace")
    summary = re.search(r"(\d+) failed, (\d+) passed in ([0-9.]+)s", text)
    if not summary:
        raise ValueError(f"MISSING_HISTORICAL_SUMMARY:{name}")
    failed, passed, duration = int(summary.group(1)), int(summary.group(2)), float(summary.group(3)); finished = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    public_log(path, LOGS / "public" / log_name)
    return {
        "suite": name, "classification": "initial run" if "initial" in name else "regression run", "command": command_from(path),
        "working_directory": "repository root", "python_executable": "python.exe (absolute local path withheld)", "python_version": sys.version.split()[0],
        "started_utc": (finished - dt.timedelta(seconds=duration)).isoformat(), "finished_utc": finished.isoformat(),
        "timestamp_source": "derived from durable log modification time and pytest-reported duration", "duration_seconds": duration,
        "test_count": failed + passed, "passed": passed, "failed": failed, "errors": 0, "skipped": 0,
        "numeric_process_exit_code": int((LOGS / exit_name).read_text().strip()), "pytest_process_completion": "confirmed through durable pytest summary and wrapper exit marker",
        "junit_path": None, "console_log_path": f"training/evaluation/e2/reason-code-remediation/logs/public/{log_name}",
    }


def main() -> int:
    runs: list[dict[str, object]] = [
        historical_failure("focused-gate-map-initial", "focused-tests-initial-failure.txt", "focused-tests-initial-failure-exit-code.txt"),
        historical_failure("focused-gate-map-regression", "focused-tests-second-failure.txt", "focused-tests-second-failure-exit-code.txt"),
    ]
    unique_ids: set[str] = set()
    specs = [
        ("focused-final", "focused-tests.xml", "focused-tests.txt", "focused-tests-exit-code.txt", "final confirmation"),
        ("focused-postscan-final", "focused-postscan-tests.xml", "focused-postscan-tests.txt", "focused-postscan-tests-exit-code.txt", "final confirmation"),
        ("focused-publication-final", "focused-publication-final-tests.xml", "focused-publication-final-tests.txt", "focused-publication-final-tests-exit-code.txt", "final confirmation"),
        ("relevant-core-final", "relevant-core-tests.xml", "relevant-core-tests.txt", "relevant-core-tests-exit-code.txt", "final confirmation"),
        ("relevant-schemas-final", "relevant-schemas-tests.xml", "relevant-schemas-tests.txt", "relevant-schemas-tests-exit-code.txt", "final confirmation"),
        ("relevant-tail-final", "relevant-tail-tests.xml", "relevant-tail-tests.txt", "relevant-tail-tests-exit-code.txt", "final confirmation"),
        ("training-system-historical", "relevant-training-system-tests.xml", "relevant-training-system-tests.txt", None, "final confirmation"),
        ("corpus-remediation-final", "corpus-tests.xml", "corpus-tests.txt", "corpus-tests-exit-code.txt", "final confirmation"),
    ]
    for spec in specs:
        item, ids = junit_run(*spec); runs.append(item); unique_ids.update(ids)
    totals = {key: sum(int(run[key]) for run in runs) for key in ("test_count", "passed", "failed", "errors", "skipped")}
    final_runs = [run for run in runs if run["classification"] == "final confirmation"]
    evidence = {
        "schema_version": "phase-e2.2-test-evidence-v2", "status": "PASS" if all(run["failed"] == 0 and run["errors"] == 0 for run in final_runs) else "FAIL",
        "execution_totals_all_completed_retained_runs": totals, "unique_passing_test_id_coverage": len(unique_ids), "runs": runs,
        "development_failures": [
            {"run": "focused-gate-map-initial", "defect": "decision gate names were not mapped to runner metric names", "remediation": "explicit decision_accuracy and decision_macro_f1 aliases", "final_status": "fixed"},
            {"run": "focused-gate-map-regression", "defect": "immediate-termination gate name was not mapped to immediate_termination_rate", "remediation": "explicit immediate_termination alias", "final_status": "fixed"},
        ],
        "incomplete_infrastructure_attempts_excluded_from_execution_totals": [
            {"log": "relevant-tests-combined-collection-error.txt", "reason": "pytest conftest namespace collision when unrelated roots were collected in one process; suites were rerun separately"},
            {"log": "relevant-tests.txt", "reason": "wrapper interruption before a complete pytest summary/JUnit update; all modules were confirmed in smaller final runs"},
        ],
        "excluded": {"modules": ["training/tests/test_live_dashboard.py"], "test_count": 7, "reason": "TensorBoard dependency unavailable in the selected local environment.", "treated_as_passing": False, "product_failure": False},
        "historical_wrapper_limitation": "test_training_system.py has a complete 14-test JUnit result (14 passed, 0 failed/errors/skipped, 7.244s); its separate numeric wrapper exit marker is unavailable because the wrapper shell was interrupted after pytest completed.",
    }
    destination = HERE / "manifests/test-evidence.json"; destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": evidence["status"], "execution_totals": totals, "unique_test_ids": len(unique_ids)}, sort_keys=True))
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
