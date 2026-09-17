from __future__ import annotations

from scripts.run_m0_demo import run_demo


def test_m0_deterministic_demo_runs_success_rejection_and_recovery_without_provider_calls():
    result = run_demo()

    assert result["suite"] == "m0-deterministic-demo-v1"
    assert result["model_calls"] == 0
    assert result["passed"] == 3
    assert result["total"] == 3
    assert result["scenarios"] == [
        {
            "scenario": "success_read_edit_validate",
            "passed": True,
            "provider_requests": 4,
            "validated_file": "app.py",
        },
        {
            "scenario": "rejected_write_stops_turn",
            "passed": True,
            "provider_requests": 1,
            "target_unchanged": True,
        },
        {
            "scenario": "long_task_recovery_skips_completed_effect",
            "passed": True,
            "initial_tool_calls": 1,
            "resumed_tool_calls": 0,
        },
    ]
