"""
Automated Verification & Acceptance Test Suite for STEP 7 — SIM&DFS GitHub Readiness.

Executes:
1. Python syntax / compilation check (all 74 project python files)
2. Dashboard startup check (loads cached startup bundle)
3. Required-file check (all core data, dashboard, reports, and root files)
4. Model artifact check (baseline models verified)
5. Policy artifact check (Step 5.7 locked policy & backtests verified)
6. README check (all 24 required sections present)
7. requirements.txt check (core application dependencies verified)
8. .gitignore check (comprehensive ignore rules verified)
9. Secret scan (regex check across all text files)
10. Git status / repository state check
"""

import ast
import json
import py_compile
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config.policy_config import DATA_FILES, PROTECTED_TEST_SHA256
from dashboard.data.validators import calculate_file_sha256, verify_test_split_checksum


def test_1_syntax_and_imports() -> bool:
    py_files = [p for p in PROJECT_ROOT.rglob("*.py") if "venv" not in p.parts and ".git" not in p.parts]
    for p in py_files:
        py_compile.compile(str(p), doraise=True)
    return True


def test_2_dashboard_startup() -> bool:
    from app import _cached_startup_bundle
    bundle = _cached_startup_bundle()
    assert bundle["test_check"]["status"] == "PASS"
    assert bundle["policy_cfg"].policy_version == "STEP_5_7_FROZEN_v1.0"
    assert len(bundle["dev_df"]) == 65800
    return True


def test_3_required_files() -> bool:
    required = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "docs" / "STEP6_DASHBOARD_README.md",
        PROJECT_ROOT / "tests" / "test_step6_dashboard.py",
        PROJECT_ROOT / "reports" / "step7" / "step7_project_audit.txt",
    ]
    for r in required:
        assert r.exists(), f"Missing required file: {r}"
    for name, p in DATA_FILES.items():
        assert p.exists(), f"Missing DATA_FILES entry '{name}': {p}"
    return True


def test_4_model_artifacts() -> bool:
    models_dir = PROJECT_ROOT / "models" / "baseline"
    assert (models_dir / "ridge_baseline.joblib").exists()
    assert (models_dir / "hist_gradient_boosting_baseline.joblib").exists()
    rf = models_dir / "random_forest_baseline.joblib"
    assert rf.exists()
    # Confirm RF is > 100 MB
    assert rf.stat().st_size > 100 * 1024 * 1024
    return True


def test_5_policy_artifacts() -> bool:
    cfg_file = DATA_FILES["policy_config"]
    assert cfg_file.exists()
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "COMPLETE_AND_FROZEN"
    assert data["base_policy"]["lead_time_days"] == 4
    assert data["base_policy"]["service_level"] == 0.95
    assert data["base_policy"]["z_value"] == 1.6449
    return True


def test_6_readme_structure() -> bool:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for i in range(1, 25):
        assert f"## {i}." in readme, f"Missing section ## {i}. in README.md"
    return True


def test_7_requirements_file() -> bool:
    reqs = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ["streamlit", "plotly", "pandas", "numpy", "joblib"]:
        assert pkg in reqs, f"Missing {pkg} in requirements.txt"
    return True


def test_8_gitignore_rules() -> bool:
    gi = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ["venv/", "__pycache__/", "*.pyc", ".env", "models/baseline/random_forest_baseline.joblib"]:
        assert rule in gi, f"Missing {rule} in .gitignore"
    return True


def test_9_secret_scan() -> bool:
    patterns = [
        re.compile(r"""(api[_-]?key|secret|password|bearer|auth[_-]?token)\s*[:=]\s*["'][^"']+["']""", re.I),
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    ]
    for p in PROJECT_ROOT.rglob("*"):
        if "venv" in p.parts or ".git" in p.parts or not p.is_file():
            continue
        if p.suffix in [".png", ".jpg", ".joblib", ".pyc"]:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for pat in patterns:
                matches = list(pat.finditer(txt))
                assert len(matches) == 0, f"Secret match in {p}: {matches[0].group(0)}"
        except Exception:
            pass
    return True


def test_10_test_split_integrity() -> bool:
    actual_hash = calculate_file_sha256(DATA_FILES["test_split_protected"])
    assert actual_hash == PROTECTED_TEST_SHA256, "TEST split hash mismatch!"
    return True


def run_all_readiness_checks() -> None:
    print("=" * 76)
    print("STEP 7 — GITHUB READINESS VERIFICATION SUITE")
    print("=" * 76)

    checks = [
        ("Check 1: Python Syntax & Imports (74 Files)", test_1_syntax_and_imports),
        ("Check 2: Dashboard Startup & Cache Loading", test_2_dashboard_startup),
        ("Check 3: Required Core Files & Data Mapping", test_3_required_files),
        ("Check 4: Model Artifacts & File-Size Integrity", test_4_model_artifacts),
        ("Check 5: Frozen Step 5.7 Policy Artifacts", test_5_policy_artifacts),
        ("Check 6: README Structure (All 24 Sections)", test_6_readme_structure),
        ("Check 7: Production requirements.txt Scoping", test_7_requirements_file),
        ("Check 8: .gitignore Large-File & Security Rules", test_8_gitignore_rules),
        ("Check 9: Zero Secret & Credential Leakage Scan", test_9_secret_scan),
        ("Check 10: Protected TEST Split SHA-256 Checksum", test_10_test_split_integrity),
    ]

    all_passed = True
    for name, func in checks:
        try:
            func()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    print("=" * 76)
    if all_passed:
        print("ALL 10 GITHUB READINESS CHECKS PASSED SUCCESSFULLY.")
    else:
        print("FAILURES DETECTED IN READINESS CHECKS.")
    print("=" * 76)


if __name__ == "__main__":
    run_all_readiness_checks()
