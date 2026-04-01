"""
NexusBot Smoke Test Suite
=========================
Tests all critical flows:
  - Greeting detection (EN / HI / MR)
  - Gibberish detection
  - Confidence threshold fallback
  - Language output correctness (Marathi, Hindi, English)
  - Number preservation in responses
  - Correct chunk retrieval (no wrong-chunk mismatches)

Run:
    python smoke_test.py
    python smoke_test.py --url http://localhost:5000   # custom backend URL
"""

import sys
import json
import time
import argparse
import requests
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_URL = "http://localhost:5000/api/chat"

COLORS = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


# ─────────────────────────────────────────────
# TEST CASE DEFINITION
# ─────────────────────────────────────────────
@dataclass
class TestCase:
    id: str
    description: str
    query: str
    lang: str                          # "en" | "hi" | "mr"
    expect_type: str                   # "greeting" | "gibberish" | "fallback" | "answer"
    must_contain: list[str] = field(default_factory=list)   # substrings that MUST appear
    must_not_contain: list[str] = field(default_factory=list)  # substrings that must NOT appear
    wrong_topics: list[str] = field(default_factory=list)   # topics that are wrong answers
    expect_lang_script: Optional[str] = None  # "devanagari" | "latin" | None


# ─────────────────────────────────────────────
# ALL TEST CASES
# ─────────────────────────────────────────────
TEST_CASES: list[TestCase] = [

    # ── GREETING TESTS ──────────────────────────────────────────
    TestCase(
        id="G01",
        description="English greeting 'hi' → greeting response in English",
        query="hi",
        lang="en",
        expect_type="greeting",
        must_contain=["NexusBot", "ask"],
        must_not_contain=["examination", "semester", "attendance", "placement package"],
        wrong_topics=["exam schedule", "hostel rules", "fee structure"],
    ),
    TestCase(
        id="G02",
        description="Marathi mode greeting 'hi' → greeting response in Marathi",
        query="hi",
        lang="mr",
        expect_type="greeting",
        must_not_contain=["examination", "semester", "End-semester"],
        expect_lang_script="devanagari",
    ),
    TestCase(
        id="G03",
        description="Hindi greeting 'नमस्ते' → greeting in Hindi",
        query="नमस्ते",
        lang="hi",
        expect_type="greeting",
        must_not_contain=["examination", "semester"],
        expect_lang_script="devanagari",
    ),
    TestCase(
        id="G04",
        description="Casual greeting 'hello' in English",
        query="hello",
        lang="en",
        expect_type="greeting",
        must_contain=["NexusBot"],
        must_not_contain=["examination", "semester"],
    ),

    # ── GIBBERISH TESTS ─────────────────────────────────────────
    TestCase(
        id="GB01",
        description="Random keysmash 'abcefbsfgaef' → didn't understand",
        query="abcefbsfgaef",
        lang="mr",
        expect_type="gibberish",
        must_not_contain=["examination", "semester", "End-semester"],
        expect_lang_script="devanagari",
    ),
    TestCase(
        id="GB02",
        description="Random keysmash in English mode",
        query="xyzqwlmnprt",
        lang="en",
        expect_type="gibberish",
        must_contain=["understand", "rephrase"],
        must_not_contain=["examination", "semester"],
    ),
    TestCase(
        id="GB03",
        description="Single character input '?' → gibberish/fallback",
        query="?",
        lang="en",
        expect_type="gibberish",
        must_not_contain=["examination"],
    ),
    TestCase(
        id="GB04",
        description="Pure numbers '12345' → gibberish/fallback",
        query="12345",
        lang="en",
        expect_type="gibberish",
        must_not_contain=["examination", "semester"],
    ),


    # ── NUMBER PRESERVATION TESTS ────────────────────────────────
    TestCase(
        id="N01",
        description="Placement package query → numbers must be preserved in response",
        query="What is the average placement package?",
        lang="en",
        expect_type="answer",
        must_contain=["LPA", "₹"],
        must_not_contain=["₹LPA", "₹ LPA"],   # broken number stripping
    ),
    TestCase(
        id="N02",
        description="Placement in Hindi → numbers preserved after translation",
        query="औसत प्लेसमेंट पैकेज क्या है?",
        lang="hi",
        expect_type="answer",
        must_contain=["6.2"],
        must_not_contain=["₹LPA"],
        expect_lang_script="devanagari",
    ),
    TestCase(
        id="N03",
        description="Placement in Marathi mode → numbers preserved in Marathi response",
        query="average placement package",
        lang="mr",
        expect_type="answer",
        must_contain=["LPA"],
        must_not_contain=["₹LPA"],
        expect_lang_script="devanagari",
    ),

    # ── WRONG CHUNK / MISMATCH TESTS ────────────────────────────
    TestCase(
        id="M01",
        description="'gym' query → must NOT return health centre info",
        query="are there gym in campus?",
        lang="en",
        expect_type="answer",
        wrong_topics=["health centre", "sick room", "doctor on duty", "first aid"],
    ),
    TestCase(
        id="M02",
        description="'library' query → must NOT return hostel info",
        query="what are the library timings?",
        lang="en",
        expect_type="answer",
        wrong_topics=["hostel", "mess", "warden"],
    ),

    # ── MARATHI LANGUAGE OUTPUT TESTS ───────────────────────────
    TestCase(
        id="MR01",
        description="Fee structure in Marathi → response must be in Marathi script",
        query="fee structure",
        lang="mr",
        expect_type="answer",
        expect_lang_script="devanagari",
        must_not_contain=["The fee", "tuition fee is"],
    ),
    TestCase(
        id="MR02",
        description="Hostel info in Marathi → Marathi script response",
        query="hostel information",
        lang="mr",
        expect_type="answer",
        expect_lang_script="devanagari",
    ),
    TestCase(
        id="MR03",
        description="Exam schedule in Marathi → Marathi script response",
        query="exam schedule",
        lang="mr",
        expect_type="answer",
        expect_lang_script="devanagari",
        must_not_contain=["End-semester examinations are held"],  # raw English must not appear
    ),

    # ── CONFIDENT CORRECT ANSWER TESTS ──────────────────────────
    TestCase(
        id="A01",
        description="Fee structure in English → relevant answer",
        query="What is the fee structure?",
        lang="en",
        expect_type="answer",
        must_contain=["fee"],
        wrong_topics=["hostel warden", "exam schedule", "placement"],
    ),
    TestCase(
        id="A02",
        description="Scholarship query in English → relevant answer",
        query="How can I apply for scholarship?",
        lang="en",
        expect_type="answer",
        must_contain=["scholar"],
        wrong_topics=["gym", "hostel mess", "exam date"],
    ),
    TestCase(
        id="A03",
        description="Hostel query in Hindi → Hindi response",
        query="छात्रावास के नियम क्या हैं?",
        lang="hi",
        expect_type="answer",
        expect_lang_script="devanagari",
        wrong_topics=["placement", "scholarship"],
    ),
]


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def has_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script characters."""
    return any('\u0900' <= ch <= '\u097F' for ch in text)


def has_latin(text: str) -> bool:
    return any('a' <= ch.lower() <= 'z' for ch in text)


def check_lang_script(response: str, expected: Optional[str]) -> tuple[bool, str]:
    if expected is None:
        return True, ""
    if expected == "devanagari":
        if has_devanagari(response):
            return True, ""
        return False, "Expected Devanagari script but got Latin/English text"
    if expected == "latin":
        if has_latin(response) and not has_devanagari(response):
            return True, ""
        return False, "Expected Latin script but got Devanagari"
    return True, ""


def call_api(url: str, query: str, lang: str, timeout: int = 10) -> dict:
    payload = {"message": query, "language": lang}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# SINGLE TEST RUNNER
# ─────────────────────────────────────────────
def run_test(tc: TestCase, url: str) -> dict:
    result = {
        "id": tc.id,
        "description": tc.description,
        "query": tc.query,
        "lang": tc.lang,
        "passed": False,
        "failures": [],
        "response": "",
        "latency_ms": 0,
    }

    # Call API
    try:
        t0 = time.time()
        data = call_api(url, tc.query, tc.lang)
        result["latency_ms"] = int((time.time() - t0) * 1000)
        response_text = data.get("response", "")
        result["response"] = response_text
    except requests.exceptions.ConnectionError:
        result["failures"].append("CONNECTION ERROR: Is the backend running?")
        return result
    except Exception as e:
        result["failures"].append(f"API ERROR: {e}")
        return result

    r = response_text.lower()

    # Check must_contain
    for keyword in tc.must_contain:
        if keyword.lower() not in r and keyword not in response_text:
            result["failures"].append(f"MISSING expected keyword: '{keyword}'")

    # Check must_not_contain
    for keyword in tc.must_not_contain:
        if keyword.lower() in r:
            result["failures"].append(f"FOUND forbidden keyword: '{keyword}'")

    # Check wrong_topics
    for topic in tc.wrong_topics:
        if topic.lower() in r:
            result["failures"].append(f"WRONG TOPIC returned: '{topic}'")

    # Check language script
    script_ok, script_msg = check_lang_script(response_text, tc.expect_lang_script)
    if not script_ok:
        result["failures"].append(script_msg)

    # Check empty response
    if not response_text.strip():
        result["failures"].append("EMPTY response returned")

    result["passed"] = len(result["failures"]) == 0
    return result


# ─────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────
def print_result(r: dict):
    status = c("green", "✅ PASS") if r["passed"] else c("red", "❌ FAIL")
    print(f"\n{c('bold', r['id'])} [{r['lang'].upper()}] {status}")
    print(f"  {c('cyan', r['description'])}")
    print(f"  Query    : \"{r['query']}\"")
    print(f"  Latency  : {r['latency_ms']}ms")

    # Truncate long responses
    resp_preview = r["response"][:180].replace("\n", " ")
    if len(r["response"]) > 180:
        resp_preview += "..."
    print(f"  Response : {resp_preview}")

    if r["failures"]:
        for f in r["failures"]:
            print(f"  {c('red', '→')} {f}")


def print_summary(results: list[dict]):
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "═" * 60)
    print(c("bold", "SMOKE TEST SUMMARY"))
    print("═" * 60)

    # Group by category
    categories = {
        "Greeting":    [r for r in results if r["id"].startswith("G")],
        "Gibberish":   [r for r in results if r["id"].startswith("GB")],
        "Numbers":     [r for r in results if r["id"].startswith("N")],
        "Mismatch":    [r for r in results if r["id"].startswith("M")],
        "Marathi":     [r for r in results if r["id"].startswith("MR")],
        "Answers":     [r for r in results if r["id"].startswith("A")],
    }

    for cat, items in categories.items():
        if not items:
            continue
        cat_pass = sum(1 for i in items if i["passed"])
        bar = c("green", "●" * cat_pass) + c("red", "●" * (len(items) - cat_pass))
        print(f"  {cat:<12} {bar}  {cat_pass}/{len(items)}")

    print()
    if failed == 0:
        print(c("green", f"  ALL {total} TESTS PASSED 🎉"))
    else:
        print(c("red",   f"  {failed}/{total} TESTS FAILED"))
        print(c("yellow", "\n  Failed tests:"))
        for r in results:
            if not r["passed"]:
                print(f"    • {r['id']}: {r['description']}")
                for f in r["failures"]:
                    print(f"      → {f}")

    print("═" * 60)

    # Save JSON report
    report_path = "smoke_test_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n  📄 Full report saved to: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NexusBot Smoke Test Suite")
    parser.add_argument("--url",      default=DEFAULT_URL, help="Backend API URL")
    parser.add_argument("--filter",   default=None,        help="Run only tests matching this ID prefix (e.g. G, MR, N)")
    parser.add_argument("--verbose",  action="store_true", help="Show full response text")
    args = parser.parse_args()

    tests = TEST_CASES
    if args.filter:
        tests = [t for t in TEST_CASES if t.id.startswith(args.filter.upper())]
        if not tests:
            print(c("red", f"No tests match filter '{args.filter}'"))
            sys.exit(1)

    print(c("bold", f"\n🤖 NexusBot Smoke Test — {len(tests)} tests → {args.url}\n"))
    print("─" * 60)

    results = []
    for tc in tests:
        r = run_test(tc, args.url)
        print_result(r)
        if args.verbose:
            print(f"  Full response:\n  {r['response']}\n")
        results.append(r)
        time.sleep(0.3)  # be gentle to the backend

    print_summary(results)

    # Exit with non-zero code if any test failed (useful for CI)
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()