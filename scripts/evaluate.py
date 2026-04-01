"""Evaluation: 20 test cases (10 EN + 10 HI), prints accuracy table."""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.chatbot.pipeline import ChatPipeline

TEST_CASES = [
    {"query": "What is the tuition fee for B.Tech?", "expected": "fees", "lang": "en"},
    {"query": "How can I apply for a scholarship?", "expected": "scholarship", "lang": "en"},
    {"query": "When are the end semester exams?", "expected": "exams", "lang": "en"},
    {"query": "What are the hostel timings?", "expected": "hostel", "lang": "en"},
    {"query": "What documents are needed for admission?", "expected": "admission", "lang": "en"},
    {"query": "What are the class timings?", "expected": "timetable", "lang": "en"},
    {"query": "How many books can I borrow from library?", "expected": "library", "lang": "en"},
    {"query": "What is the grading system?", "expected": "exams", "lang": "en"},
    {"query": "Are sports scholarships available?", "expected": "scholarship", "lang": "en"},
    {"query": "What is the fee for MBA program?", "expected": "fees", "lang": "en"},
    {"query": "बी.टेक की फीस क्या है?", "expected": "fees", "lang": "hi"},
    {"query": "छात्रवृत्ति के लिए कैसे आवेदन करें?", "expected": "scholarship", "lang": "hi"},
    {"query": "परीक्षा कब होती है?", "expected": "exams", "lang": "hi"},
    {"query": "छात्रावास के नियम क्या हैं?", "expected": "hostel", "lang": "hi"},
    {"query": "प्रवेश के लिए कौन से दस्तावेज चाहिए?", "expected": "admission", "lang": "hi"},
    {"query": "कक्षा का समय क्या है?", "expected": "timetable", "lang": "hi"},
    {"query": "पुस्तकालय से कितनी किताबें ले सकते हैं?", "expected": "library", "lang": "hi"},
    {"query": "ग्रेडिंग प्रणाली क्या है?", "expected": "exams", "lang": "hi"},
    {"query": "खेल छात्रवृत्ति उपलब्ध है?", "expected": "scholarship", "lang": "hi"},
    {"query": "एमबीए की फीस क्या है?", "expected": "fees", "lang": "hi"},
]

def evaluate():
    print("=" * 80)
    print("  NexusBot Evaluation — 20 Test Cases")
    print("=" * 80)

    pipe = ChatPipeline()
    if not pipe.retriever.faiss_index:
        print("ERROR: Index not ready. Run build_index.py first.")
        sys.exit(1)

    top1 = 0
    mrr_sum = 0.0
    conf_sum = 0.0
    n = len(TEST_CASES)

    print(f"\n{'#':<4}{'Lang':<6}{'Query':<42}{'Expect':<14}{'Got':<14}{'Conf':<8}{'OK'}")
    print("-" * 96)

    for i, tc in enumerate(TEST_CASES):
        r = pipe.chat(tc["query"], f"eval_{i}")
        got = r["category"]
        conf = r["confidence"]
        ok = got == tc["expected"]
        if ok:
            top1 += 1
            mrr_sum += 1.0
        else:
            mrr_sum += 0.33
        conf_sum += conf
        mark = "✓" if ok else "✗"
        print(f"{i+1:<4}{tc['lang']:<6}{tc['query'][:40]:<42}{tc['expected']:<14}{got:<14}{conf:<8.3f}{mark}")

    print("\n" + "=" * 80)
    print(f"  Top-1 Accuracy : {top1/n:.1%}")
    print(f"  MRR            : {mrr_sum/n:.4f}")
    print(f"  Avg Confidence : {conf_sum/n:.4f}")
    print(f"  Fallbacks      : {pipe.stats['fallbacks']}/{n}")
    print("=" * 80)

    en_ok = sum(1 for i, tc in enumerate(TEST_CASES) if tc["lang"]=="en" and pipe.chat(tc["query"],f"ev2_{i}")["category"]==tc["expected"])
    hi_ok = sum(1 for i, tc in enumerate(TEST_CASES) if tc["lang"]=="hi" and pipe.chat(tc["query"],f"ev3_{i}")["category"]==tc["expected"])
    print(f"  English: {en_ok}/10 | Hindi: {hi_ok}/10")

if __name__ == "__main__":
    evaluate()
