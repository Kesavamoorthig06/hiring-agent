"""Resume -> pipeline -> Gemini suggestions, packaged for the web server.

Reuses the repo's own PDFHandler (section extraction), ResumeEvaluator (rubric
scoring) and github helpers. Section extraction runs in parallel so one upload
takes a few LLM round-trips of wall time instead of six in a row.
"""

import json
import logging
import os
import tempfile
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from config import DEFAULT_MODEL, MODEL_PARAMETERS
from evaluator import ResumeEvaluator
from github import fetch_and_display_github_info
from llm_utils import extract_json_from_response, initialize_llm_provider
from models import Basics, JSONResume, build_evaluation_model
from pdf import PDFHandler
from roles import load_role
from transform import convert_github_data_to_text, convert_json_resume_to_text

logger = logging.getLogger(__name__)

ROLE_NAME = os.getenv("HIRING_ROLE", "software_engineering_intern")
SECTIONS = ["basics", "work", "education", "skills", "projects", "awards"]

Progress = Callable[[str, str], None]


class Priority(BaseModel):
    title: str = Field(description="Short action, e.g. 'Quantify project impact'")
    why: str = Field(description="Why this matters to a reviewer, one or two sentences")
    how: str = Field(description="Concrete steps the candidate can take")
    impact: str = Field(description="One of: high, medium, low")
    section: str = Field(description="Resume section this applies to")


class Rewrite(BaseModel):
    before: str = Field(description="A weak line copied from the resume")
    after: str = Field(description="A stronger rewrite that adds no invented facts")


class Suggestions(BaseModel):
    headline: str = Field(description="One sentence overall read of the resume")
    priorities: List[Priority] = Field(min_length=3, max_length=7)
    rewrites: List[Rewrite] = Field(max_length=4)
    missing_keywords: List[str] = Field(max_length=10)


SUGGEST_SYSTEM = (
    "You are a candid, kind resume coach for software engineering candidates. "
    "Use only facts present in the resume and the rubric evaluation. Never invent "
    "employers, metrics, or technologies. If a rewrite would need a number the "
    "candidate has not given, write a placeholder like [X%]. Be specific."
)

CHAT_SYSTEM = (
    "You are the Hiring Agent coach, chatting with a candidate about their resume. "
    "Ground every answer in the resume text and evaluation below. Be conversational, "
    "concise (under 180 words unless asked for more), and concrete. Never invent "
    "facts about the candidate; if you need a detail, ask for it. Use short "
    "paragraphs or a few bullets, no headings.\n\n"
)


def _chat_json(provider, messages, schema=None) -> dict:
    params = dict(
        MODEL_PARAMETERS.get(DEFAULT_MODEL, {"temperature": 0.2, "top_p": 0.9})
    )
    kwargs = {"format": schema} if schema else {}
    resp = provider.chat(
        model=DEFAULT_MODEL,
        messages=messages,
        options={"stream": False, **params},
        **kwargs
    )
    text = extract_json_from_response(resp["message"]["content"])
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1:
        text = text[a : b + 1]
    return json.loads(text)


def _extract_resume(pdf_path: str, progress=None) -> Optional[JSONResume]:
    handler = PDFHandler()
    text = handler.extract_text_from_pdf(pdf_path)
    if not text or len(text.strip()) < 40:
        return None

    def run(name):
        data = handler._extract_section_data(text, name)
        if data is None:
            data = handler._extract_section_data(text, name)
        return name, data

    # The six sections are independent, so they run concurrently. The pool hands
    # each call a different key, and a key still serves one prompt at a time.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    done_n = 0
    if progress:
        progress("parse", "Reading your resume", f"Sections read: 0 of {len(SECTIONS)}")
    with ThreadPoolExecutor(max_workers=len(SECTIONS)) as ex:
        futs = {ex.submit(run, name): name for name in SECTIONS}
        for fut in as_completed(futs):
            name, data = fut.result()
            results[name] = data
            done_n += 1
            if progress:
                progress("parse", "Reading your resume", f"Sections read: {done_n} of {len(SECTIONS)}")

    if results.get("basics") is None and results.get("work") is None:
        raise ValueError(
            "The AI model could not parse this resume right now. Please try again in a moment."
        )
    merged: Dict = {
        k: None
        for k in (
            "basics",
            "work",
            "volunteer",
            "education",
            "awards",
            "certificates",
            "publications",
            "skills",
            "languages",
            "interests",
            "references",
            "projects",
            "meta",
        )
    }
    for name in SECTIONS:
        if results.get(name):
            merged.update(results[name])
    if isinstance(merged.get("basics"), dict):
        try:
            merged["basics"] = Basics(**merged["basics"])
        except Exception:
            merged["basics"] = None
    return JSONResume(**merged)


def _github_data(resume: JSONResume, role) -> dict:
    profiles = (resume.basics.profiles if resume.basics else None) or []
    gh = next(
        (p for p in profiles if p.network and p.network.lower() == "github"), None
    )
    if not gh or not gh.url:
        return {}
    try:
        data = fetch_and_display_github_info(gh.url, position_title=role.position_title)
        return data if isinstance(data, dict) and "profile" in data else {}
    except Exception as exc:  # GitHub is an optional signal; never fail the run on it
        logger.warning("GitHub enrichment skipped: %s", exc)
        return {}


def analyze_pdf(pdf_bytes: bytes, progress: Progress) -> dict:
    role = load_role(ROLE_NAME)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(pdf_bytes)
        path = fh.name
    try:
        progress("parse", "Reading your resume")
        resume = _extract_resume(path, progress)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if resume is None:
        raise ValueError(
            "Could not read text from this PDF. Upload a text-based resume, not a scan."
        )

    progress("github", "Checking public profile signals")
    github = _github_data(resume, role)

    progress("score", "Scoring against the rubric")
    evaluation_model = build_evaluation_model(role)
    evaluator = ResumeEvaluator(
        role=role,
        evaluation_model=evaluation_model,
        model_name=DEFAULT_MODEL,
        model_params=MODEL_PARAMETERS.get(DEFAULT_MODEL),
    )
    resume_text = convert_json_resume_to_text(resume)
    if github:
        resume_text += convert_github_data_to_text(github)
    ev = evaluator.evaluate_resume(resume_text)

    categories, total, max_total = [], 0.0, 0
    for cat in role.categories:
        cs = getattr(ev.scores, cat.key, None)
        if cs is None:
            continue
        got = min(cs.score, cat.max)
        total += got
        max_total += cat.max
        categories.append(
            {
                "key": cat.key,
                "label": cat.label,
                "score": got,
                "max": cat.max,
                "evidence": cs.evidence,
            }
        )
    bonus = ev.bonus_points.total if getattr(ev, "bonus_points", None) else 0
    deduction = ev.deductions.total if getattr(ev, "deductions", None) else 0
    total = max(0.0, min(total + bonus - deduction, max_total + role.bonus_max))

    evaluation = {
        "total": round(total, 1),
        "max": max_total,
        "bonus": bonus,
        "deductions": deduction,
        "deduction_reasons": getattr(ev.deductions, "reasons", "") if deduction else "",
        "categories": categories,
        "strengths": list(ev.key_strengths),
        "improvements": list(ev.areas_for_improvement),
    }

    progress("suggest", "Writing suggestions")
    provider = initialize_llm_provider(DEFAULT_MODEL)
    user = (
        "RESUME:\n"
        + resume_text[:14000]
        + "\n\nRUBRIC EVALUATION (JSON):\n"
        + json.dumps(evaluation)[:6000]
        + "\n\nGive prioritized, concrete suggestions to make this resume stronger."
    )
    sug = Suggestions(
        **_chat_json(
            provider,
            [
                {"role": "system", "content": SUGGEST_SYSTEM},
                {"role": "user", "content": user},
            ],
            Suggestions.model_json_schema(),
        )
    )

    name = resume.basics.name if resume.basics and resume.basics.name else "Candidate"
    return {
        "candidate": name,
        "evaluation": evaluation,
        "suggestions": sug.model_dump(),
        "resume_text": resume_text[:14000],
    }


def chat_reply(
    history: List[dict], resume_text: str, evaluation: dict, suggestions: dict
) -> str:
    context = (
        "RESUME:\n"
        + resume_text[:12000]
        + "\n\nEVALUATION:\n"
        + json.dumps(evaluation)[:4000]
        + "\n\nSUGGESTIONS ALREADY SHOWN:\n"
        + json.dumps(suggestions)[:4000]
    )
    messages = [{"role": "system", "content": CHAT_SYSTEM + context}]
    for m in history[-12:]:
        role = "assistant" if m.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": str(m.get("content", ""))[:2000]})
    provider = initialize_llm_provider(DEFAULT_MODEL)
    params = dict(
        MODEL_PARAMETERS.get(DEFAULT_MODEL, {"temperature": 0.4, "top_p": 0.9})
    )
    params["temperature"] = 0.4
    resp = provider.chat(
        model=DEFAULT_MODEL, messages=messages, options={"stream": False, **params}
    )
    return resp["message"]["content"].strip()
