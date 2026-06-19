"""
ResearchMind AI – Summarizer
Generates structured academic paper summaries and multi-paper comparisons
using free LLM providers.
"""

from typing import List, Dict

from core.llm_handler import LLMHandler
from core.pdf_processor import PageData

_SYS = "You are a senior AI researcher who writes clear, accurate, and insightful analyses of academic papers."


# ─── Prompt Templates ─────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """\
Analyze the following research paper content and produce a structured summary.

PAPER CONTENT:
{content}

Generate a detailed summary in EXACTLY this Markdown format (keep the headers):

## 📋 Problem Statement
What specific problem does this paper address? What gap in existing knowledge motivates it?

## 🔬 Methodology
What techniques, algorithms, models, or experimental designs are used?
Include key equations or architectural choices if mentioned.

## 📊 Datasets Used
List all datasets mentioned. Include size, domain, and purpose if available.
If none mentioned, write "Not specified."

## 🏆 Key Results & Contributions
What are the main findings? Include specific metrics and benchmark comparisons.
What is the primary contribution to the field?

## ⚠️ Limitations
What weaknesses or constraints do the authors acknowledge?

## 🔮 Future Work
What directions do the authors suggest for follow-on research?

## 🏷️ Keywords
[comma-separated list of 6-10 key technical terms]

Base your analysis ONLY on the provided content. Do not fabricate results."""


_COMPARE_PROMPT = """\
You are given summaries of {n} research papers. Produce a comprehensive comparison.

{paper_summaries}

Write a structured comparison using EXACTLY this format:

## 🎯 Research Objectives
Compare what each paper aims to solve. Note similarities and divergences.

## 🔬 Methodology Comparison
| Aspect | {paper_cols} |
|--------|{sep}|
| Core approach | ... |
| Model type | ... |
| Key innovation | ... |

## 📊 Dataset Comparison
Compare the datasets used across papers (size, domain, public vs proprietary).

## 📈 Results Comparison
Compare key metrics side-by-side. Which paper achieves the best performance and on what benchmarks?

## 💪 Strengths & Weaknesses
For each paper:
- **Paper N** (filename): Strengths → ... | Weaknesses → ...

## 🔗 Key Connections & Differences
What techniques or ideas appear across multiple papers?
What fundamentally differentiates the approaches?

## 🏆 Overall Assessment
Which approach is most promising and why? What does the field still need?"""


# ─── Summarizer ───────────────────────────────────────────────────────────────

class Summarizer:

    def __init__(self):
        self.llm = LLMHandler()

    def summarize(self, pages: List[PageData], max_chars: int = 9000) -> str:
        """Generate a structured summary for a single paper."""
        content = self._prepare_content(pages, max_chars)
        prompt  = _SUMMARY_PROMPT.format(content=content)
        return self.llm.generate(
            prompt,
            system_prompt=_SYS,
            max_tokens=2500,
            temperature=0.1,
        )

    def compare(self, papers: List[Dict]) -> str:
        """
        Generate a cross-paper comparison.
        papers: list of dicts with keys: filename, summary
        """
        if len(papers) < 2:
            return "Need at least 2 papers to compare."

        paper_sections = ""
        for i, p in enumerate(papers, start=1):
            paper_sections += f"\n\n---\nPAPER {i}: {p['filename']}\n{p['summary']}"

        cols = " | ".join(f"Paper {i}" for i in range(1, len(papers) + 1))
        sep  = "|".join("---" for _ in papers)

        prompt = _COMPARE_PROMPT.format(
            n=len(papers),
            paper_summaries=paper_sections,
            paper_cols=cols,
            sep=sep,
        )
        return self.llm.generate(
            prompt,
            system_prompt=_SYS,
            max_tokens=3500,
            temperature=0.1,
        )


    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_content(pages: List[PageData], max_chars: int) -> str:
        """
        Smart content selection:
        - First 65 % → problem, intro, method
        - Last 25 %  → results, conclusion, references
        (Middle often contains dense proofs / figures we can skip)
        """
        full = "\n".join(p.text for p in pages)
        if len(full) <= max_chars:
            return full

        head_size = int(max_chars * 0.65)
        tail_size = int(max_chars * 0.25)

        head = full[:head_size]
        tail = full[-tail_size:]
        return head + "\n\n[...middle section abbreviated...]\n\n" + tail
