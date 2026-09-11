# model_router.py
"""
Smart Model Router (OpenAI)
Automatically routes requests to the most cost-effective model
"""

import os
import re
from enum import Enum
from typing import Optional, Tuple
from dotenv import load_dotenv

from langfuse import observe, get_client, propagate_attributes

# Drop-in replacement: every completion is traced with model, tokens and cost
from langfuse.openai import OpenAI

load_dotenv()

# Two tiers is all the routing needs. Override via .env to point the premium
# tier at a reasoning model without touching the routing logic.
MODEL_CHEAP = os.getenv("OPENAI_MODEL_CHEAP", "gpt-4o-mini")  # $0.15 / $0.60 per 1M
MODEL_PREMIUM = os.getenv("OPENAI_MODEL_PREMIUM", "gpt-4o")  # $2.50 / $10 per 1M

client = OpenAI()
langfuse = get_client()


class TaskType(Enum):
    SIMPLE = "simple"  # Yes/no, classification
    MODERATE = "moderate"  # Summarization, extraction
    COMPLEX = "complex"  # Analysis, reasoning
    CODE = "code"  # Code generation
    CREATIVE = "creative"  # Creative writing


class ModelRouter:
    """Route requests to optimal models based on task analysis."""

    MODELS = {
        TaskType.SIMPLE: MODEL_CHEAP,
        TaskType.MODERATE: MODEL_CHEAP,
        TaskType.CODE: MODEL_PREMIUM,
        TaskType.COMPLEX: MODEL_PREMIUM,
        TaskType.CREATIVE: MODEL_PREMIUM,
    }

    # Checked in order — the first match wins, so order encodes precedence
    RULES = [
        (
            TaskType.SIMPLE,
            [
                r"\b(yes or no)\b",
                r"\b(true or false)\b",
                r"\b(classify|categorize)\b",
                r"^is (this|it|the)",
                r"\b(which one|choose|select)\b",
            ],
        ),
        (
            TaskType.CODE,
            [
                r"\b(write|create|generate|fix|debug).*(code|function|class|script)\b",
                r"\b(python|javascript|typescript|java|rust)\b",
                r"```",  # Code blocks in prompt
            ],
        ),
        (
            TaskType.COMPLEX,
            [
                r"\b(analyze|evaluate|compare|critique)\b",
                r"\b(why|how).*(work|happen|cause)\b",
                r"\b(pros and cons|trade-?offs)\b",
                r"\b(explain.*(detail|depth))\b",
            ],
        ),
        (
            TaskType.CREATIVE,
            [
                r"\b(write|create|compose).*(story|poem|essay|blog)\b",
                r"\b(creative|imaginative|original)\b",
            ],
        ),
    ]

    def classify_task(self, prompt: str) -> Tuple[TaskType, Optional[str]]:
        """Analyze the prompt to determine task type.

        Returns (task_type, matched_pattern). The pattern is returned so the
        routing decision is auditable in the trace, not just its outcome.
        """
        prompt_lower = prompt.lower()

        for task_type, patterns in self.RULES:
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return task_type, pattern

        # Default to moderate
        return TaskType.MODERATE, None

    def route(
        self, prompt: str, override_model: Optional[str] = None
    ) -> Tuple[str, TaskType, Optional[str]]:
        """Get the optimal model for a prompt.

        Classifies once and returns everything the caller needs, so the
        regex work isn't repeated.
        """
        task_type, matched = self.classify_task(prompt)

        if override_model:
            return override_model, task_type, matched

        return self.MODELS[task_type], task_type, matched


router = ModelRouter()


def call_openai(prompt: str, model: str) -> str:
    """Call the OpenAI API and return the answer text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        name=f"routed-{model}",  # consumed by Langfuse, not sent to OpenAI
    )
    return response.choices[0].message.content


@observe()
def routed_llm_call(prompt: str, override_model: str = None) -> str:
    """LLM call with automatic model routing."""

    selected_model, task_type, matched = router.route(prompt, override_model)

    langfuse.update_current_span(
        metadata={
            "task_type": task_type.value,
            "routed_model": selected_model,
            "matched_pattern": matched or "none (default)",
            "overridden": override_model is not None,
        }
    )

    # Tags are applied before the call, so the generation inherits them and
    # you can slice cost by task type in the Langfuse dashboard
    with propagate_attributes(
        trace_name="routed-llm-call",
        tags=[f"task:{task_type.value}", f"model:{selected_model}"],
    ):
        return call_openai(prompt, model=selected_model)


if __name__ == "__main__":
    prompts = [
        "Is 2 + 2 = 4? Yes or no",  # SIMPLE   → cheap
        "Classify this review as positive or negative: 'it broke'",  # SIMPLE   → cheap
        "Summarize the history of the Nile delta in three sentences",  # MODERATE → cheap
        "Write a Python function to sort a list",  # CODE     → premium
        "Analyze the trade-offs between RAG and fine-tuning",  # COMPLEX  → premium
        "Write a short story about a farmer and a drone",  # CREATIVE → premium
    ]

    for p in prompts:
        model, task, matched = router.route(p)
        print(f"\n{'─' * 70}")
        print(f"PROMPT:  {p}")
        print(f"TASK:    {task.value}")
        print(f"MODEL:   {model}")
        print(f"MATCHED: {matched or 'no rule — default tier'}")
        answer = routed_llm_call(p)
        print(f"ANSWER:  {answer[:120]}...")

    langfuse.flush()