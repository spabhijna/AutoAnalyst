"""Exceptions raised by the pipeline loop."""

from __future__ import annotations


class PipelineAbortedError(RuntimeError):
    """
    Raised when the run cannot continue: a sub-question exhausted all CODE-GEN
    retries without a trustworthy result, the plan-revision cap was hit, or the
    scheduler found no ready task.

    Continuing past this point would cascade an unverified (possibly wrong)
    finding into all downstream sub-questions, making the entire run
    unreliable. The run stops here and returns the partial state.

    Attributes:
        sub_question: The question that could not be answered.
        last_reason:  The critique's reason from the final attempt.
        run_id:       The run identifier for log correlation.
        kind:         Short description of why the run stopped (used in the
                      message and the console banner).
    """

    RETRIES_EXHAUSTED = "max retries exhausted"

    def __init__(
        self,
        sub_question: str,
        last_reason: str,
        run_id: str,
        kind: str = RETRIES_EXHAUSTED,
    ) -> None:
        self.kind = kind
        self.sub_question = sub_question
        self.last_reason = last_reason
        self.run_id = run_id
        super().__init__(
            f"Pipeline aborted (run_id={run_id}): {kind} on "
            f"sub_question={sub_question!r}. Last critique: {last_reason}"
        )
