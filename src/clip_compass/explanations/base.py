"""Protocol for structured Mapper-node explanation providers."""

from typing import Mapping, Protocol

from .types import NodeExplanation, NodeExplanationContext


class ExplanationAgent(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]:
        ...

    def explain(
        self, context: NodeExplanationContext, *, max_hypotheses: int
    ) -> NodeExplanation:
        ...
