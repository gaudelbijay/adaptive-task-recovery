"""Instruction schema, parsing, and goal graphs.

`goal_graph.py` is D-013's reviewed core schema (Accepted, D-037) --
`Goal`, `Constraint`, `GoalGraph`. `instruction_parser.py` (D-019/D-026,
promoted D-038) is the controlled-grammar parser that compiles
instruction text into a `GoalGraph` -- reproduces every hand-authored
`GoalGraph` in this project from its own instruction text, and
generalizes to held-out paraphrases and a held-out object composition
(see `ai-notes/decisions.md` D-038 for the promotion case).

`compositional_generalization.py` (D-079/D-080) is H4's first comparative
test: the real factorized parser versus both exact memorization and a stronger
non-factorized character-ngram retriever. The retriever handles all held-out
paraphrases but cannot construct the held-out composition; the factorized
parser handles both. See the module docstring for the deliberately small scope.
D-081 expands the comparison to four training and four semantically disjoint
held-out role recombinations rather than relying on one composition.
"""

from __future__ import annotations
