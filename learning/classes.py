from enum import Enum
from typing import List
from pydantic import BaseModel, RootModel, TypeAdapter

from worker import StudentResult

class InstructionEnum(str, Enum):
    CONJECTURE = "CONJECTURE"
    PROOF = "PROOF"
    STOP = "STOP"


class MPInstruction (BaseModel):
    instruction: InstructionEnum
    agent_file: str | None = None
    seed: str | None = None
    thm_to_prove: str | None = None
    theory: tuple[str, list[str]]
    previous_conjectures: list[str] | None = None
    extract_hindsight : bool | None = None

class MPResult (BaseModel):
    instruction: InstructionEnum
    result: str | None | StudentResult | int
    seed: str | None = None
    time_taken: float

class ProofOutcome (BaseModel):
    iteration: int
    problem: str
    problem_raw: str | None = None
    problem_translated: str | None = None
    proof: list | None
    logprob: float | None
    actions: list[str] | None
    hindsight: bool
    seed_used: str | None = None

class UsefulConjecture (BaseModel):
    theorem: str
    iter_generated: int
    freq_used: int
    tot_improvement: float


ProofOutcomeList = TypeAdapter(List[ProofOutcome])
UsefulConjectureList = TypeAdapter(List[UsefulConjecture])
