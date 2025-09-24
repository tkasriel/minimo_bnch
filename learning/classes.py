from enum import Enum
from typing import List
from pydantic import BaseModel, RootModel, TypeAdapter
import os, json

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

class UsefulnessOutcome (BaseModel):
    iteration: int
    problem: str
    proof: list
    used_theorems: list
    improvement: float

class UsefulConjecture (BaseModel):
    theorem: str
    iter_generated: int
    freq_used: int
    tot_improvement: float

class LLMUsefulnessEvalTheorem(BaseModel):
    thm_string: str
    iteration: int
    proven: bool
    logprob: float
    thm_string_simple: str
    thm_org: str | None = None
    explanations: list[str] = []
    dedup_useful_at_k: list[bool] = []
    def __str__(self) -> str:
        return f"{self.thm_string} (iteration {self.iteration}) : {self.logprob} -- {self.thm_org or ''}\n\n" + "\n#####\n".join(self.explanations)
    def __hash__(self) -> int:
        return (self.thm_string + " -- " + (self.thm_org or "")).__hash__()

    def __eq__(self, other: object) -> bool:
        return self.__hash__() == other.__hash__()
    
class LLMUsefulnessEvalResult (BaseModel):
    useful_theorems: set[LLMUsefulnessEvalTheorem] = set()
    deduplicated_theorems: set[LLMUsefulnessEvalTheorem] = set()
    proven_deduplicated: set[LLMUsefulnessEvalTheorem] = set()

    def dump_to_folder (self, folder_path: str) -> None:
        add_pre = lambda filepath : os.path.join(folder_path, filepath)
        with open(add_pre("useful_theorems.txt"), "w") as f:
            f.write("\n============\n\n".join(map(str, self.useful_theorems)))
        with open(add_pre("useful_theorem_dedup.txt"), "w") as f:
            f.write("\n============\n\n".join(map(str, self.deduplicated_theorems)))
        with open(add_pre("useful_theorem_dedup_proven.txt"), "w") as f:
            f.write("\n============\n\n".join(map(str, self.proven_deduplicated)))
        
        with open(add_pre("useful_theorems.json"), "w") as f:
            json.dump([x.model_dump() for x in self.useful_theorems], f)
        with open(add_pre("useful_theorem_dedup.json"), "w") as f:
            json.dump([x.model_dump() for x in self.deduplicated_theorems], f)
    
    def extend (self, other: "LLMUsefulnessEvalResult") -> None:
        self.useful_theorems.update(other.useful_theorems)
        self.deduplicated_theorems.update(other.deduplicated_theorems)
        self.proven_deduplicated.update(other.proven_deduplicated)

ProofOutcomeList = TypeAdapter(List[ProofOutcome])
UsefulnessOutcomeList = TypeAdapter(List[UsefulnessOutcome])
UsefulConjectureList = TypeAdapter(List[UsefulConjecture])
