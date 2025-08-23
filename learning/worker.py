#!/usr/bin/env python3

import time
import gc
import io
from dataclasses import dataclass
from typing import Optional
import traceback
import os

import torch
from omegaconf import DictConfig
from celery import Celery

import peano
import proofsearch
import policy
import hindsight

@dataclass
class StudentResult:
    error: Optional[str]
    success: bool
    problem: str
    solution_actions: Optional[list[str]]
    proof: Optional[list[str]]
    extracted_examples: list[str]
    hindsight_examples: list[hindsight.HindsightExample]
    iterations: int
    logprob: float


@dataclass
class BackgroundTheory:
    theory: str
    premises: list[str]


redis_url = f'redis://{os.environ.get("REDIS", "localhost")}'
app = Celery('worker', backend=redis_url, broker=redis_url)
app.conf.task_serializer = 'pickle'
app.conf.result_serializer = 'pickle'
app.conf.worker_max_tasks_per_child = 10
app.conf.worker_max_memory_per_child = 1e9
app.conf.accept_content = ['application/json', 'application/x-python-serialize']


def try_prove(cfg, agent: proofsearch.ProofSearchAgent, theory: BackgroundTheory, statement: str, verbose: bool = False) -> StudentResult:
    # print(f"worker, curr allocated (init): {torch.cuda.memory_allocated()}")

    # print('Proving', statement[0], 'on', agent._policy._lm._lm.device)
    state = peano.PyProofState(theory.theory,
                               theory.premises,
                               statement)

    #print(f"Instantiating peano state took {time.time()-curr_time:02.1f}s")
    try:
        # start_proof_time = time.time()
        agent_result = agent.proof_search(cfg, statement, state, verbose)
        # print (f"Proof took {time.time()-start_proof_time()}")

        if agent_result.success:
            proof = agent_result.root.state_node.reconstruct_proof(
                agent_result.root.get_solution_actions())
            solution_actions = agent_result.root.get_solution_actions()
            logprob = agent_result.root.solution_logprob_under_policy(cfg, agent._policy, solution_actions)
        else:
            solution_actions, proof, logprob = None, None, None
        #print(f"Getting results took {time.time()-curr_time:02.1f}s")

        examples = []
        # Policy examples for the proved goal.
        examples.extend(agent._policy.extract_examples(root=agent_result.root))
        # Hindsight examples (policy + conjecturing).
        try:
            hindsight_examples = hindsight.extract_hindsight_examples(
                    cfg,
                    agent_result.root,
                    theory.theory,
                    theory.premises,
                    agent._policy)
        except:
            hindsight_examples = []


        return StudentResult(
            None,
            agent_result.success,
            statement,
            list(map(str, solution_actions)) if solution_actions else None,
            proof,
            agent_result.examples,
            hindsight_examples,
            agent_result.iterations,
            logprob,
        )
    except BaseException as e:
        if type(e) == KeyboardInterrupt:
            raise KeyboardInterrupt()
        tb = traceback.format_exception(e)
        print('Error in try_prove!')
        print("".join(tb))
        return StudentResult("\n".join(tb), False, statement, None, None, [],
                             None, None, None)
