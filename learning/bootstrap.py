#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import os
import io
import json
import datetime
import time
from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig
import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

import peano
from problems import load_natural_number_game_problemset
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import ProofSearchAgent, make_agent


def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"
CONJECTURE_PROMPT= 'Conj:(hard,useful,few_zeros) '
STOP = "STOP"
CONJECTURE = "CONJECTURE"
PROOF = "PROOF"

def process_main(id: int, agent: ProofSearchAgent, background_theory: worker.BackgroundTheory, instruction_queue: mp.Queue, output_queue: mp.Queue):
    instruction: tuple[str,str] = ("","")
    
    # Unfortunately PyDerivation is not pickle-able and I don't know enough rust to fix that
    d = peano.PyDerivation()
    d.incorporate(background_theory.theory)
    context = Context(d, None, [])
    print(f"Process {id} ready")

    while instruction[0] != STOP:
        instruction = instruction_queue.get()
        # print(f"Received instruction: {instruction}")
        if instruction and type(instruction) is tuple:
            if instruction[0] == CONJECTURE:
                new_conj = sample_conjecture(AgentLM(agent, CONJECTURE_PROMPT), context)
                output_queue.put((CONJECTURE, new_conj))
            elif instruction[0] == PROOF:
                thm_to_prove = instruction[1]
                result = worker.try_prove(agent, background_theory, thm_to_prove, verbose=False)
                output_queue.put((PROOF, result))
    output_queue.put((STOP, id))


async def teacher_loop(cfg: DictConfig):
    if cfg.use_multiprocessing:
        mp.set_start_method(cfg.mp_start_method)
    agent = make_agent(cfg)
    theory_folder = "theories"
    if "output" in os.path.abspath(__file__):
        # print(os.path.abspath(__file__))
        theory_folder = "../../../theories"
    with open(os.path.join(os.path.dirname(__file__), theory_folder, cfg.theory.name + '.p')) as f:
        theory = f.read()

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                                 for i in range(len(cfg.difficulty_buckets))],
                                key=lambda kv: kv[1])

    premises = cfg.theory.premises

    d = peano.PyDerivation()
    d.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    outcomes = []

    continue_dir = cfg.get('continue')
    start_iteration = 0
    # continue_dir = "/Users/tkasriel/code/rsh/minimo/learning/outputs/2025-01-28/12-49-46"

    if continue_dir is not None:
        os.chdir(continue_dir)
        print('Continuing run from', continue_dir)
        # Find largest iteration number such that i.pt exists.
        i = 0
        while os.path.exists(f'{i}.pt'):
            i += 1
        i -= 1
        start_iteration = i
        agent = torch.load(f'{i}.pt')
        print('Loaded agent from', f'{i}.pt')
        # Load examples and outcomes.
        if i > 0:
            with open(f'outcomes_{i-1}.json', 'r') as f:
                outcomes = json.load(f)
                proven_conjectures = [o['problem'] for o in outcomes
                                      if o['hindsight'] is False and
                                         o['proof'] is not None]
                seen_hindsight_goals = {o['problem'] for o in outcomes
                                        if o['hindsight'] and o['proof'] is not None}

        print('Loaded', len(proven_conjectures), 'proven conjectures from previous run.')


    if cfg.get('freeze_conjecturer', False):
        print('Ablation: Freezing conjecturer.')


    with open('log.jsonl', 'w') as log:
        for i in range(start_iteration, cfg.iterations):
            start_time = time.time()
            torch.save(agent, f'{i}.pt')
            print(f"current it: {i}")
            context = Context(d, None, [])
            background_theory = worker.BackgroundTheory(theory, premises)
            num_processes = cfg.num_processes

            # Start multiprocessing
            if cfg.use_multiprocessing:
                instruction_queue: mp.Queue = mp.Queue()
                output_queue: mp.Queue = mp.Queue()
                processes: list[mp.Process] = []
                agent.share_memory()
                for j in range(num_processes):
                    new_process = mp.Process(target=process_main,kwargs={"id": j,
                                                                         "agent":agent, 
                                                                         "background_theory": background_theory, 
                                                                         "instruction_queue": instruction_queue, 
                                                                         "output_queue": output_queue})
                    new_process.start()
                    processes.append(new_process)

            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')
            progress_bar = tqdm(total=cfg.n_conjectures)
            conjectures: list[str] = []
            if cfg.use_multiprocessing:
                for j in range(min(num_processes, cfg.n_conjectures)):
                    instruction_queue.put((CONJECTURE, "")) # You can put the seed statement here

            while len(conjectures) < cfg.n_conjectures:
                if cfg.use_multiprocessing:
                    _, proposal = output_queue.get()
                    instruction_queue.put((CONJECTURE, ""))
                else:
                    proposal = sample_conjecture(AgentLM(agent, CONJECTURE_PROMPT), context)

                if proposal and proposal not in conjectures + proven_conjectures:
                    # Contract conjectures to make them Peano-parseable.
                    contracted_proposal = d.contract(proposal)
                    if contracted_proposal not in conjectures + proven_conjectures:
                        conjectures.append(contracted_proposal)
                        progress_bar.update(1)
                

            progress_bar.close()

            print(now(), 'done, have', len(conjectures), 'conjectures')
            print(conjectures)

            log.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'conjectures': conjectures}))
            log.write('\n')
            log.flush()
            end_conjecture_time = time.time()

            # 2- Try to prove each of the conjectures

            print('Running proof search...')
            student_results: list[StudentResult] = []
            success_logprobs = []
            examples = []
            if cfg.use_multiprocessing:
                # Send the instructions
                for conjecture in conjectures:
                    instruction_queue.put((PROOF, conjecture))
                for j in range(cfg.num_processes):
                    instruction_queue.put((STOP, ""))
                
                # Process results
                num_dead = 0
                progress_bar = tqdm(total=len(conjectures))
                while len(student_results) < len(conjectures) and num_dead < cfg.num_processes:
                    res_type, res = output_queue.get()
                    if res_type == CONJECTURE:
                        # Leftover from conjecturing. We could use them, but for now I'll just throw them out
                        continue
                    if res_type == STOP:
                        num_dead += 1
                        print(f"Process {res} is done")
                        continue
                    assert res_type == PROOF
                    print(f"Conjecture: {res.problem}")
                    if res.error:
                        print("Proof search errored.")
                        print(res.error)
                        continue
                    if res.success:
                        assert res.proof
                        print("Proof search was a success! Proof is")
                        print("\n\t".join(res.solution_actions))
                        print(f"logprob: {res.logprob}")
                    else:
                        print("Proof search failed")
                    print(f"Got {len(res.hindsight_examples)} hindsight examples\n")
                    student_results.append(res)
                    progress_bar.update(1)
                progress_bar.close()

                for process in processes:
                    process.join()
                    process.close()
            else:
                for index, conjecture in enumerate(tqdm(conjectures, miniters=1)):
                    student_results.append(worker.try_prove(agent, background_theory, conjecture, True))
            end_search_time = time.time()

            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            for student_result in student_results:
                if student_result.success:
                    success_logprobs.append(student_result.logprob)

                outcomes.append({'iteration': i,
                                'problem': student_result.problem,
                                'proof': student_result.proof,
                                'logprob': student_result.logprob,
                                'actions': student_result.solution_actions,
                                'hindsight': False
                                })

                for h in student_result.hindsight_examples:
                    outcomes.append({'iteration': i,
                                    'problem': h.statement,
                                    'proof': h.proof,
                                    'logprob': h.logprob,
                                    'actions': h.solution_actions,
                                    'hindsight': True
                                    })
            save_json(outcomes, f'outcomes_{i}.json')

            if not success_logprobs:
                print(f'No solutions found in iteration {i}...')
                thresholds = [-2.906669173782248, -1.6445413855994306, -0.9526082203994054]
            #     break
            else:
                thresholds = [np.percentile(success_logprobs, p)
                            for _, p in difficulty_buckets]

                print('Thresholds:',
                    list(zip([k for k, _ in difficulty_buckets], thresholds)),
                    'min =', np.min(success_logprobs),
                    'max =', np.max(success_logprobs))

            # 3b- Classify problems into easy/hard.
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
                if student_result.success:
                    outcome = next(k
                                for i, (k, _) in enumerate(difficulty_buckets)
                                if (student_result.logprob <= thresholds[i] or
                                    i + 1 == len(difficulty_buckets)))
                else:
                    outcome = FAIL

                if not cfg.get('freeze_conjecturer', False):
                    # print(student_result.problem)
                    tags = [outcome]
                    if ": nat" in student_result.problem: tags.append("useful")
                    if student_result.problem.count("z") < 3: tags.append("few_zeros")
                    try:
                        examples.append(f'Conj:({",".join(tags)}) ' + d.elaborate(student_result.problem))
                    except BaseException: # wtf is this
                        pass

                if student_result.success:
                    proven_conjectures.append(student_result.problem)
                    proofs.append(student_result.proof)

                examples.extend(student_result.extracted_examples)

                if cfg.train_policy_on_hindsight_examples:
                    for h in student_result.hindsight_examples:
                        if h.goal not in seen_hindsight_goals:
                            outcome = next(k
                                        for i, (k, _) in enumerate(difficulty_buckets)
                                        if h.logprob <= thresholds[i] or i + 1 == len(difficulty_buckets))

                            if not cfg.get('freeze_conjecturer', False):
                                tags = [outcome]
                                if ": nat" in student_result.problem: tags.append("useful")
                                if student_result.problem.count("z") < 3: tags.append("few_zeros")
                                try:
                                    examples.append(f'Conj:({",".join(tags)}) ' + d.elaborate(student_result.problem))
                                except BaseException:
                                    pass
                            examples.extend(h.examples)
                            seen_hindsight_goals.add(h.goal)

            log.write(json.dumps({'iteration': i,
                                'msg': f'Training on {len(examples)} examples.'}))
            log.write('\n')

            # 3c- Train model on conjecturing and proof search examples.
            print(len(examples), 'accumulated training examples.')
            agent.train(examples)
            train_end_time = time.time()
            with open("time_metric.txt", "a+") as tm:
                tm.write(f"Iteration {i}: \n")
                tm.write(f"Proof search took {end_conjecture_time-start_time}s\n")
                tm.write(f"Proof search took {end_search_time-end_conjecture_time}s\n")
                tm.write(f"Training took {train_end_time-end_search_time}s\n")
                tm.write(f"Total time taken: {train_end_time-start_time}s\n")

            save_json(examples, f'examples_{i}.json')
            try:
                torch.save(student_results, f'results_{i}.json')
            except Exception as e:
                print(e) # will fix later.


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
