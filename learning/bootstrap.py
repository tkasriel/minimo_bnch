#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import os
import io
import json
import datetime

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
agent=None
background_theory = None

def init_pool(a: ProofSearchAgent, b: worker.BackgroundTheory):
    global agent, background_theory
    agent = a
    background_theory = b


def submit_task(statement: str, verbose=False):
    global agent, background_theory
    assert agent is not None
    assert background_theory is not None
    return worker.try_prove(agent, background_theory, statement, verbose)


# def get_task_result(task: mp.Process) -> StudentResult:
#     return task.join()


async def teacher_loop(cfg: DictConfig):
    if cfg.get("use_multiprocessing"):
        mp.set_start_method(cfg.get("mp_start_method"))
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
            torch.save(agent, f'{i}.pt')
            print(f"current it: {i}")
            context = Context(d, None, [])
            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')

            progress_bar = tqdm(total=cfg.n_conjectures)

            conjectures = []

            while len(conjectures) < cfg.n_conjectures:
                # print("sub proposal")
                proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard,useful,few_zeros) '), context)
                # print(proposal)

                if proposal and proposal not in conjectures + proven_conjectures:
                    # Contract conjectures to make them Peano-parseable.
                    contracted_proposal = d.contract(proposal)
                    if contracted_proposal not in conjectures + proven_conjectures:
                        conjectures.append(contracted_proposal)
                        progress_bar.update(1)
                # print("prop fini")

            progress_bar.close()


            print(now(), 'done, have', len(conjectures), 'conjectures')
            print(conjectures)

            log.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'conjectures': conjectures}))
            log.write('\n')
            log.flush()

            # 2- Try to prove each of the conjectures

            # Dump current agent.
            # buff = io.BytesIO()
            # torch.save(agent, buff)
            # agent_dump = buff.getvalue()

            print('Running proof search...')
            student_results: list[StudentResult] = []
            success_logprobs = []
            examples = []
            background_theory = worker.BackgroundTheory(theory, premises)
            if cfg.get("use_multiprocessing"):
                agent.share_memory()
                with mp.Pool(processes=cfg.get("num_processes"), initializer=init_pool, initargs=(agent, background_theory)) as p:
                    print(f"{cfg.get('num_processes')} processes initialized")
                    for res in tqdm(p.imap_unordered(submit_task, conjectures), total=len(conjectures)):
                        assert type(res) is StudentResult
                        print(f"Conjecture: {res.problem}")
                        if res.error:
                            print("Proof search errored.")
                            print(res.error)
                            continue
                        if res.success:
                            assert res.proof
                            print("Proof search was a success! Proof is")
                            print("\n\t".join(res.proof))
                            print(f"logprob: {res.logprob}")
                        else:
                            print("Proof search failed")
                        print(f"Got {len(res.hindsight_examples)} hindsight examples\n")
                        student_results.append(res)


            else:
                for index, conjecture in enumerate(tqdm(conjectures, miniters=1)):
                    init_pool(agent, background_theory)
                    student_results.append(submit_task(conjecture, True))

            # 3- Train model on proofs and outcome of conjectures (easy, hard, timeout)
            # print('Collecting', len(tasks), 'results from workers.')
            # for task in tqdm(tasks, miniters=1):
            #     student_result = get_task_result(task)

            #     if student_result.error:
            #         print('Error in prover process!')
            #         print(student_result.error)
            #         continue


                # student_results.append(student_result)

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
