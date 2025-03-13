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

import peano
from problems import load_natural_number_game_problemset
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, UsefulConjecture, sample_conjecture
from proofsearch import ProofSearchAgent, make_agent


def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"


DISTRIBUTED = os.environ.get('DISTRIBUTED', False)


def submit_task(agent: ProofSearchAgent, theory: worker.BackgroundTheory, statement: str):
    return worker.try_prove(agent, theory, statement)


def get_task_result(task: StudentResult) -> StudentResult:
    return task


async def teacher_loop(cfg: DictConfig):
    print('Running in', 'distributed mode.' if DISTRIBUTED else 'single-process mode.')

    agent = make_agent(cfg)
    # os.chdir("~/minimo")
    theory_folder = "theories"
    if "output" in os.getcwd():
        theory_folder = "../../../theories"
    with open(os.path.join(os.path.dirname(__file__), theory_folder, cfg.theory.name + '.p')) as f:
        theory = f.read()
    

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                                 for i in range(len(cfg.difficulty_buckets))],
                                key=lambda kv: kv[1])
    premises = cfg.theory.premises

    permanent_deriv = peano.PyDerivation()
    permanent_deriv.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    outcomes = []
    useful_theorems: list[UsefulConjecture] = []

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
        if os.path.exists(f"{i}.5.pt"):
            agent = torch.load(f'{i}.5.pt')
            start_iteration = i + 1
            with open(f"examples_{i}.json") as f:
                examples = json.load(f)
                agent.train(examples)
        else:
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
            with open(f"generated_theorems_{i}") as f:
                thms = json.load(f)
                useful_theorems = [UsefulConjecture(**thm) for thm in thms]

        print('Loaded', len(proven_conjectures), 'proven conjectures from previous run.')


    if cfg.get('freeze_conjecturer', False):
        print('Ablation: Freezing conjecturer.')

    conjecture_index = len(useful_theorems)
    with open('log.jsonl', 'w') as log:
        for i in range(start_iteration, cfg.iterations):
            torch.save(agent, f'{i}.pt')
            print(f"current it: {i}")
            d = permanent_deriv.clone()
            if i > 0:
                d.incorporate("\n\n".join(map(lambda x: x.theorem, useful_theorems)))
            context = Context(d, None, [])
            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')

            progress_bar = tqdm(total=cfg.n_conjectures)

            conjectures: list[str] = []

            while len(conjectures) < cfg.n_conjectures:
                proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard,useful) '), context)

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

            # 2- Try to prove each of the conjectures
            tasks: list[StudentResult] = []

            # Dump current agent.
            # buff = io.BytesIO()
            # torch.save(agent, buff)
            # agent_dump = buff.getvalue()

            print('Submitting tasks...')
            student_results: list[StudentResult] = []
            success_logprobs = []
            examples = []
            background_theory = worker.BackgroundTheory(theory + "\n\n" + "\n\n".join([thm.theorem for thm in useful_theorems]), premises + [thm.theorem.split(" : ")[0] for thm in useful_theorems])
            for index, conjecture in enumerate(tqdm(conjectures, miniters=1)):
                task = submit_task(
                    agent,
                    background_theory,
                    conjecture)
                # if i == cfg.iterations:
                #     student_result = get_task_result(task)
                #     student_results.append(student_result)

                #     if student_result.error:
                #         print('Error in prover process!')
                #         print(student_result.error)
                #         continue
                #     if student_result.success:
                #         success_logprobs.append(student_result.logprob)

                #     outcomes.append({'iteration': i,
                #                     'problem': student_result.problem,
                #                     'proof': student_result.proof,
                #                     'logprob': student_result.logprob,
                #                     'actions': student_result.solution_actions,
                #                     'hindsight': False
                #                     })

                #     for h in student_result.hindsight_examples:
                #         outcomes.append({'iteration': i,
                #                         'problem': h.statement,
                #                         'proof': h.proof,
                #                         'logprob': h.logprob,
                #                         'actions': h.solution_actions,
                #                         'hindsight': True
                #                         })
                #     if student_result.success:
                #         proven_conjectures.append(student_result.problem)
                #         proofs.append(student_result.proof)

                #     examples.extend(student_result.extracted_examples)
                #     print(len(examples), 'accumulated training examples.')
                #     save_json(examples, f'examples_{i}_{index}.json')
                #     save_json(outcomes, f'outcomes_{i}_{index}.json')
                #     theory
                #     agent.train(examples)
                tasks.append(task)

            # 3- Train model on proofs and outcome of conjectures (easy, hard, timeout)
            print('Collecting', len(tasks), 'results from workers.')
            for task in tqdm(tasks, miniters=1):
                student_result = get_task_result(task)

                if student_result.error:
                    print('Error in prover process!')
                    print(student_result.error)
                    continue


                student_results.append(student_result)

            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            for student_result in student_results:
                if student_result.success:
                    success_logprobs.append(student_result.logprob)
                    for conj in useful_theorems:
                        conj_name = conj.theorem.split(" : ")[0]
                        if conj_name in student_result.proof: #type: ignore
                            conj.freq_used += 1


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

            if not success_logprobs:
                print(f'No solutions found in iteration {i} - stopping learning loop...')
                break

            thresholds = [np.percentile(success_logprobs, p)
                        for _, p in difficulty_buckets]

            print('Thresholds:',
                list(zip([k for k, _ in difficulty_buckets], thresholds)),
                'min =', np.min(success_logprobs),
                'max =', np.max(success_logprobs))

            # Cut the least used theorems
            useful_theorems = [thm for thm in useful_theorems if not (i - thm.iter_generated >= 3 and thm.freq_used == 0)]
            useful_theorems.sort(key=lambda thm: thm.freq_used/(i-thm.iter_generated))
            useful_theorems = useful_theorems[len(useful_theorems)//10:]
            for thm in useful_theorems:
                if thm.freq_used == 0:
                    continue
                thm_str = thm.theorem.split(" : ")[1][:-1]
                to_add = f'Conj:(useful) ' + d.elaborate(thm_str)
                if not to_add in examples:
                    examples.append(to_add)

            # 3b- Classify problems into easy/hard.
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
                if student_result.success:
                    outcome = next(k
                                for i, (k, _) in enumerate(difficulty_buckets)
                                if (student_result.logprob <= thresholds[i] or
                                    i + 1 == len(difficulty_buckets)))
                    
                    useful_theorems.append(UsefulConjecture(f"c{conjecture_index} : " + student_result.problem + ".", i, 0))
                    conjecture_index += 1
                else:
                    outcome = FAIL
                

                if not cfg.get('freeze_conjecturer', False):
                    tags = [outcome]
                    # if ": nat" in student_result.problem: tags.append("useful")
                    # if student_result.problem.count("z") < 3: tags.append("few_zeros")
                    
                    examples.append(f'Conj:({",".join(tags)}) ' + d.elaborate(student_result.problem))
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
                                # if ": nat" in student_result.problem: tags.append("useful")
                                # if student_result.problem.count("z") < 3: tags.append("few_zeros")
                                examples.append(f'Conj:({",".join(tags)}) ' + d.elaborate(student_result.problem))
                            examples.extend(h.examples)
                            seen_hindsight_goals.add(h.goal)

            # Finally, we add our generated and proven theorems into our useful theorems pile

            # 3c- Train model on conjecturing and proof search examples.
            
            log.write('\n')
            torch.save(agent, f"{i}.5.pt")
            save_json([thm.to_dict() for thm in useful_theorems], f"generated_theorems_{i}.json")
            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            print(len(examples), 'accumulated training examples.')
            log.write(json.dumps({'iteration': i,
                                    'msg': f'Training on {len(examples)} examples.'}))
            torch.save(student_results, f'results_{i}.json')
            agent.train(examples)


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
