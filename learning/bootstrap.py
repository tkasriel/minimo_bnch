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
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent


def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"


DISTRIBUTED = os.environ.get('DISTRIBUTED', False)


def submit_task(agent_dump: bytes, theory: worker.BackgroundTheory, statement: str):
    if DISTRIBUTED:
        return worker.try_prove.apply_async((agent_dump, theory, statement))
    else:
        return worker.try_prove.run(agent_dump, theory, statement)


def get_task_result(task):
    if DISTRIBUTED:
        return task.get()
    else:
        return task


async def teacher_loop(cfg: DictConfig):
    print('Running in', 'distributed mode.' if DISTRIBUTED else 'single-process mode.')

    agent = make_agent(cfg)
    os.chdir("/Users/tkasriel/code/rsh/minimo")
    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
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
        for i in range(start_iteration, cfg.iterations + 1):
            i = 5
            torch.save(agent, f'{i}.pt')
            if i == cfg.iterations:
                print('test')
                # agent._max_mcts_nodes = 1000
                conjectures = [x[1] for x in load_natural_number_game_problemset()._statements.values()]

#                 theory = """
# = : [('t : type) -> 't -> 't -> prop].

# nat : type.
# false : prop.

# z : nat.
# s : [nat -> nat].

# + : [nat -> nat -> nat].
# * : [nat -> nat -> nat].
# ^ : [nat -> nat -> nat].

# #forward s.
# #forward +.

# /* Defining axioms for addition */
# +_z : [('n : nat) -> (= (+ 'n z) 'n)].
# +_s : [('n : nat) -> ('m : nat) -> (= (+ 'n (s 'm)) (s (+ 'n 'm)))].
# #forward +_z ((+ 'n z) : nat).
# #forward +_s ((+ 'n (s 'm)) : nat).



# /* Defining axioms for multiplication */
# *_z : [('n : nat) -> (= (* 'n z) z)].
# *_s : [('n : nat) -> ('m : nat) -> (= (* 'n (s 'm)) (+ 'n (* 'n 'm)))].
# #forward *_z ((* 'n z) : nat).
# #forward *_s ((* 'n (s 'm)) : nat).

# /* Defining axioms for exponentiation */
# ^_z : [('n : nat) -> (= (^ 'n z) (s z))].
# ^_s : [('n : nat) -> ('m : nat) -> (= (^ 'n (s 'm)) (* 'n (^ 'n 'm)))].
# #forward ^_z ((^ 'n z) : nat).
# #forward ^_s ((^ 'n (s 'm)) : nat).

# /* Natural number induction */
# nat_ind : [('p : [nat -> prop]) -> ('p z) -> [('n : nat) -> ('p 'n) -> ('p (s 'n))] -> [('n : nat) -> ('p 'n)]].
# #backward nat_ind infer subgoal subgoal.

# #forward succ_inj ((= (s 'a) (s 'b)) : 't).
# succ_inj : [('a : nat) -> ('b : nat) -> (= (s 'a) (s 'b)) -> (= 'a 'b)].

# exists : [('t : type) -> ('p : ['t -> prop]) -> prop].
# ex_intro : [('t : type) -> ('p : ['t -> prop]) -> ('x : 't) -> ('p 'x) -> (exists 't 'p)].
# ex_wit : [('t : type) -> ('p : ['t -> prop]) -> (exists 't 'p) -> 't].
# ex_elim : [('t : type) -> ('p : ['t -> prop]) -> ('e : (exists 't 'p)) -> ('p (ex_wit 't 'p 'e))].

# #backward ex_intro infer infer infer subgoal.
# #forward ex_wit.
# #forward ex_elim ('e : (exists 't 'p)).

# leq : [nat -> nat -> prop] = (lambda ('a : nat, 'b : nat)
#                                      (exists nat (lambda ('c : nat) (= (+ 'a 'c) 'b)))).

# #forward rewrite.
# #forward eq_refl.
# #forward eq_symm ((= 'a 'b) : 't).

# and : [prop -> prop -> prop].

# and_elim_l : [('p : prop) -> ('q : prop) -> (and 'p 'q) -> 'p].
# #forward and_elim_l ('po : (and 'p 'q)).
# and_elim_r : [('p : prop) -> ('q : prop) -> (and 'p 'q) -> 'q].
# #forward and_elim_r ('po : (and 'p 'q)).
# and_intro : [('p : prop) -> ('q : prop) -> 'p -> 'q -> (and 'p 'q)].
# #backward and_intro infer infer subgoal subgoal.

# or : [prop -> prop -> prop].

# or_l : [('p : prop) -> ('q : prop) -> 'p -> (or 'p 'q)].
# #backward or_l infer infer subgoal.

# or_r : [('p : prop) -> ('q : prop) -> 'q -> (or 'p 'q)].
# #backward or_r infer infer subgoal.

# or_elim : [('p : prop) -> ('q : prop) -> (or 'p 'q) ->
#            ('r : prop) -> ['p -> 'r] -> ['q -> 'r]
#            -> 'r].
# #backward or_elim infer infer infer infer subgoal subgoal.

# false_elim : [('p : prop) -> false -> 'p].
# #backward false_elim infer infer.

# not : [prop -> prop] = (lambda ('p0 : prop) ['p0 -> false]).

# iff : [prop -> prop -> prop] = (lambda ('p1 : prop, 'p2 : prop) (and ['p1 -> 'p2] ['p2 -> 'p1])).

# zero_ne_succ : [('a : nat) -> (not (= z (s 'a)))].
# #forward zero_ne_succ.

# empty : type.

# #forward a_zero_add ((+ z 'n) : nat).
# #forward a_succ_add ((+ (s 'a) 'b) : nat).
# #forward a_add_assoc ((+ (+ 'a 'b) 'c) : nat).
# /* #forward a_add_assoc ((+ 'a (+ 'b 'c)) : nat). */
# #forward a_add_comm ((+ 'a 'b) : nat).
# #forward a_succ_eq_add_one. /* ((s 'n) : nat). */
# /* #forward a_succ_eq_add_one ((+ 'n (s z)) : nat). */
# #forward a_add_right_comm ((+ (+ 'a 'b) 'c) : nat).

# #forward m_zero_mul ((* z 'm) : nat).
# #forward m_mul_one ((* 'm (s z)) : nat).
# #forward m_one_mul ((* (s z) 'm) : nat).
# #forward m_mul_add ((* 't (+ 'a 'b)) : nat).
# #forward m_mul_assoc ((* (* 'a 'b) 'c) : nat).
# #forward m_mul_comm ((* 'a 'b) : nat).


# #forward succ_eq_succ_of_eq ('_ : (= 'a 'b)) ((s 'a) : nat) ((s 'b) : nat).
#     """
                # premises = ['eq_symm', 'eq_refl', 'rewrite', 'nat_ind', '+_z', '+_s'] #+ ['a_add_assoc', 'a_add_comm'] + ['a_zero_add', 'a_succ_add']
                premises = ListConfig(['eq_symm', 'eq_refl', 'rewrite', 'nat_ind', '+_z', '+_s'])
                # premises.extend()
            else:
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
            tasks = []
            student_results = []

            # Dump current agent.
            buff = io.BytesIO()
            torch.save(agent, buff)
            agent_dump = buff.getvalue()

            print('Submitting tasks...')
            success_logprobs = []
            examples = []
            for index, conjecture in enumerate(tqdm(conjectures, miniters=1)):
                task = submit_task(
                    agent_dump,
                    worker.BackgroundTheory(theory, premises),
                    conjecture)
                if i == cfg.iterations:
                    student_result = get_task_result(task)
                    student_results.append(student_result)

                    if student_result.error:
                        print('Error in prover process!')
                        print(student_result.error)
                        continue
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
                    if student_result.success:
                        proven_conjectures.append(student_result.problem)
                        proofs.append(student_result.proof)

                    examples.extend(student_result.extracted_examples)
                    print(len(examples), 'accumulated training examples.')
                    save_json(examples, f'examples_{i}_{index}.json')
                    save_json(outcomes, f'outcomes_{i}_{index}.json')
                    agent.train(examples)
                    buff = io.BytesIO()
                    torch.save(agent, buff)
                    agent_dump = buff.getvalue()
                    
                    
                else:
                    tasks.append(task)

            # 3- Train model on proofs and outcome of conjectures (easy, hard, timeout)
            if i < cfg.iterations:
                print('Collecting', len(tasks), 'results from workers.')
                student_results = []
                for task in tqdm(tasks, miniters=1):
                    student_result = get_task_result(task)

                    if student_result.error:
                        print('Error in prover process!')
                        print(student_result.error)
                        continue


                    student_results.append(student_result)

                success_logprobs = []

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

                if not success_logprobs:
                    print(f'No solutions found in iteration {i} - stopping learning loop...')
                    break

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
                                    if ": nat" in student_result.problem: tags.append("useful")
                                    if student_result.problem.count("z") < 3: tags.append("few_zeros")
                                    examples.append(f'Conj:({",".join(tags)}) ' + d.elaborate(student_result.problem))
                                examples.extend(h.examples)
                                seen_hindsight_goals.add(h.goal)

                log.write(json.dumps({'iteration': i,
                                    'msg': f'Training on {len(examples)} examples.'}))
                log.write('\n')

                # 3c- Train model on conjecturing and proof search examples.
                if i + 1 < cfg.iterations:
                    print(len(examples), 'accumulated training examples.')
                    agent.train(examples)

            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            torch.save(student_results, f'results_{i}.json')


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
