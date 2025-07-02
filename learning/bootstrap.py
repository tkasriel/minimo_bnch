#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import math
import os
import io
import json
import datetime
import random
import time
from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig
import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

import peano
import dsprover
from problems import load_natural_number_game_problemset
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, UsefulConjecture, sample_conjecture
from proofsearch import ProofSearchAgent, make_agent
from convert_to_lean import convert_arith
import re
from dotenv import load_dotenv


load_dotenv()

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"
CONJECTURE_PROMPT= 'Conj:(hard,useful) '
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
                seed_statement = instruction[1]
                new_conj = sample_conjecture(AgentLM(agent, CONJECTURE_PROMPT), context, seed=seed_statement)
                output_queue.put((CONJECTURE, new_conj, seed_statement))
            # elif instruction[0] == PROOF:
            #     thm_to_prove = instruction[1]
            #     result = worker.try_prove(agent, background_theory, thm_to_prove, verbose=False)
            #     output_queue.put((PROOF, result, None))
    output_queue.put((STOP, id, None))


async def teacher_loop(cfg: DictConfig):
    # if cfg.use_multiprocessing:
    #     mp.set_start_method(cfg.mp_start_method)

    agent = make_agent(cfg)
    # os.chdir("~/minimo")
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

    permanent_deriv = peano.PyDerivation()
    permanent_deriv.incorporate(theory)
    proven_conjectures = []
    comp_to_raw_dict = dict()
    seed_used = {}
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
        if i >= 0:
            if os.path.exists(f"{i}.5.pt"):
                agent = torch.load(f'{i}.5.pt')
                start_iteration = i + 1
                with open(f"examples_{i}.json") as f:
                    examples = json.load(f)
                    agent.train(examples)
            else:
                start_iteration = i
                agent = torch.load(f'{i}.pt', map_location=None if torch.cuda.is_available() else "cpu", weights_only=False)
                print('Loaded agent from', f'{i}.pt')
            # Load examples and outcomes.
            if i > 0:
                with open(f'outcomes_{i-1}.json', 'r') as f:
                    outcomes = json.load(f)
                    proven_conjectures = [o['problem'] for o in outcomes
                                        if o['proof'] is not None]
                    comp_to_raw_dict = {o['problem']: o['problem_raw'] for o in outcomes if "problem_raw" in o.keys()}
                with open(f"generated_theorems_{i-1}.json") as f:
                    thms = json.load(f)
                    useful_theorems = [UsefulConjecture(**thm) for thm in thms]

        print('Loaded', len(proven_conjectures), 'proven conjectures from previous run.')

    with open('log.jsonl', 'w') as log:
        for i in range(start_iteration, cfg.iterations):
            background_theory = worker.BackgroundTheory(theory + "\n\n" + "\n\n".join([thm.theorem for thm in useful_theorems]), premises + [thm.theorem.split(" : ")[0] for thm in useful_theorems])
            d = permanent_deriv.clone()
            context = Context(d, None, [])
            if i > 0:
                d.incorporate("\n\n".join(map(lambda a: f"c{a[0]} : {a[1].theorem} .", enumerate(useful_theorems))))
            start_time = time.time()
            torch.save(agent, f'{i}.pt')
            print(f"current it: {i}")
    
            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')
            progress_bar = tqdm(total=cfg.n_conjectures)
            conjectures: list[tuple[str, str]] = []

            def renumber_var_names(statement):
                matches = re.findall("'a\\d+", statement)
                if matches:
                    unique_matches = sorted(list(set(matches)), key = lambda i: int(i[2:]))

                    for i in reversed(unique_matches):
                        statement = statement.replace(i, f"[var{i[2:]}]")
                    unique_matches = sorted(list(set(matches)), key = lambda i: int(i[2:]))

                    for i, name in enumerate(unique_matches):
                        statement = statement.replace(f"[var{name[2:]}]", f"'a{i}")
                    return statement
                else:
                    return statement
            
            def simplify_decls(statement):
                decl_clauses, last_clause = statement.split("->")[:-1], statement.split("->")[-1]

                used_variables = set(re.findall("'a\\d+", last_clause))
                are_clauses_useful = [False for _ in decl_clauses]

                for _ in range(len(decl_clauses)):
                    for i, clause in enumerate(decl_clauses):
                        clause_vars = re.findall("'a\\d+", clause)
                        if clause_vars:
                            for var in clause_vars:
                                if(var in used_variables):
                                    are_clauses_useful[i] = True
                                    used_variables.update(clause_vars)

                decl_clauses = [decl_clauses[i].strip() for i in range(len(decl_clauses)) if are_clauses_useful[i]]
                recombined =  " -> ".join(decl_clauses + [last_clause.strip()])

                if "->" in recombined:
                    recombined = recombined if "[" in recombined else "[" + recombined
                    recombined = recombined if "]" in recombined else recombined + "]"
                else:
                    recombined = recombined.replace("[", "").replace("]", "").strip()
                
                return renumber_var_names(recombined)


            def get_seed_statement():
                if(np.random.random() > 0.5 and len(proven_conjectures) > 0):
                    # seed_conj = "[('a0: nat) -> ('a1: (= 'a0 z)) -> (= (s 'a0) o)]"
                    seed_conj = np.random.choice(proven_conjectures)
                    seed_conj = comp_to_raw_dict[str(seed_conj)]
                    seed_conj = simplify_decls(seed_conj)

                    matches = re.findall("'a\\d+", seed_conj)
                    if matches:
                        max_var_count = max(int(i[2:]) for i in matches)
                    else:
                        max_var_count = -1

                    decl_clauses, last_clause = seed_conj.split("->")[:-1], seed_conj.split("->")[-1][:-1]
                    last_clause = f" ('a{max_var_count + 1} :{last_clause}) "

                    seed = "->".join(decl_clauses + [last_clause])
                    seed = seed if "[" in seed else "[" + seed
                    return seed
                else:
                    seed = None
                #print(seed)
                return seed


            while len(conjectures) < cfg.n_conjectures:
                seed = get_seed_statement()
                proposal = sample_conjecture(AgentLM(agent, CONJECTURE_PROMPT), context, seed=seed)
                seed_cur = seed

                if proposal and proposal not in conjectures + proven_conjectures:
                    # Contract conjectures to make them Peano-parseable.
                    contracted_proposal = d.contract(proposal)
                    #print("contracted: " + str(contracted_proposal))
                    if contracted_proposal not in conjectures + proven_conjectures:
                        comp_to_raw_dict[str(contracted_proposal)] = proposal
                        seed_used[str(contracted_proposal)] = seed_cur

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

            print('Running proof search...')
            student_results: list[dict] = []
            success_logprobs = []
            examples = []
            lean_conjectures = [convert_arith(conjecture, index, flag_matters=False) for index, conjecture in enumerate(conjectures)]
            results = dsprover.prove(lean_conjectures)
            for k in range(len(results)):
                proof = results[k][0]
                logprob = results[k][1]
                if proof:
                    student_results.append({"problem": conjectures[k], "translated": lean_conjectures[k], "proof": proof, "logprob": logprob})
                else:
                    student_results.append({"problem": conjectures[k], "translated": lean_conjectures[k], "proof": None, "logprob": 0.0})
            
            print('test')
            end_search_time = time.time()
            # with open(f"proof_results_{i}.json", "r") as f:
            #     student_results = json.load(f)

            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            for student_result in student_results:
                if student_result["proof"]:
                    success_logprobs.append(student_result["logprob"])

                # TODO: Add usefulness back
                # if student_result.success:
                    
                #     for conj in useful_theorems:
                #         conj_name = conj.theorem.split(" : ")[0]
                #         if any([conj_name in r for r in student_result.proof]): #type: ignore
                #             conj.freq_used += 1
                print([f"{key} : {type(s)}" for key, s in student_result.items()])
                outcomes.append({'iteration': i,
                                'problem': student_result["problem"],
                                "problem_translated" : student_result["translated"],
                                'problem_raw': comp_to_raw_dict.get(str(student_result["problem"]), str(student_result["problem"])),
                                'proof': student_result["proof"],
                                'logprob': student_result["logprob"],
                                'seed_used': seed_used[str(student_result["problem"])]
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

                # Cut the least used theorems
                useful_theorems = [thm for thm in useful_theorems if not (i - thm.iter_generated >= 3 and thm.tot_improvement <= 1e-7)]
                useful_theorems.sort(key=lambda thm: thm.tot_improvement/(i-thm.iter_generated))
                useful_theorems = useful_theorems[len(useful_theorems)//10:]
                for thm in useful_theorems:
                    if thm.tot_improvement <= 1e-7:
                        continue
                    thm_arr = list(map(str, thm.theorem.split(" : ")[1:]))
                    thm_str = (" : ".join(thm_arr))[:-1]
                    to_add = f'Conj:(hard,useful) ' + d.elaborate(thm_str)
                    if to_add not in examples:
                        examples.append(to_add)
                    
            # Calculate usefulness
            if len(useful_theorems) == 0:
                hard_problems = {s["problem"]: float(s["logprob"]) for s in student_results if float(s["logprob"]) <= thresholds[-1]}
                pairs_tested: list[tuple[UsefulConjecture, str]] = []
                for hard_problem in hard_problems.keys():
                    potential_theorems = random.sample(useful_theorems, math.floor(math.sqrt(len(useful_theorems))))
                    pairs_tested.extend([(pt, hard_problem) for pt in potential_theorems])
                results_to_test = [a.theorem + " -> " + b for a,b in potential_theorems]
                results = dsprover.prove(results_to_test)
                for z in zip(pairs_tested, results):
                    pair = z[0]
                    res = z[1]
                    org_logprob = hard_problem[pair[1]]
                    pair[0].freq_used += 1
                    if logprob > org_logprob:
                        pair[0].tot_improvement += logprob - org_logprob
            
            # 3b- Classify problems into easy/hard.
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
                outcome = next(k
                            for i, (k, _) in enumerate(difficulty_buckets)
                            if (student_result["logprob"] <= thresholds[i] or
                                i + 1 == len(difficulty_buckets)))
                if ": nat" in student_result["problem"]:
                    useful_theorems.append(UsefulConjecture(student_result["problem"], i, 0, 0.0))
                # conjecture_index += 1
                if not cfg.get('freeze_conjecturer', False):
                    tags = [outcome]
                    # if student_result.problem.count("z") < 3: tags.append("few_zeros")
                    examples.append(f'Conj:({",".join(tags)}) ' + d.elaborate(student_result["problem"]))
                proven_conjectures.append(student_result["problem"])
                proofs.append(student_result["proof"])

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

            save_json([thm.to_dict() for thm in useful_theorems], f"generated_theorems_{i}.json")
            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            print(len(examples), 'accumulated training examples.')
            log.write(json.dumps({'iteration': i,
                                    'msg': f'Training on {len(examples)} examples.'}))
            
            torch.save(student_results, f'results_{i}.json')


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
