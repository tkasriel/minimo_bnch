#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""
from dotenv import load_dotenv
load_dotenv()

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
from convert_to_lean import convert_arith
from problems import load_natural_number_game_problemset
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, UsefulConjecture, sample_conjecture
from proofsearch import ProofSearchAgent, make_agent
import re

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"
CONJECTURE_PROMPT= 'Conj:(hard,useful) '
STOP = "STOP"
CONJECTURE = "CONJECTURE"
PROOF = "PROOF"

def process_main(id: int, agent: ProofSearchAgent, instruction_queue: mp.Queue, output_queue: mp.Queue):
    instruction: tuple[str,str] = ("","")
    
    # Unfortunately PyDerivation is not pickle-able and I don't know enough rust to fix that
    current_theory = ""
    background_theory = None
    d = None
    context = None
    print(f"Process {id} ready")

    while instruction[0] != STOP:
        instruction = instruction_queue.get()
        # input will have the following:
        # (INSTRUCTION, input, theory, addtn args)
        # We'll first process the third
        if instruction[2][0] != current_theory:
            current_theory = instruction[2][0]
            background_theory = worker.BackgroundTheory(*instruction[2])
            d = peano.PyDerivation()
            d.incorporate(current_theory)
            context = Context(d, None, [])
        # print(f"Received instruction: {instruction}")
        if instruction and type(instruction) is tuple:
            if instruction[0] == CONJECTURE:
                seed_statement = instruction[1]
                previous_conjectures = instruction[3]
                new_conj = sample_conjecture(AgentLM(agent, CONJECTURE_PROMPT), context, previous_conjectures, seed=seed_statement)
                output_queue.put((CONJECTURE, new_conj, seed_statement))
            elif instruction[0] == PROOF:
                thm_to_prove = instruction[1]
                result = worker.try_prove(agent, background_theory, thm_to_prove, verbose=False)
                output_queue.put((PROOF, result, None))
    output_queue.put((STOP, id, None))

def batch_prove (cfg, conjectures: list[str], theory: str, premises: list[str], instruction_queue : mp.Queue, output_queue: mp.Queue) -> list[StudentResult]:
    for conjecture in conjectures:
        instruction_queue.put((PROOF, conjecture, (theory, premises)))
    
    # Process results
    num_dead = 0
    student_results = []
    progress_bar = tqdm(total=len(conjectures))
    while len(student_results) < len(conjectures) and num_dead < cfg.num_processes:
        res_type, res, _ = output_queue.get()
        if res_type != PROOF:
            # Leftover from conjecturing. We could use them, but for now I'll just throw them out
            continue
        if res_type == STOP:
            num_dead += 1
            print(f"Process {res} is done")
            continue
        assert res_type == PROOF
        if res.error:
            print("Proof search errored.")
            print(res.error)
            continue
        if res.success:
            assert res.proof
            print("Proof search was a success! Proof actions taken:")
            print("\t"+"\n\t".join(res.solution_actions))
            print(f"logprob: {res.logprob}")
        else:
            print("Proof search failed")
        print(f"Got {len(res.hindsight_examples)} hindsight examples\n")
        student_results.append(res)
        progress_bar.update(1)
    progress_bar.close()
    return student_results


async def teacher_loop(cfg: DictConfig):
    if cfg.use_multiprocessing:
        mp.set_start_method(cfg.mp_start_method)

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
    usefulness_outcomes = []
    useful_theorems: list[UsefulConjecture] = []

    continue_dir = cfg.get('continue')
    start_iteration = 0
    # continue_dir = "/Users/tkasriel/code/rsh/minimo/learning/outputs/2025-01-28/12-49-46"

    if continue_dir is not None:
        os.makedirs(continue_dir, exist_ok=True)
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
                                        if o['hindsight'] is False and
                                            o['proof'] is not None]
                    seen_hindsight_goals = {o['problem'] for o in outcomes
                                            if o['hindsight'] and o['proof'] is not None}
                    comp_to_raw_dict = {o['problem']: o['problem_raw'] for o in outcomes if "problem_raw" in o.keys()}
                with open(f"generated_theorems_{i-1}.json") as f:
                    thms = json.load(f)
                    useful_theorems = [UsefulConjecture(**thm) for thm in thms]
                with open(f"usefulness_outcomes_{i-1}.json") as f:
                    usefulness_outcomes = json.load(f)

        print('Loaded', len(proven_conjectures), 'proven conjectures from previous run.')


    if cfg.get('freeze_conjecturer', False):
        print('Ablation: Freezing conjecturer.')


    conjecture_index = len(useful_theorems)
    if cfg.use_multiprocessing:
        # Start multiprocessing
        num_processes = cfg.num_processes
        instruction_queue: mp.Queue = mp.Queue()
        output_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []
        agent.share_memory()
        for j in range(num_processes):
            new_process = mp.Process(target=process_main,kwargs={"id": j,
                                                                "agent":agent, 
                                                                "instruction_queue": instruction_queue, 
                                                                "output_queue": output_queue})
            new_process.start()
            processes.append(new_process)
    with open('log.jsonl', 'w') as log:
        
        for i in range(start_iteration, cfg.iterations):
            start_time = time.time()
            torch.save(agent, f'{i}.pt')
            print(f"current it: {i}")
            background_theory = worker.BackgroundTheory(theory, premises)
            # if i > 0:
                # d.incorporate("\n\n".join(map(lambda x: x.theorem, useful_theorems)))
            context = Context(permanent_deriv, None, [])
            

            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')
            progress_bar = tqdm(total=cfg.n_conjectures)
            conjectures: list[str] = []

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
                # return None
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

            if cfg.use_multiprocessing:
                for j in range(min(num_processes, cfg.n_conjectures)):
                    instruction_queue.put((CONJECTURE, get_seed_statement(), (theory, premises), proven_conjectures)) # You can put the seed statement here

            while len(conjectures) < cfg.n_conjectures:
                if cfg.use_multiprocessing:
                    _, proposal, seed_cur = output_queue.get()
                    instruction_queue.put((CONJECTURE, get_seed_statement(), (theory, premises), conjectures + proven_conjectures))
                else:
                    seed = get_seed_statement()
                    # print(seed)
                    proposal = sample_conjecture(AgentLM(agent, CONJECTURE_PROMPT), context, conjectures + proven_conjectures, seed=seed)
                    seed_cur = seed
                # if proposal in conjectures + proven_conjectures:
                #     print ("Duplicate entry: " + proposal)
                # if not proposal:
                #     print ("Failed")
                if proposal and proposal not in conjectures + proven_conjectures:
                    # Contract conjectures to make them Peano-parseable.
                    contracted_proposal: str = permanent_deriv.contract(proposal)
                    #print("contracted: " + str(contracted_proposal))
                    if contracted_proposal not in conjectures + proven_conjectures:
                        comp_to_raw_dict[str(contracted_proposal)] = proposal
                        seed_used[str(contracted_proposal)] = seed_cur

                        conjectures.append(contracted_proposal)
                        progress_bar.update(1)
                    # else:
                    #     print("Duplicate entry (2): " + contracted_proposal)
                

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
                student_results = batch_prove(cfg, conjectures, theory, premises, instruction_queue, output_queue)
            else:
                for index, conjecture in enumerate(tqdm(conjectures, miniters=1)):
                    student_results.append(worker.try_prove(agent, background_theory, conjecture, True))
            end_search_time = time.time()

            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            for student_result in student_results:
                if student_result.success:
                    success_logprobs.append(student_result.logprob)
                    for conj in useful_theorems:
                        conj_name = conj.theorem.split(" : ")[0]
                        if any([conj_name in r for r in student_result.proof]): #type: ignore
                            conj.freq_used += 1


                outcomes.append({'iteration': i,
                                'problem': student_result.problem,
                                'problem_raw': comp_to_raw_dict[str(student_result.problem)],
                                'problem_translated': convert_arith(student_result.problem, 0, False),
                                'proof': student_result.proof,
                                'logprob': student_result.logprob,
                                'actions': student_result.solution_actions,
                                'hindsight': False,
                                'seed_used': seed_used[str(student_result.problem)]
                                })
                if not student_result.hindsight_examples:
                    continue #This shouldn't happen, but it's happened in the past and I dont want it to crash the program.
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
                    tags = [outcome]
                    # if ": nat" in student_result.problem: tags.append("useful")
                    # if student_result.problem.count("z") < 3: tags.append("few_zeros")
                    
                    examples.append(f'Conj:({",".join(tags)}) ' + permanent_deriv.elaborate(student_result.problem))
                if student_result.success:
                    examples.append(student_result.problem)
                    #proven_conjectures.append(student_result.problem_no_seed)
                    proofs.append(student_result.proof)

                examples.extend(student_result.extracted_examples)

                long_ex = []
                if os.path.exists("long_examples.json"):
                    with open("long_examples.json", "r") as f:
                        long_ex = json.load(f)
                if cfg.train_policy_on_hindsight_examples and student_result.hindsight_examples:
                    extract_examples(cfg, difficulty_buckets, permanent_deriv, seen_hindsight_goals, examples, student_result, thresholds, long_ex)

            # Now we check for usefulness
            hard_theorems = [res for res in student_results if res.proof and res.logprob < thresholds[1]]
            print ("Beginning usefulness check")
            if len(useful_theorems) > 0:
                theorems_to_check = random.sample(useful_theorems, int(math.sqrt(len(useful_theorems))))
                new_theory = theory + "\n\n" + "\n\n".join(map(lambda x: x.theorem, theorems_to_check))
                new_premises = premises + [thm.theorem.split(" : ")[0] for thm in theorems_to_check]
                hard_problems = [ht.problem for ht in hard_theorems]
                res = []
                if cfg.use_multiprocessing:
                    res = batch_prove(cfg, hard_problems, new_theory, new_premises, instruction_queue, output_queue)
                    
                else:
                    bk = worker.BackgroundTheory(new_theory, new_premises)
                    for hard_theorem in hard_theorems:
                        res.append(worker.try_prove(agent, bk, hard_theorem.problem))
                for proof_res, hard_theorem in tqdm(zip(res, hard_theorems)):
                    if proof_res.proof:
                        usefulness_outcomes.append({
                            "iteration": i,
                            "problem": convert_arith(hard_theorem.problem, 0, False),
                            "proof": proof_res.proof,
                            "used_theorems": list(map(lambda x: x.theorem, theorems_to_check)),
                            "improvement": proof_res.logprob - hard_theorem.logprob
                        })
                        if proof_res.logprob > hard_theorem.logprob:
                            for thm in theorems_to_check:
                                improvement = proof_res.logprob - hard_theorem.logprob
                                thm.tot_improvement += improvement
                                thm.freq_used += 1
                            # for line in proof_res.proof:
                            #     for thm in theorems_to_check:
                            #         thm_name = thm.theorem.split(" : ")[0]
                            #         if thm_name in line:
                            #             improvement = proof_res.logprob - hard_theorem.logprob
                            #             thm.tot_improvement += improvement
                            #             thm.freq_used += 1
                            #             break
                                        
                    # Train the model to use the old theorems.

                    extract_examples(cfg, difficulty_buckets, permanent_deriv, seen_hindsight_goals, examples, proof_res, thresholds, long_ex, verbose=True)
            useful_theorems = [thm for thm in useful_theorems if not (i - thm.iter_generated >= 3 and thm.tot_improvement <= 1e-7)]
            useful_theorems.sort(key=lambda thm: thm.freq_used/(i-thm.iter_generated))
            useful_theorems = useful_theorems[len(useful_theorems)//10:]
            
            for thm in useful_theorems:
                if thm.tot_improvement <= 1e-7:
                    continue
                thm_arr = list(map(str, thm.theorem.split(" : ")[1:]))

                thm_str = (" : ".join(thm_arr))[:-1]
                to_add = f'Conj:(useful) ' + permanent_deriv.elaborate(thm_str)
                if not to_add in examples:
                    examples.append(to_add)
            end_usefulness_time = time.time()
            
            for student_result in student_results:
                if student_result.success and ": nat" in student_result.problem:
                    conjecture_index += 1
                    useful_theorems.append(UsefulConjecture(f"c{conjecture_index:04} : " + student_result.problem + ".", i, 0, 0.0))

            log.write(json.dumps({'iteration': i,
                                'msg': f'Training on {len(examples)} examples.'}))
            log.write('\n')

            # 3c- Train model on conjecturing and proof search examples.
            print(len(examples), 'accumulated training examples.')
            agent.train(examples)
            train_end_time = time.time()
            with open("time_metric.txt", "a+") as tm:
                tm.write(f"Iteration {i}: \n")
                tm.write(f"Conjecturing took {end_conjecture_time-start_time}s\n")
                tm.write(f"Proof search took {end_search_time-end_conjecture_time}s\n")
                tm.write(f"Usefulness check took {end_usefulness_time-end_search_time}s\n")
                tm.write(f"Training took {train_end_time-end_usefulness_time}s\n")
                tm.write(f"Total time taken: {train_end_time-start_time}s\n")

            save_json([thm.to_dict() for thm in useful_theorems], f"generated_theorems_{i}.json")
            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            save_json(usefulness_outcomes, f'usefulness_outcomes_{i}.json')
            print(len(examples), 'accumulated training examples.')
            log.write(json.dumps({'iteration': i,
                                    'msg': f'Training on {len(examples)} examples.'}))
            
            torch.save(student_results, f'results_{i}.json')
    for process in processes:
        instruction_queue.put((STOP, "", None))
    for process in processes:
        process.join()
        process.close()

def extract_examples(cfg, difficulty_buckets, permanent_deriv, seen_hindsight_goals, examples, student_result: StudentResult, thresholds, long_ex, verbose=False):
    for h in student_result.hindsight_examples:
        if h.goal not in seen_hindsight_goals:
            if len(h.proof) > 4:
                long_ex.append({
                                    "goal": str(h.goal),
                                    "proof": h.proof,
                                    "length": str(len(h.proof))
                                })
            outcome = next(k
                                        for i, (k, _) in enumerate(difficulty_buckets)
                                        if h.logprob <= thresholds[i] or i + 1 == len(difficulty_buckets))

            if not cfg.get('freeze_conjecturer', False):
                tags = [outcome]
                try:
                    examples.append(f'Conj:({",".join(tags)}) ' + permanent_deriv.elaborate(student_result.problem))
                except BaseException:
                    pass
            if verbose: 
                print (f"Adding {len(h.examples)} examples for {h.goal}")
            seen_hindsight_goals.add(h.goal)
            # for ex in h.examples:
            #     if "c0" in ex:
            #         print (ex)
            examples.extend(h.examples)
        



@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
