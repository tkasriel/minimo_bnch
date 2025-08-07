#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""


import asyncio
import math
import os
import json
import datetime
import random
import time

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

import peano
from classes import InstructionEnum, MPInstruction, MPResult, ProofOutcome, ProofOutcomeList, UsefulConjecture, UsefulConjectureList
from convert_to_lean import convert_arith
import worker
from worker import StudentResult
from util import format_blocks_with_indent, get_seed_statement, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import ProofSearchAgent, make_agent
import re

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"
CONJECTURE_PROMPT= 'Conj:(hard,useful) '

def process_main(id: int, cfg, agent: ProofSearchAgent, instruction_queue: mp.Queue, output_queue: mp.Queue):
    instruction: MPInstruction | None = None
    from dotenv import load_dotenv
    load_dotenv()
    
    # Unfortunately PyDerivation is not pickle-able and I don't know enough rust to fix that
    current_theory = ""
    background_theory = None
    d = None
    context = None
    print(f"Process {id} ready")

    while not instruction or instruction.instruction != InstructionEnum.STOP:
        instruction = MPInstruction.model_validate(instruction_queue.get())
        
        # Import new theory if needed
        if instruction.theory[0] != current_theory:
            current_theory = instruction.theory[0]
            background_theory = worker.BackgroundTheory(*instruction.theory)
            d = peano.PyDerivation() # type: ignore
            d.incorporate(current_theory)
            context = Context(d, None, [])
        
        # Process each instruction
        start_time = time.time()
        if instruction.instruction == InstructionEnum.CONJECTURE:
            assert type(instruction.previous_conjectures) is list
            seed_statement = instruction.seed
            previous_conjectures: list[str] = instruction.previous_conjectures
            new_conj = sample_conjecture(cfg, AgentLM(agent, CONJECTURE_PROMPT), context, previous_conjectures, seed=seed_statement)
            output_queue.put(MPResult(instruction=InstructionEnum.CONJECTURE, result=new_conj, seed=seed_statement, time_taken=time.time()-start_time))
        
        elif instruction.instruction == InstructionEnum.PROOF:
            assert instruction.thm_to_prove
            assert type(background_theory) is worker.BackgroundTheory
            result = worker.try_prove(cfg, agent, background_theory, instruction.thm_to_prove, verbose=False)
            output_queue.put(MPResult(instruction=InstructionEnum.PROOF, result=result, time_taken=time.time()-start_time))
    output_queue.put(MPResult(instruction=InstructionEnum.STOP, result=id, time_taken=0.0))

def batch_prove (cfg, conjectures: list[str], theory: str, premises: list[str], instruction_queue : mp.Queue, output_queue: mp.Queue) -> list[StudentResult]:
    for conjecture in conjectures:
        instruction_queue.put(MPInstruction(instruction=InstructionEnum.PROOF, thm_to_prove=conjecture, theory=(theory, premises)))
    
    # Process results
    num_dead = 0
    student_results = []
    progress_bar = tqdm(total=len(conjectures))
    while len(student_results) < len(conjectures) and num_dead < cfg.num_processes:
        mp_result = MPResult.model_validate(output_queue.get())
        if mp_result.instruction == InstructionEnum.CONJECTURE:
            # Leftover from conjecturing. We could use them, but for now I'll just throw them out
            continue
        if mp_result.instruction == InstructionEnum.STOP:
            num_dead += 1
            print(f"Process {mp_result.result} is done")
            continue
        assert mp_result.instruction == InstructionEnum.PROOF
        student_result = mp_result.result
        assert type(student_result) is StudentResult
        if student_result.error:
            print("Proof search errored.")
            print(student_result.error)
            continue
        if student_result.success:
            assert student_result.proof and student_result.solution_actions
            print("Proof search was a success! Proof actions taken:")
            print("\t"+"\n\t".join(student_result.solution_actions))
            print(f"logprob: {student_result.logprob}")
        else:
            print("Proof search failed")
        print(f"Got {len(student_result.hindsight_examples)} hindsight examples\n")
        student_results.append(student_result)
        progress_bar.update(1)
    progress_bar.close()
    return student_results

def batch_conjecture (cfg, proven_conjectures: list[str], comp_to_raw_dict: dict[str, str], permanent_deriv, seed_used: dict[str,str], theory: str, premises: list[str], instruction_queue: mp.Queue, output_queue: mp.Queue) -> list[str]:
    progress_bar = tqdm(total=cfg.n_conjectures)
    conjectures: list[str] = []
    failures = 0
    dups = 0
    for j in range(min(cfg.num_processes, cfg.n_conjectures)):
        instruction_queue.put(MPInstruction(instruction=InstructionEnum.CONJECTURE, 
                                            seed=get_seed_statement(cfg, proven_conjectures, comp_to_raw_dict), 
                                            theory=(theory, premises), 
                                            previous_conjectures=proven_conjectures))
    while len(conjectures) < cfg.n_conjectures:
        conjecture_result = MPResult.model_validate(output_queue.get())
        # assert type(conjecture_result.result) is str
        instruction_queue.put(MPInstruction(instruction=InstructionEnum.CONJECTURE, 
                                            seed=get_seed_statement(cfg, proven_conjectures, comp_to_raw_dict), 
                                            theory=(theory, premises), 
                                            previous_conjectures=conjectures + proven_conjectures))

        if conjecture_result.result and conjecture_result.result not in conjectures + proven_conjectures:
            # Contract conjectures to make them Peano-parseable.
            contracted_proposal: str = permanent_deriv.contract(conjecture_result.result)
            #print("contracted: " + str(contracted_proposal))
            if contracted_proposal not in conjectures + proven_conjectures:
                assert type(conjecture_result.result) is str
                comp_to_raw_dict[str(contracted_proposal)] = conjecture_result.result
                seed_used[str(contracted_proposal)] = conjecture_result.seed # type: ignore

                conjectures.append(contracted_proposal)
                progress_bar.update(1)
        #     else:
        #         dups += 1
        #         if dups % 10 == 0:
        #             print (f"Current dup count: {dups}")
        # else:
        #     if conjecture_result.result:
        #         dups += 1
        #         if dups % 10 == 0:
        #             print (f"Current dup count: {dups}")
        #     else:
        #         failures += 1
        #         if failures % 10 == 0:
        #             print (f"Current fail count: {failures}")
            
    progress_bar.close()
    return conjectures

def extract_examples(cfg, difficulty_buckets, permanent_deriv, seen_hindsight_goals, examples, student_result: StudentResult, thresholds, verbose=False):
    for h in student_result.hindsight_examples:
        if h.goal not in seen_hindsight_goals:
            outcome = next(k
                                        for i, (k, _) in enumerate(difficulty_buckets)
                                        if h.logprob <= thresholds[i] or i + 1 == len(difficulty_buckets))

            if not cfg.get('freeze_conjecturer', False):
                tags = [outcome]
                try:
                    examples.append(f'Conj:({",".join(tags)}) ' + permanent_deriv.elaborate(student_result.problem))
                except BaseException:
                    pass
            # if verbose: 
                # print (f"Adding {len(h.examples)} examples for {h.goal}")
            seen_hindsight_goals.add(h.goal)
            # for ex in h.examples:
            #     if "c0" in ex:
            #         print (ex)
            examples.extend(h.examples)
 

async def teacher_loop(cfg: DictConfig):
    if cfg.use_multiprocessing:
        mp.set_start_method(cfg.mp_start_method)

    agent = make_agent(cfg)

    # Debugging only
    theory_folder = "theories"
    if "output" in os.path.abspath(__file__):
        theory_folder = "../../../theories"
    with open(os.path.join(os.path.dirname(__file__), theory_folder, cfg.theory.name + '.p')) as f:
        theory = f.read()
    

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                                 for i in range(len(cfg.difficulty_buckets))],
                                key=lambda kv: kv[1])
    premises = cfg.theory.premises

    permanent_deriv = peano.PyDerivation() # type: ignore
    permanent_deriv.incorporate(theory)
    proven_conjectures: list[str] = []
    comp_to_raw_dict = dict()
    seed_used = {}
    seen_hindsight_goals = set()
    proofs = []
    outcomes: list[ProofOutcome] = []
    usefulness_outcomes = []
    useful_theorems: list[UsefulConjecture] = []
    # examples: ProofExamplesList = []

    continue_dir = cfg.get('continue')
    start_iteration = 0

    if continue_dir is not None:
        os.makedirs(continue_dir, exist_ok=True)
        os.chdir(continue_dir)
        print('Continuing run from', continue_dir)
        with open("flags.txt", "w") as f:
            f.write(str(cfg))
        # Find largest iteration number such that i.pt exists.
        i = 0
        while os.path.exists(f'{i}.pt'):
            i += 1
        i -= 1
        if i >= 0:
            # Load examples and outcomes.
            if os.path.exists(f"{i}.5.pt"):
                agent: ProofSearchAgent = torch.load(f'{i}.5.pt')
                start_iteration = i + 1
                with open(f"examples_{i}.json") as f:
                    examples = json.load(f)
                    agent.train(examples)
            else:
                start_iteration = i
                agent = torch.load(f'{i}.pt', map_location=None if torch.cuda.is_available() else "cpu", weights_only=False)
                print('Loaded agent from', f'{i}.pt')
        if i > 0:
            with open(f'outcomes_{i-1}.json', 'r') as f:
                j = json.load(f)
                outcomes = ProofOutcomeList.validate_python(j)
            proven_conjectures = [o.problem for o in outcomes
                                if o.hindsight is False and
                                    o.proof is not None]
            seen_hindsight_goals = {o.problem for o in outcomes
                                    if o.hindsight and o.proof is not None}
            comp_to_raw_dict = {o.problem: o.problem_raw for o in outcomes if o.problem_raw}
            with open(f"generated_theorems_{i-1}.json") as f:
                thms = json.load(f)
                useful_theorems = UsefulConjectureList.validate_python(thms)
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
                                                                 "cfg": cfg,
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
            context = Context(permanent_deriv, None, [])
            student_results: list[StudentResult] = []
            success_logprobs = []
            examples = []


            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')
            if cfg.use_multiprocessing:
                conjectures = batch_conjecture(cfg, proven_conjectures, comp_to_raw_dict, permanent_deriv, seed_used, theory, premises, instruction_queue, output_queue) # type: ignore
            else:
                conjectures: list[str] = []
                progress_bar = tqdm(total=cfg.n_conjectures)
                while len(conjectures) < cfg.n_conjectures:
                    seed = get_seed_statement(cfg, proven_conjectures, comp_to_raw_dict)
                    proposal = sample_conjecture(cfg, AgentLM(agent, CONJECTURE_PROMPT), context, conjectures + proven_conjectures, seed=seed)
                    seed_cur = seed

                    if proposal and proposal not in conjectures + proven_conjectures:
                        # Contract conjectures to make them Peano-parseable.
                        contracted_proposal: str = permanent_deriv.contract(proposal)
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

            # 2- Try to prove each of the conjectures

            print('Running proof search...')
            if cfg.use_multiprocessing:
                student_results = batch_prove(cfg, conjectures, theory, premises, instruction_queue, output_queue) # type: ignore # type: ignore
            else:
                for index, conjecture in enumerate(tqdm(conjectures, miniters=1)):
                    student_results.append(worker.try_prove(cfg, agent, background_theory, conjecture, True))
            end_search_time = time.time()

            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
            for student_result in student_results:
                if student_result.success:
                    success_logprobs.append(student_result.logprob)
                outcomes.append(ProofOutcome(
                                iteration=i,
                                problem=student_result.problem,
                                problem_raw=comp_to_raw_dict.get(student_result.problem, None),
                                problem_translated=str(convert_arith(student_result.problem, 0, False)),
                                proof=student_result.proof,
                                logprob=student_result.logprob,
                                actions=student_result.solution_actions,
                                hindsight=False,
                                seed_used=seed_used.get(student_result.problem)))
                if not student_result.hindsight_examples:
                    continue #This shouldn't happen, but it's happened in the past and I dont want it to crash the program.
                for h in student_result.hindsight_examples:
                    outcomes.append(ProofOutcome(
                                    iteration=i,
                                    problem=h.statement,
                                    problem_raw=None,
                                    seed_used=None,
                                    problem_translated=str(convert_arith(h.statement, 0, False)),
                                    proof=h.proof,
                                    logprob=h.logprob,
                                    actions=h.solution_actions,
                                    hindsight=True
                    ))
            save_json(outcomes, f'outcomes_{i}.json')

            if not success_logprobs:
                print(f'No solutions found in iteration {i}...')
                thresholds = [-2.906669173782248, -1.6445413855994306, -0.9526082203994054]
            #     break
            else:
                thresholds: list[float] = [np.percentile(success_logprobs, p)
                            for _, p in difficulty_buckets]

                print('Thresholds:',
                    list(zip([k for k, _ in difficulty_buckets], thresholds)),
                    'min =', np.min(success_logprobs),
                    'max =', np.max(success_logprobs))

            # 3b- Classify problems into easy/hard.
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
                if student_result.success:
                    outcome: str = next(k
                                for i, (k, _) in enumerate(difficulty_buckets)
                                if (student_result.logprob <= thresholds[i] or
                                    i + 1 == len(difficulty_buckets)))
                else:
                    outcome = "fail"
                

                if not cfg.get('freeze_conjecturer', False):
                    tags = [outcome]
                    
                    examples.append(f'Conj:({",".join(tags)}) ' + permanent_deriv.elaborate(student_result.problem))
                if student_result.success:
                    examples.append(student_result.problem)
                    proofs.append(student_result.proof)

                examples.extend(student_result.extracted_examples)

                if cfg.train_policy_on_hindsight_examples and student_result.hindsight_examples:
                    extract_examples(cfg, difficulty_buckets, permanent_deriv, seen_hindsight_goals, examples, student_result, thresholds)

            # Now we check for usefulness
            hard_theorems = [res for res in student_results if res.proof and res.logprob < thresholds[1]]
            print (len(hard_theorems), len(useful_theorems))
            if len(useful_theorems) > 0:
                print ("Beginning usefulness check")
                if cfg.max_proof_expansion < 0:
                    theorems_to_check = random.sample(useful_theorems, int(math.sqrt(len(useful_theorems))))
                else:
                    theorems_to_check = useful_theorems
                new_theory = theory + "\n\n" + "\n\n".join(map(lambda x: x.theorem, theorems_to_check))
                new_premises = premises + [thm.theorem.split(" : ")[0] for thm in theorems_to_check]
                hard_problems = [ht.problem for ht in hard_theorems]
                res = []
                if cfg.use_multiprocessing:
                    res = batch_prove(cfg, hard_problems, new_theory, new_premises, instruction_queue, output_queue) # type: ignore
                else:
                    bk = worker.BackgroundTheory(new_theory, new_premises)
                    for hard_theorem in tqdm(hard_theorems):
                        res.append(worker.try_prove(cfg, agent, bk, hard_theorem.problem))
                for proof_res, hard_theorem in zip(res, hard_theorems):
                    if proof_res.proof:
                        usefulness_outcomes.append({
                            "iteration": i,
                            "problem": convert_arith(hard_theorem.problem, 0, False),
                            "proof": proof_res.proof,
                            "used_theorems": list(map(lambda x: x.theorem, theorems_to_check)),
                            "improvement": proof_res.logprob - hard_theorem.logprob
                        })
                        if proof_res.logprob > hard_theorem.logprob:
                            if cfg.metric_use_useage:
                                for line in proof_res.proof:
                                    for thm in theorems_to_check:
                                        thm_name = thm.theorem.split(" : ")[0]
                                        if thm_name in line:
                                            improvement = proof_res.logprob - hard_theorem.logprob
                                            thm.tot_improvement += improvement
                                            thm.freq_used += 1
                                            break
                            else:
                                for thm in theorems_to_check:
                                    improvement = proof_res.logprob - hard_theorem.logprob
                                    thm.tot_improvement += improvement
                                    thm.freq_used += 1
                                        
                    # Train the model to use the old theorems.
                    if proof_res.hindsight_examples and cfg.train_on_usefulness_testing:
                        extract_examples(cfg, difficulty_buckets, permanent_deriv, seen_hindsight_goals, examples, proof_res, thresholds, verbose=True)
            useful_theorems = [thm for thm in useful_theorems if not (i - thm.iter_generated >= 3 and thm.tot_improvement <= 1e-7)]
            if cfg.metric_use_useage:
                useful_theorems.sort(key=lambda thm: (thm.tot_improvement)/(i-thm.iter_generated))
            else:
                useful_theorems.sort(key=lambda thm: (thm.tot_improvement/(thm.freq_used+1))/(i-thm.iter_generated))
            
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
                if student_result.success:
                    conjecture_index += 1
                    useful_theorems.append(UsefulConjecture(theorem=f"c{conjecture_index:04} : " + student_result.problem + ".",
                                                            iter_generated=i, 
                                                            freq_used=0,
                                                            tot_improvement=0.0))

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

            save_json(useful_theorems, f"generated_theorems_{i}.json")
            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            save_json(usefulness_outcomes, f'usefulness_outcomes_{i}.json')
            print(len(examples), 'accumulated training examples.')
            log.write(json.dumps({'iteration': i,
                                    'msg': f'Training on {len(examples)} examples.'}))
            
            torch.save(student_results, f'results_{i}.json')

    
    if cfg.prove_all_at_end:
        print ("Attempting final proof")
        final_outcomes: list[ProofOutcome] = []
        to_prove_again = [o.problem for o in outcomes if not o.hindsight]
        if cfg.use_multiprocessing:
            res = batch_prove(cfg, to_prove_again, theory, premises, instruction_queue, output_queue)
        else:
            res = []
            background_theory = worker.BackgroundTheory(theory, premises)
            for o in tqdm(to_prove_again):
                res.append(worker.try_prove(cfg, agent, background_theory, o))
        for r in res:
            final_outcomes.append(ProofOutcome(
                                iteration=-1,
                                problem=r.problem,
                                problem_raw=comp_to_raw_dict.get(r.problem, None),
                                problem_translated=str(convert_arith(r.problem, 0, False)),
                                proof=r.proof,
                                logprob=r.logprob,
                                actions=r.solution_actions,
                                hindsight=False,
                                seed_used=seed_used.get(r.problem)))
        save_json(final_outcomes, "final_outcomes.json")
        


    if cfg.use_multiprocessing:
        for process in processes: # type: ignore
            instruction_queue.put(MPInstruction(instruction=InstructionEnum.STOP, theory=("",[]))) # type: ignore
        for process in processes: # type: ignore
            process.join()
            process.close()        

@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
