import json
import os
import sys
from typing import Any, Callable

import tqdm

vars: dict[str, str] = {}
uses: dict[str, int] = {}
currvar = 0

def _find_atom(text: str) -> int:
    depth = 0
    for i in range(0, len(text)):
        depth += text[i].count("(") + text[i].count("[")
        depth -= text[i].count(")") + text[i].count("]")
        if depth < 0 or (depth == 0 and text[i] == " "):
            return i
    return len(text)

def _find_atoms (line: str) -> list[str]:
    out: list[str] = []
    while line:
        end_ind = _find_atom(line)
        out.append(line[:end_ind])
        line = line[end_ind+1:].lstrip()
    return out



def _handle_singleton(text: str) -> str:
    if text.isnumeric() or text in ["Nat", ":", "->", "→", "false", "true", "Prop", "G"]:
        return text
    global vars, uses, currvar
    if text in vars:
        if text not in uses:
            uses[text] = 0
        uses[text] += 1
        return vars[text]
    else:
        vars[text] = f"v{currvar}"
        currvar += 1
        return vars[text]

def _convert(conjecture: str, pre="", simplify: bool = True) -> tuple[str, bool]:
    global currvar, vars

    if conjecture[0] == "(":
        # complex expression
        atoms = _find_atoms(conjecture[1:-1])
        if len(atoms) == 3 and atoms[1] != "->" and atoms[1] != "→":
            if atoms[1] == ":":
                return f"({_handle_singleton(atoms[0])} : {_convert(atoms[2], simplify=simplify)[0]})", False
            if atoms[0] in "+*=" or atoms[0] in ["∧", "∨", "↔", "•"]:
                op1, flag1 = _convert(atoms[1], simplify=simplify)
                op2, flag2 = _convert(atoms[2], simplify=simplify)
                if simplify:
                    if op1 == "false" or op2 == "false" and atoms[0] == "∧":
                        return "false", False
                    if op1 == "true" or op2 == "true" and atoms[0] == "∨":
                        return "true", False
                    if op1 == "0" or op2 == "0":
                        if atoms[0] == "*":
                            return "0", False
                        if atoms[0] == "+":
                            return (op2 if op1 == "0" else op1), False
                flag = flag1 or flag2
                if simplify:
                    if atoms[0] == "=" and op1 == op2:
                        flag = True
                return f"({op1} {atoms[0]} {op2})", flag
                
            print(conjecture)
            sys.exit()
        if len(atoms) == 2 and atoms[0] == "Nat.succ":
            op, flag = _convert(atoms[1], simplify=simplify)
            return f"(Nat.succ {op})", flag
        if len(atoms) == 2 and atoms[0] == "not":
            op, flag = _convert(atoms[1], simplify=simplify)
            if simplify:
                if op == "false":
                    return "true", False
                if op == "true":
                    return "false", False
            return f"(¬ {op})", flag
        if len(atoms) == 2 and atoms[0] == "inv":
            op, flag = _convert(atoms[1], simplify=simplify)

            return f"({op}⁻¹)", flag

        if len(atoms) == 2:
            operator = _handle_singleton(atoms[0])
            op, _ = _convert(atoms[1], simplify=simplify)
            return f"({operator} {op})", False
        converted = []
        for atom in atoms:
            curr_conv, _ = _convert(atom, simplify=simplify)
            converted.append(curr_conv)
        return "(" + " ".join(converted) + ")", _ or (any([converted[-1] in converted[i] for i in range(0, len(converted)-2)]))

    else:
        # singular atom
        return _handle_singleton(conjecture), False

def convert_arith(conjecture: str, it: int, simplify: bool=True) -> str | bool:
    global vars, uses, currvar
    vars = {}
    uses = {}
    currvar = 0
    og_conj = conjecture

    conjecture = conjecture.replace("z", "0").replace("o", "1").replace("nat", "Nat").replace("s", "Nat.succ").replace("'", "").replace("[", "(").replace("]",")").replace("pr1p", "Prop")
    out, flag = _convert(conjecture, simplify=simplify)
    return out

def convert_prop(conjecture: str, it: int, simplify: bool = True) -> str | bool:
    global vars, uses, currvar
    vars = {}
    uses = {}
    currvar = 0

    og_conj = conjecture
    conjecture = conjecture.replace("iff", "↔").replace ("->", "→").replace("'", "").replace("[", "(").replace("]", ")").replace("prop", "Prop").replace("and", "∧").replace("or", "∨")
    out, flag = _convert (conjecture, simplify=simplify)
    
    return out

def convert_group(conjecture: str, it: int, simplify: bool = True) -> str | bool:
    global vars, uses, currvar
    vars = {}
    uses = {}
    currvar = 0

    og_conj = conjecture
    conjecture = conjecture.replace("op", "•").replace("id", "1").replace("'", "").replace("[", "(").replace("]", ")")
    out, flag = _convert (conjecture, simplify=simplify)
    
    return out

def convert_peano_to_lean(conjecture: str, it: int, simplify: bool=True, theory_string: str = "nat-mul"):
    if("nat" in theory_string):
        return convert_arith(conjecture, it, simplify)
    elif ("prop" in theory_string):
        return convert_prop(conjecture, it, simplify)
    elif ("group" in theory_string):
        return convert_group(conjecture, it, simplify)
    else:
        raise Exception("Unsupported theory for Lean4 conversion")

if __name__ == "__main__":
    nng = [
        "[(x : nat) -> (= z (* z x))]",
        "[('a : nat) -> (= (+ 'a (s z)) (s 'a))]",
        "[('m : nat) -> (= (* z 'm) z)]",
        "[('x : nat) -> ('y : nat) -> ('z : nat) -> (= (+ (* x 'y) z) (+ (* x 'y) z))]",
        "[('x : nat) -> ('y : nat) -> (= 'y (+ 'x n7)) -> (= (* n2 'y) (* n2 (+ 'x n7)))]",
        "[('a : nat) -> ('b : nat) -> (= (s 'a) 'b) -> (= (s (s 'a)) (s 'b))]",
        "[('n : nat) -> (= (+ z 'n) 'n)]",
        "[('a : nat) -> ('b : nat) -> ('c : nat) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))]",
        "[('a : nat) -> ('b : nat) -> (= (+ (s 'a) 'b) (s (+ 'a 'b)))]",
        "[('n : nat) -> (= (s 'n) (+ 'n (s z)))]",
        "[('m : nat) -> (= (* 'm (s z)) 'm)]",
        "(= z (+ z z))"
    ]

    conjs_prop = [
        "(and false (iff (or (iff (iff (and (or (or (not false) false) false) false) (iff (not (iff false false)) false)) (or (and false false) false)) false) (and (not (and (or (and (iff (iff false false) (not (or (not (not (not false))) false))) false) false) false)) false)))",
        "(and false (or (and (and (and false (and false false)) false) (and false (and (iff false (iff (or (not false) (or (iff false false) false)) (not false))) false))) (not false)))",
        "[('a0 : (iff false false)) -> (and false (and false false))]",
        "[('a0 : prop) -> false]",
        "[('a0 : false) -> false]",
        "[('a0 : (iff (and (iff false (iff false false)) (iff (not false) (not (and false (or (or false false) false))))) false)) -> ('a1 : prop) -> false]",
        "[('P : prop) -> ('Q : prop) -> 'P -> 'Q -> (and 'P 'Q)]",
        "[('P : prop) -> ('Q : prop) -> (and 'P 'Q) -> 'P]",
        "[('P : prop) -> ('Q : prop) -> (and 'P 'Q) -> 'Q]",
        "[('P : prop) -> ('Q : prop) -> 'P -> (or 'P 'Q)]",
        "[('P : prop) -> ('Q : prop) -> 'Q -> (or 'P 'Q)]",
        "[('P : prop) -> ['P -> false] -> (not 'P)]",
        "[('P : prop) -> (not 'P) -> 'P -> false]",
        "[false -> ('P : prop) -> 'P]",
        "[('P : prop) -> ('Q : prop) -> ['P -> 'Q] -> ['Q -> 'P] -> (iff 'P 'Q)]",
        "[('P : prop) -> ('Q : prop) -> (iff 'P 'Q) -> ['P -> 'Q]]",
        "[('P : prop) -> ('Q : prop) -> (iff 'P 'Q) -> ['Q -> 'P]]",
        "[('P : prop) -> (or 'P (not 'P))]"
    ]
    
    conjs_group = [
    "[('a0 : G) -> (= (op id (op 'a0 'a0)) (inv 'a0))]",
    "[('a0 : G) -> (= (op id (inv id)) id)]",
    "[('a0 : (= (inv id) (inv id))) -> (= (inv id) (op id id))]",
    "(= id id)",
    "[('a0 : (= (inv (op (op (inv id) (inv id)) id)) id)) -> ('a1 : (= id (inv id))) -> ('a2 : G) -> (= id (inv (op (inv 'a2) 'a2)))]",
    "[('a0 : G) -> ('a1 : G) -> ('a2 : (= (op 'a0 id) id)) -> (= id id)]",
    "[('a0 : (= (op (op (inv (op id (op id (inv id)))) (op id id)) id) id)) -> ('a1 : G) -> (= 'a1 id)]",
    "[('a : G) -> ('b : G) -> ('c : G) -> (= (op (op 'a 'b) 'c) (op 'a (op 'b 'c)))]",
    "[('a : G) -> ('b : G) -> (= (op 'a 'b) (op 'b 'a))]",
    "[('a : G) -> (= (op id 'a) 'a)]",
    "[('a : G) -> (= (op (inv 'a) 'a) id)]"
    ]

    output = [convert_peano_to_lean(c, i, False, "propositional-logic") for i, c in enumerate(conjs_prop)]
    [print(i) for i in output]