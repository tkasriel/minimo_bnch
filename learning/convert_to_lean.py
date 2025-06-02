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
    if text.isnumeric() or text in ["Nat", ":", "->", "→", "false", "true", "Prop"]:
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

def _convert(conjecture: str, pre="") -> tuple[str, bool]:
    global currvar, vars

    if conjecture[0] == "(":
        # complex expression
        atoms = _find_atoms(conjecture[1:-1])
        if len(atoms) == 3 and atoms[1] != "->" and atoms[1] != "→":
            if atoms[1] == ":":
                return f"({_handle_singleton(atoms[0])} : {_convert(atoms[2])[0]})", False
            if atoms[0] in "+*=" or atoms[0] in ["∧", "∨", "↔"]:
                op1, flag1 = _convert(atoms[1])
                op2, flag2 = _convert(atoms[2])
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
                if atoms[0] == "=" and op1 == op2:
                    flag = True
                return f"({op1} {atoms[0]} {op2})", flag
                
            print(conjecture)
            sys.exit()
        if len(atoms) == 2 and atoms[0] == "Nat.succ":
            op, flag = _convert(atoms[1])
            return f"(Nat.succ {op})", flag
        if len(atoms) == 2 and atoms[0] == "not":
            op, flag = _convert(atoms[1])
            if op == "false":
                return "true", False
            if op == "true":
                return "false", False
            return f"(¬ {op})", flag
        if len(atoms) == 2:
            operator = _handle_singleton(atoms[0])
            op, _ = _convert(atoms[1])
            return f"({operator} {op})", False
        converted = []
        for atom in atoms:
            curr_conv, _ = _convert(atom)
            converted.append(curr_conv)
        return "(" + " ".join(converted) + ")", _ or (any([converted[-1] in converted[i] for i in range(0, len(converted)-2)]))

    else:
        # singular atom
        return _handle_singleton(conjecture), False

def convert_arith(conjecture: str, it: int, flag_matters: bool = True) -> str | bool:
    global vars, uses, currvar
    vars = {}
    uses = {}
    currvar = 0
    if ": nat" not in conjecture:
        return False
    og_conj = conjecture

    conjecture = conjecture.replace("z", "0").replace("o", "1").replace("nat", "Nat").replace("s", "Nat.succ").replace("'", "").replace("[", "(").replace("]",")").replace("pr1p", "Prop")
    out, flag = _convert(conjecture)
    if (flag or sum(uses.values()) == 0) and flag_matters:
        return False
    
    pre = "theorem problem" + str(it) + ": "
    return pre + out + ":= by\n"

def convert_prop(conjecture: str, it: int, flag_matters: bool = True) -> str | bool:
    og_conj = conjecture
    conjecture = conjecture.replace("iff", "↔").replace ("->", "→").replace("'", "").replace("[", "(").replace("]", ")").replace("prop", "Prop").replace("and", "∧").replace("or", "∨")
    if ": Prop" not in conjecture:
        return False
    out, flag = _convert (conjecture)
    if flag and flag_matters:
        return False
    
    pre = "theorem problem" + str(it) + ": "
    return pre + out + ":= by\n"
