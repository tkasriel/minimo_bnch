from conjecture import Context, App
from peano import PyDerivation
import re

def test_app_parse():
    def has_trivial_outcome(conjecture):
        # Find the statement after the last "->"
        parts = conjecture.split("->")
        if len(parts) <= 1:
            return True  # Not enough arrows, probably incomplete
            
        # check if it only involves constant values and operators
        last_statement = parts[-1].strip()
        if ']' in last_statement:
            last_statement = last_statement[:last_statement.index(']')].strip()
        print(f"Last statement: {last_statement}")
        
        tokens = set([ch for ch in last_statement])
        constant_tokens = set(['z', ' ', '+', '*', 'o', '(', ')', '=', 's', ']', '.'])
        non_trivial_tokens = tokens - constant_tokens
        print(non_trivial_tokens)
        if not non_trivial_tokens:
            return True  # only involves trivial tokens
        
        # matches trivial arithmetic identities (modulo symmetry)
        trvial_identities = [
            r"\(= +\(\* +'[a-zA-Z0-9_]+ +z\) +z\)",   # (* n z) = z
            r"\(= +z +\(\* +'[a-zA-Z0-9_]+ +z\)\)",   # z = (* n z)
            r"\(= +\(\+ +'[a-zA-Z0-9_]+ +z\) +'[a-zA-Z0-9_]+\)",  # (+ n z) = n
            r"\(= +'[a-zA-Z0-9_]+ +\(\+ +'[a-zA-Z0-9_]+ +z\)\)",  # n = (+ n z)
            r"\(= +\(\* +o +'[a-zA-Z0-9_]+\) +'[a-zA-Z0-9_]+\)",  # (* o n) = n
            r"\(= +'[a-zA-Z0-9_]+ +\(\* +o +'[a-zA-Z0-9_]+\)\)",  # n = (* o n)
            r"\(= +'[a-zA-Z0-9_]+ +'[a-zA-Z0-9_]+\)",  # (= 'a0 'a0)
        ]
        for pattern in known_identities:
            if re.fullmatch(pattern, last_statement):
                return True
            
        # tautology — conclusion appeared earlier
        prior_string = "->".join(parts[:-1])
        if last_statement in prior_string:
            return True
            
        return False
    
    trivial_patterns = [
            "[('a0 : nat) -> ('a1 : nat) -> ('a2 : (= o 'a0)) -> ('a3 : (= 'a0 o)) -> ('a4 : nat) -> (= o 'a0)]",  # one equals one
            "[('a0 : (= z o)) -> ('a1 : nat) -> ('a2 : (= o o)) -> (= z o)]",
            "[('a0 : nat) -> (= (* (+ (+ (+ z z) o) z) z) z)]",
            "[('a0 : nat) -> ('a1 : (= 'a0 'a0)) -> ('a2 : nat) -> (= 'a0 'a0)]",
            "[('a0 : nat) -> (= z (* 'a0 z))]"
        ]
    
    for conjecture in trivial_patterns:
        if has_trivial_outcome(conjecture):
            print(f"Conjecture '{conjecture}' has a trivial outcome.")
        else:
            print(f"Conjecture '{conjecture}' does not have a trivial outcome.")


if __name__ == "__main__":
    test_app_parse()


# (a0: nat) -> (a1: nat) -> (a2: (= o a0)) -> ... -> (a3: (= a0 o)) -> (a4: nat) -> (= z z)
# log-prob low : model think it's good and it's very very hard