= : [('t : type) -> 't -> 't -> prop].

nat : type.
false : prop.

z : nat.
s : [nat -> nat].

+ : [nat -> nat -> nat].
* : [nat -> nat -> nat].
^ : [nat -> nat -> nat].

#forward s.
#forward +.

/* Defining axioms for addition */
+_z : [('n : nat) -> (= (+ 'n z) 'n)].
+_s : [('n : nat) -> ('m : nat) -> (= (+ 'n (s 'm)) (s (+ 'n 'm)))].
#forward +_z ((+ 'n z) : nat).
#forward +_s ((+ 'n (s 'm)) : nat).

/* Defining axioms for multiplication */
*_z : [('n : nat) -> (= (* 'n z) z)].
*_s : [('n : nat) -> ('m : nat) -> (= (* 'n (s 'm)) (+ 'n (* 'n 'm)))].
#forward *_z ((* 'n z) : nat).
#forward *_s ((* 'n (s 'm)) : nat).

/* Defining axioms for exponentiation */
^_z : [('n : nat) -> (= (^ 'n z) (s z))].
^_s : [('n : nat) -> ('m : nat) -> (= (^ 'n (s 'm)) (* 'n (^ 'n 'm)))].
#forward ^_z ((^ 'n z) : nat).
#forward ^_s ((^ 'n (s 'm)) : nat).

/* Natural number induction */
nat_ind : [('p : [nat -> prop]) -> ('p z) -> [('n : nat) -> ('p 'n) -> ('p (s 'n))] -> [('n : nat) -> ('p 'n)]].
#backward nat_ind infer subgoal subgoal.

#forward succ_inj ((= (s 'a) (s 'b)) : 't).
succ_inj : [('a : nat) -> ('b : nat) -> (= (s 'a) (s 'b)) -> (= 'a 'b)].

leq : [nat -> nat -> prop].
lt : [nat -> nat -> prop].
gt : [nat -> nat -> prop].

#forward rewrite.
#forward eq_refl.
#forward eq_symm ((= 'a 'b) : 't).

and : [prop -> prop -> prop].

and_elim_l : [('p : prop) -> ('q : prop) -> (and 'p 'q) -> 'p].
#forward and_elim_l ('po : (and 'p 'q)).
and_elim_r : [('p : prop) -> ('q : prop) -> (and 'p 'q) -> 'q].
#forward and_elim_r ('po : (and 'p 'q)).
and_intro : [('p : prop) -> ('q : prop) -> 'p -> 'q -> (and 'p 'q)].
#backward and_intro infer infer subgoal subgoal.

or : [prop -> prop -> prop].

or_l : [('p : prop) -> ('q : prop) -> 'p -> (or 'p 'q)].
#backward or_l infer infer subgoal.

or_r : [('p : prop) -> ('q : prop) -> 'q -> (or 'p 'q)].
#backward or_r infer infer subgoal.

or_elim : [('p : prop) -> ('q : prop) -> (or 'p 'q) ->
           ('r : prop) -> ['p -> 'r] -> ['q -> 'r]
           -> 'r].
#backward or_elim infer infer infer infer subgoal subgoal.

false_elim : [('p : prop) -> false -> 'p].
#backward false_elim infer infer.

not : [prop -> prop].

/* Introduction rule for negation */
#backward not_i.
not_i : [('P : prop) -> ['P -> false] -> (not 'P)].
/* Elimination rule for negation */
not_e : [('P : prop) -> (not 'P) -> 'P -> false].
#backward exfalso.
exfalso : [false -> ('P : prop) -> 'P].

iff : [prop -> prop -> prop].

/* Introduction rules for equivalence */
#backward iff_i.
iff_i : [('P : prop) -> ('Q : prop) -> ['P -> 'Q] -> ['Q -> 'P] -> (iff 'P 'Q)].
/* Elimination rules for equivalence */
#forward iff_el ('_ : (iff 'P 'Q)).
iff_el : [('P : prop) -> ('Q : prop) -> (iff 'P 'Q) -> ['P -> 'Q]].
#forward iff_er ('_ : (iff 'P 'Q)).
iff_er : [('P : prop) -> ('Q : prop) -> (iff 'P 'Q) -> ['Q -> 'P]].

zero_ne_succ : [('a : nat) -> (not (= z (s 'a)))].
#forward zero_ne_succ.

#forward a_zero_add ((+ z 'n) : nat).
#forward a_succ_add ((+ (s 'a) 'b) : nat).
#forward a_add_assoc ((+ (+ 'a 'b) 'c) : nat).
/* #forward a_add_assoc ((+ 'a (+ 'b 'c)) : nat). */
#forward a_add_comm ((+ 'a 'b) : nat).
#forward a_succ_eq_add_one. /* ((s 'n) : nat). */
/* #forward a_succ_eq_add_one ((+ 'n (s z)) : nat). */
#forward a_add_right_comm ((+ (+ 'a 'b) 'c) : nat).

#forward m_zero_mul ((* z 'm) : nat).
#forward m_mul_one ((* 'm (s z)) : nat).
#forward m_one_mul ((* (s z) 'm) : nat).
#forward m_mul_add ((* 't (+ 'a 'b)) : nat).
#forward m_mul_assoc ((* (* 'a 'b) 'c) : nat).
#forward m_mul_comm ((* 'a 'b) : nat).


#forward succ_eq_succ_of_eq ('_ : (= 'a 'b)) ((s 'a) : nat) ((s 'b) : nat).