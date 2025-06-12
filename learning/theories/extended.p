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



exists : [('t : type) -> ('p : ['t -> prop]) -> prop].
ex_intro : [('t : type) -> ('p : ['t -> prop]) -> ('x : 't) -> ('p 'x) -> (exists 't 'p)].
ex_wit : [('t : type) -> ('p : ['t -> prop]) -> (exists 't 'p) -> 't].
ex_elim : [('t : type) -> ('p : ['t -> prop]) -> ('e : (exists 't 'p)) -> ('p (ex_wit 't 'p 'e))].

#backward ex_intro infer infer infer subgoal.
#forward ex_wit.
#forward ex_elim ('e : (exists 't 'p)).

leq : [nat -> nat -> prop] = (lambda ('a : nat, 'b : nat)
                                     (exists nat (lambda ('c : nat) (= (+ 'a 'c) 'b)))).

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

not : [prop -> prop] = (lambda ('p0 : prop) ['p0 -> false]).

iff : [prop -> prop -> prop] = (lambda ('p1 : prop, 'p2 : prop) (and ['p1 -> 'p2] ['p2 -> 'p1])).

zero_ne_succ : [('a : nat) -> (not (= z (s 'a)))].
#forward zero_ne_succ.

empty : type.

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