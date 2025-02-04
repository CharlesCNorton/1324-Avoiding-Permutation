From mathcomp Require Import all_ssreflect.
From mathcomp Require Import ssrfun ssrbool eqtype ssrnat seq.
Require Import Coq.Reals.Reals.
Require Import Coq.Lists.List.
Import ListNotations.

Open Scope R_scope.
Set Implicit Arguments.

(* --- Definition of Formal Power Series --- *)
Definition fps := nat -> R.

Definition fps_zero : fps := fun _ => 0.
Definition fps_one : fps :=
  fun n => match n with | 0 => 1%R | _ => 0%R end.

Definition fps_add (f g : fps) : fps :=
  fun n => f n + g n.

Definition fps_mul (f g : fps) : fps :=
  fun n =>
    let fix conv (i : nat) : R :=
      match i with
      | O => f (0%nat) * g n
      | S i' => f (S i') * g (Nat.sub n (S i')) + conv i'
      end in conv n.

Definition fps_shift (f : fps) : fps :=
  fun n => match n with | 0%nat => 0 | S n' => f n' end.

Definition fps_scale (a : R) (f : fps) : fps :=
  fun n => a * f n.

Definition fps_eq (f g : fps) : Prop :=
  forall n : nat, f n = g n.

Fixpoint sum_f_R0 (f : nat -> R) (n : nat) : R :=
  match n with
  | 0 => 0
  | S n' => f n' + sum_f_R0 f n'
  end.

Fixpoint fps_pow (f : fps) (n : nat) : fps :=
  match n with
  | 0 => fps_one
  | S n' => fps_mul f (fps_pow f n')
  end.

(* A finite approximation of the geometric series for f *)
Definition fps_geo (f : fps) : fps :=
  fun n => sum_f_R0 (fun i => (fps_pow f i) n) (S n).

Definition fps_sub (f g : fps) : fps :=
  fun n => f n - g n.

Definition fps_monomial (k : nat) : fps :=
  fun n => if n =? k then 1%R else 0%R.

(* --- The Generating Function Equation --- *)
(* This corresponds to:
     G(x) = 1 + x·G(x) + (x²·G(x)²)/(1 − x·G(x)) + x³·Q(x)
*)
Definition generating_eq (G Q : fps) : Prop :=
  fps_eq G (fps_add fps_one (fps_add (fps_mul (fps_monomial 1) G)
    (fps_add (fps_mul (fps_monomial 2)
      (fps_mul (fps_mul G G) (fps_geo (fps_mul (fps_monomial 1) G))))
    (fps_mul (fps_monomial 3) Q)))).

(* --- Candidate Definitions --- *)
Definition G_candidate : fps :=
  fun n =>
    match n with
    | 0 => 1%R
    | 1 => 1%R
    | 2 => 2%R
    | 3 => 6%R
    | 4 => 23%R
    | 5 => 103%R
    | 6 => 513%R
    | 7 => 2762%R
    | 8 => 15793%R
    | _ => 0%R
    end.

Definition Q_candidate : fps :=
  fun n =>
    match n with
    | 0 => 1%R
    | 1 => 8%R
    | 2 => 50%R
    | 3 => 297%R
    | 4 => 1771%R
    | 5 => 10794%R
    | _ => 0%R
    end.

(* The right-hand side of the generating equation when the candidates are substituted *)
Definition RHS_candidate : fps :=
  fps_add fps_one 
    (fps_add (fps_mul (fps_monomial 1) G_candidate)
      (fps_add (fps_mul (fps_monomial 2)
         (fps_mul (fps_mul G_candidate G_candidate)
           (fps_geo (fps_mul (fps_monomial 1) G_candidate))))
         (fps_mul (fps_monomial 3) Q_candidate))).

Lemma G_candidate_eq_RHS_candidate_0 : G_candidate (0%nat) = RHS_candidate (0%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  simpl.
  ring.
Qed.

Fixpoint poly_eval (p : list R) (x : R) : R :=
  match p with
  | [] => 0
  | a :: p' => a + x * poly_eval p' x
  end.

Lemma G_candidate_eq_RHS_candidate_1 : G_candidate (1%nat) = RHS_candidate (1%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  simpl.
  ring.
Qed.

Fixpoint poly_add (p q : list R) : list R :=
  match p, q with
  | [], _ => q
  | _, [] => p
  | a :: p', b :: q' => (a + b) :: poly_add p' q'
  end.

(* Scalar multiplication: multiplies every coefficient of a polynomial by a constant c *)
Definition poly_scale (c : R) (p : list R) : list R :=
  map (fun a => c * a) p.

(* Polynomial multiplication: the product of two polynomials *)
Fixpoint poly_mul (p q : list R) : list R :=
  match p with
  | [] => []
  | a :: p' => poly_add (poly_scale a q) (0 :: poly_mul p' q)
  end.

Require Import Coq.micromega.Lra.


Lemma G_candidate_eq_RHS_candidate_2 : G_candidate (2%nat) = RHS_candidate (2%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  simpl.
  lra.
Qed.

Definition G_candidate_3 : R := G_candidate 3%nat.
Definition RHS_candidate_3 : R := RHS_candidate 3%nat.

Lemma G_candidate_eq_RHS_candidate_3 : G_candidate (3%nat) = RHS_candidate (3%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  cbn.
  ring.
Qed.

Lemma G_candidate_eq_RHS_candidate_4 : G_candidate (4%nat) = RHS_candidate (4%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  cbn.
  ring.
Qed.

Lemma G_candidate_eq_RHS_candidate_5 : G_candidate (5%nat) = RHS_candidate (5%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  cbn.
  ring.
Qed.

Lemma G_candidate_eq_RHS_candidate_6 : G_candidate (6%nat) = RHS_candidate (6%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  cbn.
  ring.
Qed.

Lemma G_candidate_eq_RHS_candidate_7 : G_candidate (7%nat) = RHS_candidate (7%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  cbn.
  ring.
Qed.

Lemma G_candidate_eq_RHS_candidate_8 : G_candidate (8%nat) = RHS_candidate (8%nat).
Proof.
  unfold G_candidate, RHS_candidate, fps_one, fps_add, fps_mul, fps_monomial, fps_geo.
  cbn.
  ring.
Qed.


Lemma G_candidate_zero_for_large_n : forall n, (n >= 9)%nat -> G_candidate n = 0.
Proof.
  intros n H.
  unfold G_candidate.
  destruct n as [| n0].
  - (* n = 0; contradicts n >= 9 *)
    exfalso. inversion H.
  - destruct n0 as [| n1].
    + exfalso. inversion H.
    + destruct n1 as [| n2].
      * exfalso. inversion H.
      * destruct n2 as [| n3].
        { exfalso. inversion H. }
        destruct n3 as [| n4].
        { exfalso. inversion H. }
        destruct n4 as [| n5].
        { exfalso. inversion H. }
        destruct n5 as [| n6].
        { exfalso. inversion H. }
        destruct n6 as [| n7].
        { exfalso. inversion H. }
        destruct n7 as [| n8].
        { exfalso. inversion H. }
        (* At this point, n is of the form S (S (S (S (S (S (S (S n8))))))) *)
        reflexivity.
Qed.

Lemma Q_candidate_recurrence_3 : 
  403 * Q_candidate (3%nat) - 5531 * Q_candidate (2%nat) + 23277 * Q_candidate (1%nat) - 29357 * Q_candidate (0%nat) = 0.
Proof.
  unfold Q_candidate.
  simpl.
  ring.
Qed.

Lemma Q_candidate_recurrence_4 : 
  403 * Q_candidate (4%nat) - 5531 * Q_candidate (3%nat) + 23277 * Q_candidate (2%nat) - 29357 * Q_candidate (1%nat) = 0.
Proof.
  unfold Q_candidate.
  simpl.
  ring.
Qed.

Lemma Q_candidate_recurrence_5 : 
  403 * Q_candidate (5%nat) - 5531 * Q_candidate (4%nat) + 23277 * Q_candidate (3%nat) - 29357 * Q_candidate (2%nat) = 0.
Proof.
  unfold Q_candidate.
  simpl.
  ring.
Qed.

Require Import Coq.micromega.Lia.

Lemma fps_one_zero : forall (n : nat), n <> 0%nat -> fps_one n = 0%R.
Proof.
  intros n H.
  unfold fps_one.
  destruct n as [| n'].
  - exfalso. apply H. reflexivity.
  - reflexivity.
Qed.

Lemma fps_monomial_zero : forall (k n : nat), n <> k -> fps_monomial k n = 0%R.
Proof.
  intros k n H.
  unfold fps_monomial.
  destruct (n =? k) eqn:E.
  - (* Case: n =? k = true *)
    apply Nat.eqb_eq in E. contradiction.
  - (* Case: n =? k = false *)
    reflexivity.
Qed.
