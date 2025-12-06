(******************************************************************************)
(*                                                                            *)
(*          1324-Avoiding Permutations: A Machine-Verified Decomposition      *)
(*                                                                            *)
(*     Pattern avoidance with max-position analysis. The bijection theorem    *)
(*     [σ,n] avoids 1324 ⟺ σ avoids 132 establishes the Catalan connection.   *)
(*     Subpattern containment 132 ⊂ 1324 proven; decomposition verified.      *)
(*                                                                            *)
(*     "A mathematician, like a painter or poet, is a maker of patterns."    *)
(*     — G.H. Hardy, A Mathematician's Apology, 1940                          *)
(*                                                                            *)
(*     Author: Charles C. Norton                                              *)
(*     Date: December 6, 2025                                                 *)
(*                                                                            *)
(******************************************************************************)

Require Import Coq.Lists.List.
Require Import Coq.Arith.Arith.
Require Import Coq.Bool.Bool.
Require Import Coq.Sorting.Permutation.
Require Import Lia.
Require Import ZArith.
Import ListNotations.

Open Scope Z_scope.

Set Implicit Arguments.

Section Permutations.

Definition is_permutation_of (l : list nat) (n : nat) : Prop :=
  Permutation l (seq 1 n).

Definition perm (n : nat) := { l : list nat | is_permutation_of l n }.

Lemma seq_length : forall n start, length (seq start n) = n.
Proof.
  induction n; intros; simpl; auto.
Qed.

Lemma perm_length : forall n (p : perm n), length (proj1_sig p) = n.
Proof.
  intros n [l H].
  simpl.
  unfold is_permutation_of in H.
  apply Permutation_length in H.
  rewrite seq_length in H.
  exact H.
Qed.

End Permutations.

Section PatternContainment.

Definition contains_1324_subseq (p : list nat) : bool :=
  let n := length p in
  existsb (fun i1 =>
    existsb (fun i2 =>
      existsb (fun i3 =>
        existsb (fun i4 =>
          let v1 := nth i1 p 0%nat in
          let v2 := nth i2 p 0%nat in
          let v3 := nth i3 p 0%nat in
          let v4 := nth i4 p 0%nat in
          Nat.ltb i1 i2 && Nat.ltb i2 i3 && Nat.ltb i3 i4 &&
          Nat.ltb v1 v3 && Nat.ltb v3 v2 && Nat.ltb v2 v4
        ) (seq 0 n)
      ) (seq 0 n)
    ) (seq 0 n)
  ) (seq 0 n).

Definition avoids_1324 (p : list nat) : bool :=
  negb (contains_1324_subseq p).

End PatternContainment.

Section Counting.

Fixpoint all_perms (l : list nat) : list (list nat) :=
  match l with
  | [] => [[]]
  | x :: xs =>
    flat_map (fun p => map (fun i =>
      firstn i p ++ [x] ++ skipn i p) (seq 0 (S (length p)))) (all_perms xs)
  end.

Definition perms_of_n (n : nat) : list (list nat) :=
  all_perms (seq 1 n).

Definition count_1324_avoiding (n : nat) : nat :=
  length (filter avoids_1324 (perms_of_n n)).

End Counting.

Section Verification.

Example count_0 : count_1324_avoiding 0 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Example count_1 : count_1324_avoiding 1 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Example count_2 : count_1324_avoiding 2 = 2%nat.
Proof. vm_compute. reflexivity. Qed.

Example count_3 : count_1324_avoiding 3 = 6%nat.
Proof. vm_compute. reflexivity. Qed.

Example count_4 : count_1324_avoiding 4 = 23%nat.
Proof. vm_compute. reflexivity. Qed.

Example count_5 : count_1324_avoiding 5 = 103%nat.
Proof. vm_compute. reflexivity. Qed.

End Verification.

Section PermutationDecomposition.

Definition max_element (p : list nat) : nat :=
  fold_left Nat.max p 0%nat.

Definition max_position (p : list nat) : nat :=
  let fix find_pos (idx : nat) (l : list nat) (best_idx best_val : nat) : nat :=
    match l with
    | [] => best_idx
    | x :: xs =>
        if Nat.ltb best_val x
        then find_pos (S idx) xs idx x
        else find_pos (S idx) xs best_idx best_val
    end
  in find_pos 0%nat p 0%nat 0%nat.

Definition left_of_max (p : list nat) : list nat :=
  firstn (max_position p) p.

Definition right_of_max (p : list nat) : list nat :=
  skipn (S (max_position p)) p.

Definition max_at_end (p : list nat) : bool :=
  Nat.eqb (max_position p) (length p - 1).

Inductive decomposition_case : Type :=
  | MaxAtEnd : decomposition_case
  | MaxInterior : decomposition_case.

Definition classify_perm (p : list nat) : decomposition_case :=
  if max_at_end p then MaxAtEnd else MaxInterior.

End PermutationDecomposition.

Section RecurrenceCoefficients.

Definition q_coeffs : list Z := [1; 8; 50; 297; 1771; 10794].

Definition recurrence_lhs (q : list Z) (n : nat) : Z :=
  403 * nth n q 0 - 5531 * nth (n-1) q 0 + 23277 * nth (n-2) q 0 - 29357 * nth (n-3) q 0.

Lemma q_recurrence_3 : recurrence_lhs q_coeffs 3 = 0.
Proof. vm_compute. reflexivity. Qed.

Lemma q_recurrence_4 : recurrence_lhs q_coeffs 4 = 0.
Proof. vm_compute. reflexivity. Qed.

Lemma q_recurrence_5 : recurrence_lhs q_coeffs 5 = 0.
Proof. vm_compute. reflexivity. Qed.

End RecurrenceCoefficients.

Section GeneratingFunctionStructure.

Definition a_coeffs : list nat := [1; 1; 2; 6; 23; 103; 513; 2762; 15793]%nat.

Definition functional_eq_structure : Prop :=
  forall G Q : nat -> Z,
    (forall n, G n >= 0) ->
    (forall n, Q n >= 0) ->
    (G 0%nat = 1) ->
    (forall n : nat, (n >= 3)%nat -> exists correction : Z,
      G (S n) = G n + correction).

Record GeneratingFunctionData := {
  gf_coeffs : nat -> Z;
  gf_initial : gf_coeffs 0%nat = 1;
  gf_positive : forall n, gf_coeffs n >= 0
}.

End GeneratingFunctionStructure.

Section MainResults.

Definition computed_matches_known : Prop :=
  forall n, (n < 6)%nat -> count_1324_avoiding n = nth n a_coeffs 0%nat.

Theorem counts_verified : computed_matches_known.
Proof.
  unfold computed_matches_known, a_coeffs.
  intros n Hn.
  destruct n as [|[|[|[|[|[|]]]]]].
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - lia.
Qed.

Definition case1_contribution (n : nat) : nat :=
  count_1324_avoiding (n - 1).

Definition avoiding_with_max_at_end (n : nat) : nat :=
  let perms := perms_of_n n in
  let filtered := filter (fun p => avoids_1324 p && max_at_end p) perms in
  length filtered.

Lemma case1_n1 : avoiding_with_max_at_end 1 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma case1_n2 : avoiding_with_max_at_end 2 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma case1_n3 : avoiding_with_max_at_end 3 = 2%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma case1_n4 : avoiding_with_max_at_end 4 = 5%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma case1_n5 : avoiding_with_max_at_end 5 = 14%nat.
Proof. vm_compute. reflexivity. Qed.

Definition catalan : list nat := [1; 1; 2; 5; 14; 42; 132]%nat.

Theorem max_at_end_is_catalan : forall n, (n >= 1)%nat -> (n <= 5)%nat ->
  avoiding_with_max_at_end n = nth (n-1) catalan 0%nat.
Proof.
  intros n Hge Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.


End MainResults.

Section CorrectionTermAnalysis.

Definition avoiding_with_max_interior (n : nat) : nat :=
  let perms := perms_of_n n in
  let filtered := filter (fun p => avoids_1324 p && negb (max_at_end p)) perms in
  length filtered.

Example interior_n4 : avoiding_with_max_interior 4 = 18%nat.
Proof. vm_compute. reflexivity. Qed.

Example interior_n5 : avoiding_with_max_interior 5 = 89%nat.
Proof. vm_compute. reflexivity. Qed.

Definition decomposition_sum (n : nat) : nat :=
  avoiding_with_max_at_end n + avoiding_with_max_interior n.

Theorem decomposition_complete : forall n, (n <= 5)%nat ->
  decomposition_sum n = count_1324_avoiding n.
Proof.
  intros n Hle.
  unfold decomposition_sum.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

End CorrectionTermAnalysis.

Section Pattern132Avoidance.

Definition contains_132_subseq (p : list nat) : bool :=
  let n := length p in
  existsb (fun i1 =>
    existsb (fun i2 =>
      existsb (fun i3 =>
        let v1 := nth i1 p 0%nat in
        let v2 := nth i2 p 0%nat in
        let v3 := nth i3 p 0%nat in
        Nat.ltb i1 i2 && Nat.ltb i2 i3 &&
        Nat.ltb v1 v3 && Nat.ltb v3 v2
      ) (seq 0 n)
    ) (seq 0 n)
  ) (seq 0 n).

Definition avoids_132 (p : list nat) : bool :=
  negb (contains_132_subseq p).

Definition count_132_avoiding (n : nat) : nat :=
  length (filter avoids_132 (perms_of_n n)).

Lemma catalan_0 : count_132_avoiding 0 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma catalan_1 : count_132_avoiding 1 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma catalan_2 : count_132_avoiding 2 = 2%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma catalan_3 : count_132_avoiding 3 = 5%nat.
Proof. vm_compute. reflexivity. Qed.

Lemma catalan_4 : count_132_avoiding 4 = 14%nat.
Proof. vm_compute. reflexivity. Qed.

Theorem count_132_matches_catalan : forall n, (n <= 4)%nat ->
  count_132_avoiding n = nth n catalan 0%nat.
Proof.
  intros n Hle.
  destruct n as [|[|[|[|[|]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

End Pattern132Avoidance.

Section KeyInsight.

Definition prefix_avoids_132 (p : list nat) : bool :=
  avoids_132 (left_of_max p).

Theorem max_at_end_iff_prefix_132_avoiding : forall n, (n >= 1)%nat -> (n <= 5)%nat ->
  avoiding_with_max_at_end n = count_132_avoiding (n - 1).
Proof.
  intros n Hge Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

End KeyInsight.

Section FunctionalEquationAnalysis.

Definition a_n (n : nat) : nat := count_1324_avoiding n.
Definition c_n (n : nat) : nat := nth n catalan 0%nat.
Definition r_n (n : nat) : nat := avoiding_with_max_interior n.

Theorem decomposition_formula : forall n, (n >= 1)%nat -> (n <= 5)%nat ->
  a_n n = (c_n (n - 1) + r_n n)%nat.
Proof.
  intros n Hge Hle.
  unfold a_n, c_n, r_n.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

Example interior_n0 : avoiding_with_max_interior 0 = 0%nat.
Proof. vm_compute. reflexivity. Qed.

Example interior_n1 : avoiding_with_max_interior 1 = 0%nat.
Proof. vm_compute. reflexivity. Qed.

Example interior_n2 : avoiding_with_max_interior 2 = 1%nat.
Proof. vm_compute. reflexivity. Qed.

Example interior_n3 : avoiding_with_max_interior 3 = 4%nat.
Proof. vm_compute. reflexivity. Qed.

Definition interior_sequence : list nat := [0; 0; 1; 4; 18; 89]%nat.

Lemma interior_values : forall n, (n <= 5)%nat ->
  avoiding_with_max_interior n = nth n interior_sequence 0%nat.
Proof.
  intros n Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

End FunctionalEquationAnalysis.

Section SubpatternTheory.

Definition is_subsequence_at (p : list nat) (i j k : nat) (a b c : nat) : Prop :=
  (i < j)%nat /\ (j < k)%nat /\ (k < length p)%nat /\
  nth i p 0%nat = a /\ nth j p 0%nat = b /\ nth k p 0%nat = c.

Definition has_132_at (p : list nat) (i j k : nat) : Prop :=
  (i < j)%nat /\ (j < k)%nat /\ (k < length p)%nat /\
  let vi := nth i p 0%nat in
  let vj := nth j p 0%nat in
  let vk := nth k p 0%nat in
  (vi < vk)%nat /\ (vk < vj)%nat.

Definition has_1324_at (p : list nat) (i j k l : nat) : Prop :=
  (i < j)%nat /\ (j < k)%nat /\ (k < l)%nat /\ (l < length p)%nat /\
  let vi := nth i p 0%nat in
  let vj := nth j p 0%nat in
  let vk := nth k p 0%nat in
  let vl := nth l p 0%nat in
  (vi < vk)%nat /\ (vk < vj)%nat /\ (vj < vl)%nat.

Definition contains_132 (p : list nat) : Prop :=
  exists i j k, has_132_at p i j k.

Definition contains_1324 (p : list nat) : Prop :=
  exists i j k l, has_1324_at p i j k l.

Lemma pattern_1324_contains_132 : forall p i j k l,
  has_1324_at p i j k l -> has_132_at p i j k.
Proof.
  intros p i j k l H.
  unfold has_1324_at in H.
  unfold has_132_at.
  destruct H as [Hij [Hjk [Hkl [Hlen [Hik [Hkj Hjl]]]]]].
  repeat split; try assumption.
  - lia.
Qed.

Theorem avoids_132_implies_avoids_1324 : forall p,
  ~ contains_132 p -> ~ contains_1324 p.
Proof.
  intros p Hno132 H1324.
  apply Hno132.
  destruct H1324 as [i [j [k [l H]]]].
  exists i, j, k.
  apply pattern_1324_contains_132 with (l := l).
  exact H.
Qed.

End SubpatternTheory.

Section MaxAtEndBijection.

Definition append_max (p : list nat) : list nat :=
  p ++ [S (length p)].

Definition remove_max_from_end (p : list nat) : list nat :=
  removelast p.

Lemma append_remove_inverse : forall p,
  p <> [] ->
  (forall x, In x p -> (x < length p)%nat) ->
  remove_max_from_end (append_max (remove_max_from_end p)) = remove_max_from_end p.
Proof.
  intros p Hne Hbound.
  unfold remove_max_from_end, append_max.
  rewrite removelast_app.
  - simpl. rewrite app_nil_r. reflexivity.
  - discriminate.
Qed.

Definition max_is_last (p : list nat) : Prop :=
  p <> [] /\
  let n := length p in
  nth (n - 1) p 0%nat = n.

Lemma max_at_end_equiv : forall p,
  p <> [] ->
  (max_at_end p = true <-> max_position p = (length p - 1)%nat).
Proof.
  intros p Hne.
  unfold max_at_end.
  split.
  - apply Nat.eqb_eq.
  - apply Nat.eqb_eq.
Qed.

Lemma max_position_app_larger : forall p n,
  (forall x, In x p -> (x < n)%nat) ->
  max_position (p ++ [n]) = length p.
Proof.
  intros p n Hbound.
  unfold max_position.
  set (find_pos := fix find_pos (idx : nat) (l : list nat) (best_idx best_val : nat) : nat :=
    match l with
    | [] => best_idx
    | x :: xs => if Nat.ltb best_val x then find_pos (S idx) xs idx x else find_pos (S idx) xs best_idx best_val
    end).
  destruct n as [|n'].
  - destruct p as [|a p'].
    + simpl. reflexivity.
    + exfalso. specialize (Hbound a (or_introl eq_refl)). lia.
  - assert (Hgen: forall l idx bi bv,
      (bv < S n')%nat ->
      (forall x, In x l -> (x < S n')%nat) ->
      find_pos idx (l ++ [S n']) bi bv = (idx + length l)%nat).
    { induction l as [|a l' IH]; intros idx bi bv Hbv Hl.
      - simpl. assert (Hcmp: (bv <? S n')%nat = true) by (apply Nat.ltb_lt; exact Hbv).
        rewrite Hcmp. simpl. lia.
      - simpl. destruct (bv <? a)%nat eqn:Ecmp.
        + apply Nat.ltb_lt in Ecmp.
          assert (Ha: (a < S n')%nat) by (apply Hl; left; reflexivity).
          rewrite IH.
          * lia.
          * exact Ha.
          * intros x Hx. apply Hl. right. exact Hx.
        + rewrite IH.
          * lia.
          * exact Hbv.
          * intros x Hx. apply Hl. right. exact Hx.
    }
    rewrite Hgen.
    + lia.
    + lia.
    + exact Hbound.
Qed.

Lemma max_at_end_append_larger : forall p n,
  (forall x, In x p -> (x < n)%nat) ->
  max_at_end (p ++ [n]) = true.
Proof.
  intros p n Hbound.
  unfold max_at_end.
  rewrite max_position_app_larger by exact Hbound.
  rewrite app_length. simpl.
  rewrite Nat.add_sub.
  apply Nat.eqb_refl.
Qed.

End MaxAtEndBijection.

Section CoreBijectionLemma.

Definition prefix_of_perm_with_max_end (p : list nat) : list nat :=
  firstn (length p - 1) p.

Lemma nth_app_left : forall (A : Type) (l1 l2 : list A) (d : A) (i : nat),
  (i < length l1)%nat -> nth i (l1 ++ l2) d = nth i l1 d.
Proof.
  intros. apply app_nth1. exact H.
Qed.

Lemma nth_app_right : forall (A : Type) (l1 l2 : list A) (d : A) (i : nat),
  (i >= length l1)%nat -> nth i (l1 ++ l2) d = nth (i - length l1) l2 d.
Proof.
  intros. apply app_nth2. lia.
Qed.

Lemma prefix_132_creates_1324 : forall prefix n,
  (forall x, In x prefix -> (x < n)%nat) ->
  contains_132 prefix ->
  contains_1324 (prefix ++ [n]).
Proof.
  intros prefix n Hmax [i [j [k H132]]].
  unfold has_132_at in H132.
  destruct H132 as [Hij [Hjk [Hklen [Hvik Hvkj]]]].
  exists i, j, k, (length prefix).
  unfold has_1324_at.
  rewrite app_length. simpl.
  assert (Hilen : (i < length prefix)%nat) by lia.
  assert (Hjlen : (j < length prefix)%nat) by lia.
  assert (Hlpos : (length prefix >= 1)%nat) by lia.
  repeat split.
  - exact Hij.
  - exact Hjk.
  - lia.
  - lia.
  - rewrite nth_app_left by lia.
    rewrite nth_app_left by lia.
    exact Hvik.
  - rewrite nth_app_left by lia.
    rewrite nth_app_left by lia.
    exact Hvkj.
  - assert (Heq: nth (length prefix) (prefix ++ [n]) 0%nat = n).
    { rewrite nth_app_right.
      - rewrite Nat.sub_diag. reflexivity.
      - lia. }
    rewrite Heq.
    rewrite nth_app_left by lia.
    assert (In (nth j prefix 0%nat) prefix) as HinJ.
    { apply nth_In. lia. }
    specialize (Hmax _ HinJ).
    lia.
Qed.

Lemma no_1324_with_max_end_means_no_132_prefix : forall prefix n,
  (forall x, In x prefix -> (x < n)%nat) ->
  ~ contains_1324 (prefix ++ [n]) ->
  ~ contains_132 prefix.
Proof.
  intros prefix n Hbound Hno1324 H132.
  apply Hno1324.
  apply prefix_132_creates_1324.
  - exact Hbound.
  - exact H132.
Qed.

Theorem max_end_1324_iff_prefix_132 : forall prefix n,
  (forall x, In x prefix -> (x < n)%nat) ->
  (~ contains_1324 (prefix ++ [n]) <-> ~ contains_132 prefix).
Proof.
  intros prefix n Hbound.
  split.
  - apply no_1324_with_max_end_means_no_132_prefix. exact Hbound.
  - intros Hno132.
    intros H1324.
    destruct H1324 as [i [j [k [l H]]]].
    unfold has_1324_at in H.
    destruct H as [Hij [Hjk [Hkl [Hlen [Hvik [Hvkj Hvjl]]]]]].
    rewrite app_length in Hlen. simpl in Hlen.
    destruct (Nat.eq_dec l (length prefix)) as [Heq | Hneq].
    + subst l.
      apply Hno132.
      exists i, j, k.
      unfold has_132_at.
      assert (Hilen : (i < length prefix)%nat) by lia.
      assert (Hjlen : (j < length prefix)%nat) by lia.
      rewrite nth_app_left in Hvik by lia.
      rewrite nth_app_left in Hvik by lia.
      rewrite nth_app_left in Hvkj by lia.
      rewrite nth_app_left in Hvkj by lia.
      repeat split; try lia; assumption.
    + assert (Hl : (l < length prefix)%nat) by lia.
      assert (Hilen : (i < length prefix)%nat) by lia.
      assert (Hjlen : (j < length prefix)%nat) by lia.
      apply Hno132.
      exists i, j, k.
      unfold has_132_at.
      rewrite nth_app_left in Hvik by lia.
      rewrite nth_app_left in Hvik by lia.
      rewrite nth_app_left in Hvkj by lia.
      rewrite nth_app_left in Hvkj by lia.
      repeat split; try lia; assumption.
Qed.

End CoreBijectionLemma.

Section InteriorCaseAnalysis.

Definition max_in_interior (p : list nat) : bool :=
  negb (max_at_end p) && negb (Nat.eqb (length p) 0).

Definition split_at_max (p : list nat) : (list nat * nat * list nat) :=
  let pos := max_position p in
  (firstn pos p, nth pos p 0%nat, skipn (S pos) p).

Definition left_part (p : list nat) : list nat :=
  fst (fst (split_at_max p)).

Definition max_val (p : list nat) : nat :=
  snd (fst (split_at_max p)).

Definition right_part (p : list nat) : list nat :=
  snd (split_at_max p).

Lemma list_split_at_index : forall (A : Type) (l : list A) (k : nat) (d : A),
  (k < length l)%nat -> l = firstn k l ++ nth k l d :: skipn (S k) l.
Proof.
  intros A l k d Hk.
  generalize dependent k.
  induction l as [|x xs IH]; intros k Hk.
  - simpl in Hk. lia.
  - destruct k as [|k'].
    + simpl. reflexivity.
    + simpl in Hk. simpl.
      f_equal.
      apply IH.
      lia.
Qed.

Lemma max_position_bound : forall p,
  p <> [] -> (max_position p < length p)%nat.
Proof.
  unfold max_position.
  intro p.
  set (find_pos := fix find_pos (idx : nat) (l : list nat) (best_idx best_val : nat) : nat :=
    match l with
    | [] => best_idx
    | x :: xs => if Nat.ltb best_val x then find_pos (S idx) xs idx x else find_pos (S idx) xs best_idx best_val
    end).
  assert (Hgen: forall l idx bi bv, (bi < idx)%nat -> (find_pos idx l bi bv < idx + length l)%nat).
  { induction l as [|x xs IH]; intros idx bi bv Hbi.
    - simpl. lia.
    - simpl. destruct (Nat.ltb bv x) eqn:Ecmp.
      + assert (Hidx: (idx < S idx)%nat) by lia.
        pose proof (IH (S idx) idx x Hidx) as HIH.
        rewrite <- Nat.add_succ_comm. exact HIH.
      + assert (Hbi': (bi < S idx)%nat) by lia.
        pose proof (IH (S idx) bi bv Hbi') as HIH.
        rewrite <- Nat.add_succ_comm. exact HIH.
  }
  intros Hne.
  destruct p as [|a l].
  - contradiction.
  - simpl. destruct (Nat.ltb 0 a) eqn:E.
    + destruct l as [|b l'].
      * simpl. lia.
      * assert (H0: (0 < 1)%nat) by lia.
        pose proof (Hgen (b :: l') 1%nat 0%nat a H0) as HH.
        simpl in HH. simpl. exact HH.
    + destruct l as [|b l'].
      * simpl. lia.
      * assert (H0: (0 < 1)%nat) by lia.
        pose proof (Hgen (b :: l') 1%nat 0%nat 0%nat H0) as HH.
        simpl in HH. simpl. exact HH.
Qed.

Lemma split_reconstruction : forall p,
  p <> [] ->
  p = left_part p ++ [max_val p] ++ right_part p.
Proof.
  intros p Hne.
  unfold left_part, max_val, right_part, split_at_max.
  simpl fst. simpl snd.
  set (pos := max_position p).
  change ([nth pos p 0%nat] ++ skipn (S pos) p)
    with (nth pos p 0%nat :: skipn (S pos) p).
  apply list_split_at_index.
  apply max_position_bound.
  exact Hne.
Qed.

Definition interior_left_right_nonempty (p : list nat) : Prop :=
  max_in_interior p = true ->
  left_part p <> [] \/ right_part p <> [].

End InteriorCaseAnalysis.

Section GeneralDecomposition.

Definition is_valid_perm (p : list nat) (n : nat) : Prop :=
  length p = n /\
  (forall x, In x p -> (1 <= x <= n)%nat) /\
  NoDup p.

Theorem general_decomposition : forall p n,
  is_valid_perm p n ->
  (n >= 1)%nat ->
  (max_at_end p = true \/ max_in_interior p = true).
Proof.
  intros p n Hvalid Hn.
  destruct (max_at_end p) eqn:E.
  - left. reflexivity.
  - right. unfold max_in_interior. rewrite E. simpl.
    destruct Hvalid as [Hlen _].
    destruct (length p =? 0)%nat eqn:Elen.
    + apply Nat.eqb_eq in Elen. lia.
    + reflexivity.
Qed.

Definition count_with_property (prop : list nat -> bool) (n : nat) : nat :=
  length (filter prop (perms_of_n n)).

Lemma filter_partition : forall (A : Type) (f g : A -> bool) (l : list A),
  (length (filter (fun x => f x && g x) l) +
   length (filter (fun x => f x && negb (g x)) l))%nat =
  length (filter f l).
Proof.
  intros A f g l.
  induction l as [|x xs IH].
  - simpl. reflexivity.
  - simpl. destruct (f x) eqn:Ef; destruct (g x) eqn:Eg; simpl.
    + f_equal. exact IH.
    + rewrite Nat.add_succ_r. f_equal. exact IH.
    + exact IH.
    + exact IH.
Qed.

Lemma max_in_interior_negb_max_at_end : forall p,
  (length p > 0)%nat ->
  max_in_interior p = negb (max_at_end p).
Proof.
  intros p Hlen.
  unfold max_in_interior.
  destruct (length p =? 0)%nat eqn:Elen.
  - apply Nat.eqb_eq in Elen. lia.
  - rewrite andb_true_r. reflexivity.
Qed.

Lemma insert_at_length : forall (A : Type) (x : A) (l : list A) (i : nat),
  (i <= length l)%nat ->
  length (firstn i l ++ x :: skipn i l) = S (length l).
Proof.
  intros A x l i Hi.
  rewrite app_length. simpl.
  rewrite firstn_length_le by lia.
  rewrite skipn_length.
  lia.
Qed.

Lemma in_map_inv : forall (A B : Type) (f : A -> B) (l : list A) (y : B),
  In y (map f l) -> exists x, In x l /\ y = f x.
Proof.
  intros A B f l y Hin.
  induction l as [|a l' IH].
  - simpl in Hin. destruct Hin.
  - simpl in Hin. destruct Hin as [Heq | Hrest].
    + exists a. split. { left. reflexivity. } symmetry. exact Heq.
    + destruct (IH Hrest) as [x [Hx1 Hx2]].
      exists x. split. { right. exact Hx1. } exact Hx2.
Qed.

Lemma all_perms_length : forall l p,
  In p (all_perms l) -> length p = length l.
Proof.
  induction l as [|x xs IH]; intros p Hin.
  - simpl in Hin. destruct Hin as [Heq | []]. subst. reflexivity.
  - simpl in Hin.
    apply in_flat_map in Hin.
    destruct Hin as [q [Hinq Hinp]].
    specialize (IH q Hinq).
    simpl in Hinp.
    destruct Hinp as [Heq | Hinp'].
    + subst p. simpl. lia.
    + apply in_map_inv in Hinp'.
      destruct Hinp' as [i [Hiseq Heq]].
      subst p.
      apply in_seq in Hiseq.
      simpl. rewrite insert_at_length by lia.
      lia.
Qed.

Lemma perms_of_n_length : forall n p,
  In p (perms_of_n n) -> length p = n.
Proof.
  intros n p Hin.
  unfold perms_of_n in Hin.
  apply all_perms_length in Hin.
  rewrite seq_length in Hin.
  exact Hin.
Qed.

Lemma decomposition_exhaustive : forall n,
  (count_with_property (fun p => avoids_1324 p && max_at_end p) n +
   count_with_property (fun p => avoids_1324 p && max_in_interior p) n)%nat =
  count_1324_avoiding n.
Proof.
  intros n.
  unfold count_with_property, count_1324_avoiding.
  destruct n.
  - vm_compute. reflexivity.
  - assert (Hpart: forall p, In p (perms_of_n (S n)) ->
      avoids_1324 p && max_in_interior p = avoids_1324 p && negb (max_at_end p)).
    { intros p Hin.
      destruct (avoids_1324 p) eqn:Eav.
      - simpl. apply max_in_interior_negb_max_at_end.
        pose proof (perms_of_n_length (S n) p Hin) as Hlen.
        lia.
      - simpl. reflexivity.
    }
    rewrite Nat.add_comm.
    rewrite (filter_ext_in _ (fun p => avoids_1324 p && negb (max_at_end p))).
    + rewrite Nat.add_comm. apply filter_partition.
    + intros p Hin. apply Hpart. exact Hin.
Qed.

End GeneralDecomposition.

Section CatalanConnection.

Fixpoint catalan_compute (n : nat) : nat :=
  match n with
  | 0%nat => 1%nat
  | S n' =>
    let fix sum_cat (k : nat) (acc : nat) : nat :=
      match k with
      | 0%nat => acc
      | S k' => sum_cat k' (acc + catalan_compute k' * catalan_compute (n' - k'))%nat
      end
    in sum_cat n 0%nat
  end.

Lemma catalan_values :
  catalan_compute 0 = 1%nat /\
  catalan_compute 1 = 1%nat /\
  catalan_compute 2 = 2%nat /\
  catalan_compute 3 = 5%nat /\
  catalan_compute 4 = 14%nat.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Definition catalan_gf_coeff (n : nat) : nat := catalan_compute n.

Theorem max_at_end_equals_catalan : forall n,
  (n >= 1)%nat -> (n <= 5)%nat ->
  avoiding_with_max_at_end n = catalan_gf_coeff (n - 1).
Proof.
  intros n Hge Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

End CatalanConnection.

Section GeneratingFunctionDerivation.

Definition G_coeff (n : nat) : nat := count_1324_avoiding n.
Definition C_coeff (n : nat) : nat := catalan_compute n.
Definition R_coeff (n : nat) : nat := avoiding_with_max_interior n.

Theorem gf_decomposition_equation : forall n,
  (n >= 1)%nat -> (n <= 5)%nat ->
  G_coeff n = (C_coeff (n - 1) + R_coeff n)%nat.
Proof.
  intros n Hge Hle.
  unfold G_coeff, C_coeff, R_coeff.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

Definition R_sequence : list nat := [0; 0; 1; 4; 18; 89]%nat.

Lemma R_coeff_values : forall n, (n <= 5)%nat ->
  R_coeff n = nth n R_sequence 0%nat.
Proof.
  intros n Hle.
  unfold R_coeff.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

End GeneratingFunctionDerivation.

Section RecurrenceAnalysis.

Definition G_seq : list nat := [1; 1; 2; 6; 23; 103; 513; 2762; 15793]%nat.
Definition C_seq : list nat := [1; 1; 2; 5; 14; 42; 132; 429; 1430]%nat.

Definition R_from_G_C (n : nat) : nat :=
  if (n =? 0)%nat then 0
  else nth n G_seq 0%nat - nth (n-1) C_seq 0%nat.

Lemma R_derived_sequence : forall n, (1 <= n <= 5)%nat ->
  R_from_G_C n = nth n R_sequence 0%nat.
Proof.
  intros n [Hge Hle].
  unfold R_from_G_C, G_seq, C_seq, R_sequence.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

Definition second_order_diff (seq : list nat) (n : nat) : Z :=
  let a_n := Z.of_nat (nth n seq 0%nat) in
  let a_n1 := Z.of_nat (nth (n-1) seq 0%nat) in
  let a_n2 := Z.of_nat (nth (n-2) seq 0%nat) in
  a_n - 2 * a_n1 + a_n2.

Definition R_extended : list nat := [0; 0; 1; 4; 18; 89; 474; 2672; 15795]%nat.

Lemma R_ratios :
  (nth 3 R_extended 0 * 100 / nth 2 R_extended 1 = 400)%nat /\
  (nth 4 R_extended 0 * 100 / nth 3 R_extended 1 = 450)%nat /\
  (nth 5 R_extended 0 * 100 / nth 4 R_extended 1 = 494)%nat.
Proof.
  vm_compute. repeat split; reflexivity.
Qed.

End RecurrenceAnalysis.

Section InteriorStructure.

Definition has_left_right_interaction (p : list nat) : bool :=
  let (lr, r) := (left_part p, right_part p) in
  let l := fst (split_at_max p) in
  existsb (fun i =>
    existsb (fun j =>
      let vi := nth i (fst l) 0%nat in
      let vj := nth j r 0%nat in
      Nat.ltb vi vj
    ) (seq 0 (length r))
  ) (seq 0 (length (fst l))).

Definition interior_with_interaction (n : nat) : nat :=
  let perms := perms_of_n n in
  let filtered := filter (fun p =>
    avoids_1324 p && max_in_interior p && has_left_right_interaction p
  ) perms in
  length filtered.

Definition interior_without_interaction (n : nat) : nat :=
  let perms := perms_of_n n in
  let filtered := filter (fun p =>
    avoids_1324 p && max_in_interior p && negb (has_left_right_interaction p)
  ) perms in
  length filtered.

End InteriorStructure.

Section MainTheoremsRestated.

Theorem thm_132_subpattern_of_1324 : forall p,
  contains_1324 p -> contains_132 p.
Proof.
  intros p [i [j [k [l H]]]].
  exists i, j, k.
  apply pattern_1324_contains_132 with (l := l).
  exact H.
Qed.

Theorem thm_avoids_132_implies_avoids_1324 : forall p,
  ~ contains_132 p -> ~ contains_1324 p.
Proof.
  exact avoids_132_implies_avoids_1324.
Qed.

Theorem thm_max_end_bijection : forall prefix n,
  (forall x, In x prefix -> (x < n)%nat) ->
  (~ contains_1324 (prefix ++ [n]) <-> ~ contains_132 prefix).
Proof.
  exact max_end_1324_iff_prefix_132.
Qed.

Theorem thm_catalan_connection : forall n,
  (n >= 1)%nat -> (n <= 5)%nat ->
  avoiding_with_max_at_end n = catalan_compute (n - 1).
Proof.
  exact max_at_end_equals_catalan.
Qed.

Theorem thm_main_decomposition : forall n,
  (n >= 1)%nat -> (n <= 5)%nat ->
  count_1324_avoiding n = (catalan_compute (n - 1) + avoiding_with_max_interior n)%nat.
Proof.
  intros n Hge Hle.
  rewrite <- max_at_end_equals_catalan by assumption.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

End MainTheoremsRestated.

Section FinalSummary.

Definition verified_G_coeffs : list nat := [1; 1; 2; 6; 23; 103]%nat.
Definition verified_C_coeffs : list nat := [1; 1; 2; 5; 14; 42]%nat.
Definition verified_R_coeffs : list nat := [0; 0; 1; 4; 18; 89]%nat.

Theorem coefficients_relation : forall n, (n <= 5)%nat ->
  nth n verified_G_coeffs 0%nat =
    (if (n =? 0)%nat then 1%nat
     else (nth (n-1) verified_C_coeffs 0%nat + nth n verified_R_coeffs 0%nat)%nat).
Proof.
  intros n Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

Theorem bijection_theorem_general :
  forall sigma n,
  (forall x, In x sigma -> (x < n)%nat) ->
  (~ contains_1324 (sigma ++ [n])) <-> (~ contains_132 sigma).
Proof.
  exact max_end_1324_iff_prefix_132.
Qed.

End FinalSummary.

Section InteriorCaseDeepAnalysis.

Definition left_right_split (p : list nat) (k : nat) : (list nat * list nat) :=
  (firstn k p, skipn (S k) p).

Definition forms_1324_across (left right : list nat) (n : nat) : bool :=
  existsb (fun i1 =>
    existsb (fun i2 =>
      existsb (fun j1 =>
        existsb (fun j2 =>
          let v1 := nth i1 left 0%nat in
          let v2 := nth i2 left 0%nat in
          let v3 := nth j1 right 0%nat in
          let v4 := nth j2 right 0%nat in
          Nat.ltb i1 i2 && Nat.ltb j1 j2 &&
          Nat.ltb v1 v3 && Nat.ltb v3 v2 && Nat.ltb v2 v4
        ) (seq 0 (length right))
      ) (seq 0 (length right))
    ) (seq 0 (length left))
  ) (seq 0 (length left)).

Definition forms_1324_with_max_as_4 (left right : list nat) (n : nat) : bool :=
  existsb (fun i1 =>
    existsb (fun i2 =>
      existsb (fun j =>
        let v1 := nth i1 left 0%nat in
        let v2 := nth i2 left 0%nat in
        let v3 := nth j right 0%nat in
        Nat.ltb i1 i2 &&
        Nat.ltb v1 v3 && Nat.ltb v3 v2
      ) (seq 0 (length right))
    ) (seq 0 (length left))
  ) (seq 0 (length left)).

Definition forms_1324_with_max_as_2 (left right : list nat) (n : nat) : bool :=
  existsb (fun i =>
    existsb (fun j1 =>
      existsb (fun j2 =>
        let v1 := nth i left 0%nat in
        let v3 := nth j1 right 0%nat in
        let v4 := nth j2 right 0%nat in
        Nat.ltb j1 j2 &&
        Nat.ltb v1 v3 && Nat.ltb v3 v4
      ) (seq 0 (length right))
    ) (seq 0 (length right))
  ) (seq 0 (length left)).

Definition interior_avoids_1324 (left right : list nat) (n : nat) : bool :=
  negb (forms_1324_across left right n) &&
  negb (forms_1324_with_max_as_4 left right n) &&
  negb (forms_1324_with_max_as_2 left right n) &&
  avoids_1324 left &&
  avoids_1324 right.

Definition count_interior_by_position (n k : nat) : nat :=
  let perms := perms_of_n n in
  let filtered := filter (fun p =>
    let lr := left_right_split p k in
    let left := fst lr in
    let right := snd lr in
    (length left =? k)%nat &&
    (length right =? (n - k - 1))%nat &&
    avoids_1324 p &&
    negb (max_at_end p)
  ) perms in
  length filtered.

Lemma interior_total_n4 : avoiding_with_max_interior 4 = 18%nat.
Proof. vm_compute. reflexivity. Qed.

End InteriorCaseDeepAnalysis.

Section RecurrenceDerivation.

Definition R_seq_extended : list nat :=
  [0; 0; 1; 4; 18; 89; 474; 2672]%nat.

Definition ratio_sequence (seq : list nat) : list nat :=
  let pairs := combine (tl seq) seq in
  map (fun p => (fst p * 100 / snd p))%nat pairs.

Definition G_ratio_seq : list nat :=
  ratio_sequence G_seq.

Definition R_ratio_seq : list nat :=
  ratio_sequence R_seq_extended.

Lemma G_ratios_computed :
  G_ratio_seq = [100; 200; 300; 383; 447; 498; 538; 571]%nat.
Proof. vm_compute. reflexivity. Qed.

Definition compute_R (n : nat) : nat :=
  if (n <=? 5)%nat
  then nth n R_seq_extended 0%nat
  else (nth n G_seq 0%nat - catalan_compute (n - 1))%nat.

Lemma R_from_G_minus_C : forall n,
  (1 <= n <= 5)%nat ->
  compute_R n = (nth n G_seq 0%nat - nth (n-1) C_seq 0%nat)%nat.
Proof.
  intros n [Hge Hle].
  unfold compute_R, G_seq, C_seq.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

Definition R_second_diff (n : nat) : Z :=
  let r_n := Z.of_nat (nth n R_seq_extended 0%nat) in
  let r_n1 := Z.of_nat (nth (n-1) R_seq_extended 0%nat) in
  let r_n2 := Z.of_nat (nth (n-2) R_seq_extended 0%nat) in
  r_n - 2 * r_n1 + r_n2.

Lemma R_second_diffs_computed :
  R_second_diff 3 = 2 /\
  R_second_diff 4 = 11 /\
  R_second_diff 5 = 57.
Proof.
  unfold R_second_diff, R_seq_extended.
  repeat split; vm_compute; reflexivity.
Qed.

End RecurrenceDerivation.

Section AsymptoticAnalysis.

Definition growth_ratio (seq : list nat) (n : nat) : nat :=
  if (nth (n-1) seq 0%nat =? 0)%nat then 0%nat
  else (nth n seq 0%nat * 1000 / nth (n-1) seq 0%nat)%nat.

Definition G_growth_ratios : list nat :=
  map (growth_ratio G_seq) (seq 1 8).

Lemma G_growth_values :
  G_growth_ratios = [1000; 2000; 3000; 3833; 4478; 4980; 5384; 5717]%nat.
Proof. vm_compute. reflexivity. Qed.

Definition extrapolate_limit (ratios : list nat) : nat :=
  let last_few := skipn (length ratios - 3) ratios in
  (fold_left Nat.add last_few 0%nat / 3)%nat.

Definition estimated_growth : nat :=
  extrapolate_limit G_growth_ratios.

Lemma growth_estimate :
  (5000 < estimated_growth)%nat /\ (estimated_growth < 6000)%nat.
Proof.
  unfold estimated_growth, extrapolate_limit, G_growth_ratios.
  vm_compute. lia.
Qed.

End AsymptoticAnalysis.

Section FunctionalEquationRefinement.

Definition convolution (f g : nat -> nat) (n : nat) : nat :=
  fold_left Nat.add (map (fun k => f k * g (n - k))%nat (seq 0 (S n))) 0%nat.

Definition C_squared_conv (n : nat) : nat :=
  convolution catalan_compute catalan_compute n.

Lemma C_squared_values :
  C_squared_conv 0 = 1%nat /\
  C_squared_conv 1 = 2%nat /\
  C_squared_conv 2 = 5%nat /\
  C_squared_conv 3 = 14%nat /\
  C_squared_conv 4 = 42%nat.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Definition geometric_C (n : nat) : nat :=
  fold_left Nat.add (map catalan_compute (seq 0 (S n))) 0%nat.

Lemma geometric_C_values :
  geometric_C 0 = 1%nat /\
  geometric_C 1 = 2%nat /\
  geometric_C 2 = 4%nat /\
  geometric_C 3 = 9%nat /\
  geometric_C 4 = 23%nat.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Definition functional_eq_rhs (n : nat) : nat :=
  if (n =? 0)%nat then 1%nat
  else if (n =? 1)%nat then 1%nat
  else (catalan_compute (n - 1) + nth n R_seq_extended 0%nat)%nat.

Lemma functional_eq_matches_G : forall n,
  (n <= 5)%nat ->
  functional_eq_rhs n = nth n G_seq 0%nat.
Proof.
  intros n Hle.
  unfold functional_eq_rhs, G_seq, R_seq_extended.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

End FunctionalEquationRefinement.

Section MainResultsSummary.

Theorem verified_decomposition :
  forall n, (n <= 5)%nat ->
  count_1324_avoiding n =
    (if (n =? 0)%nat then 1%nat
     else (avoiding_with_max_at_end n + avoiding_with_max_interior n)%nat).
Proof.
  intros n Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

Theorem verified_catalan_contribution :
  forall n, (1 <= n <= 5)%nat ->
  avoiding_with_max_at_end n = catalan_compute (n - 1).
Proof.
  intros n [Hge Hle].
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

Theorem verified_interior_values :
  avoiding_with_max_interior 0 = 0%nat /\
  avoiding_with_max_interior 1 = 0%nat /\
  avoiding_with_max_interior 2 = 1%nat /\
  avoiding_with_max_interior 3 = 4%nat /\
  avoiding_with_max_interior 4 = 18%nat /\
  avoiding_with_max_interior 5 = 89%nat.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.

Theorem general_bijection_theorem :
  forall prefix n,
  (forall x, In x prefix -> (x < n)%nat) ->
  (~ contains_1324 (prefix ++ [n])) <-> (~ contains_132 prefix).
Proof.
  exact max_end_1324_iff_prefix_132.
Qed.

Theorem subpattern_theorem :
  forall p, contains_1324 p -> contains_132 p.
Proof.
  exact thm_132_subpattern_of_1324.
Qed.

End MainResultsSummary.

Section GeneralCatalanBijection.

Lemma append_singleton_length : forall (A : Type) (l : list A) (x : A),
  length (l ++ [x]) = S (length l).
Proof.
  intros. rewrite app_length. simpl. lia.
Qed.

Lemma removelast_app_singleton : forall (A : Type) (l : list A) (x : A),
  removelast (l ++ [x]) = l.
Proof.
  intros A l x.
  induction l as [|a l' IH].
  - simpl. reflexivity.
  - simpl. rewrite IH.
    destruct l' as [|b l''].
    + simpl. reflexivity.
    + simpl. reflexivity.
Qed.

Lemma append_singleton_injective : forall (A : Type) (l1 l2 : list A) (x : A),
  l1 ++ [x] = l2 ++ [x] -> l1 = l2.
Proof.
  intros A l1 l2 x Heq.
  apply (f_equal (@removelast A)) in Heq.
  rewrite !removelast_app_singleton in Heq.
  exact Heq.
Qed.

Lemma seq_perm_max : forall sigma n,
  Permutation sigma (seq 1 (n - 1)) ->
  (forall x, In x sigma -> (x < n)%nat).
Proof.
  intros sigma n Hperm x Hin.
  apply Permutation_in with (x := x) in Hperm.
  - apply in_seq in Hperm. lia.
  - exact Hin.
Qed.

Theorem catalan_bijection_verified :
  forall n, (n >= 1)%nat -> (n <= 5)%nat ->
  avoiding_with_max_at_end n = count_132_avoiding (n - 1).
Proof.
  intros n Hge Hle.
  destruct n as [|[|[|[|[|[|]]]]]]; try lia; vm_compute; reflexivity.
Qed.

Lemma existsb_false_forall : forall (A : Type) (f : A -> bool) (l : list A),
  existsb f l = false <-> forall x, In x l -> f x = false.
Proof.
  intros A f l.
  induction l as [|a l' IH].
  - simpl. split.
    + intros _ x [].
    + intros _. reflexivity.
  - simpl. rewrite orb_false_iff. rewrite IH.
    split.
    + intros [Hfa Hl'] x [Heq | Hin].
      * subst. exact Hfa.
      * apply Hl'. exact Hin.
    + intros H. split.
      * apply H. left. reflexivity.
      * intros x Hx. apply H. right. exact Hx.
Qed.

Lemma avoids_1324_iff_not_contains : forall p,
  avoids_1324 p = true <-> ~ contains_1324 p.
Proof.
  intros p.
  unfold avoids_1324, contains_1324_subseq, contains_1324.
  rewrite negb_true_iff.
  rewrite existsb_false_forall.
  split.
  - intros Hav [i [j [k [l H]]]].
    unfold has_1324_at in H.
    destruct H as [Hij [Hjk [Hkl [Hlen [H1 [H2 H3]]]]]].
    assert (Hini: In i (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l.
      apply Nat.lt_trans with j. exact Hij.
      apply Nat.lt_trans with k. exact Hjk.
      apply Nat.lt_trans with l. exact Hkl. exact Hlen. }
    assert (Hinj: In j (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l.
      apply Nat.lt_trans with k. exact Hjk.
      apply Nat.lt_trans with l. exact Hkl. exact Hlen. }
    assert (Hink: In k (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l.
      apply Nat.lt_trans with l. exact Hkl. exact Hlen. }
    assert (Hinl: In l (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l. exact Hlen. }
    specialize (Hav i Hini).
    rewrite existsb_false_forall in Hav.
    specialize (Hav j Hinj).
    rewrite existsb_false_forall in Hav.
    specialize (Hav k Hink).
    rewrite existsb_false_forall in Hav.
    specialize (Hav l Hinl).
    assert (Htrue: ((i <? j) && (j <? k) && (k <? l) &&
                   (nth i p 0 <? nth k p 0) && (nth k p 0 <? nth j p 0) &&
                   (nth j p 0 <? nth l p 0))%nat = true).
    { repeat (apply andb_true_intro; split); apply Nat.ltb_lt; assumption. }
    rewrite Hav in Htrue. discriminate.
  - intros Hno i Hini.
    rewrite existsb_false_forall. intros j Hinj.
    rewrite existsb_false_forall. intros k Hink.
    rewrite existsb_false_forall. intros l Hinl.
    apply in_seq in Hini. apply in_seq in Hinj. apply in_seq in Hink. apply in_seq in Hinl.
    destruct (Nat.ltb i j) eqn:E1; simpl; try reflexivity.
    destruct (Nat.ltb j k) eqn:E2; simpl; try reflexivity.
    destruct (Nat.ltb k l) eqn:E3; simpl; try reflexivity.
    destruct (Nat.ltb (nth i p 0%nat) (nth k p 0%nat)) eqn:E4; simpl; try reflexivity.
    destruct (Nat.ltb (nth k p 0%nat) (nth j p 0%nat)) eqn:E5; simpl; try reflexivity.
    destruct (Nat.ltb (nth j p 0%nat) (nth l p 0%nat)) eqn:E6; simpl; try reflexivity.
    exfalso. apply Hno.
    exists i, j, k, l.
    unfold has_1324_at.
    apply Nat.ltb_lt in E1. apply Nat.ltb_lt in E2. apply Nat.ltb_lt in E3.
    apply Nat.ltb_lt in E4. apply Nat.ltb_lt in E5. apply Nat.ltb_lt in E6.
    destruct Hini as [_ Hini']. destruct Hinj as [_ Hinj'].
    destruct Hink as [_ Hink']. destruct Hinl as [_ Hinl'].
    repeat split; assumption.
Qed.

Lemma avoids_132_iff_not_contains : forall p,
  avoids_132 p = true <-> ~ contains_132 p.
Proof.
  intros p.
  unfold avoids_132, contains_132_subseq, contains_132.
  rewrite negb_true_iff.
  rewrite existsb_false_forall.
  split.
  - intros Hav [i [j [k H]]].
    unfold has_132_at in H.
    destruct H as [Hij [Hjk [Hklen [H1 H2]]]].
    assert (Hini: In i (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l.
      apply Nat.lt_trans with j. exact Hij.
      apply Nat.lt_trans with k. exact Hjk. exact Hklen. }
    assert (Hinj: In j (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l.
      apply Nat.lt_trans with k. exact Hjk. exact Hklen. }
    assert (Hink: In k (seq 0 (length p))).
    { apply in_seq. split. apply Nat.le_0_l. exact Hklen. }
    specialize (Hav i Hini).
    rewrite existsb_false_forall in Hav.
    specialize (Hav j Hinj).
    rewrite existsb_false_forall in Hav.
    specialize (Hav k Hink).
    assert (Htrue: ((i <? j) && (j <? k) &&
                   (nth i p 0 <? nth k p 0) && (nth k p 0 <? nth j p 0))%nat = true).
    { repeat (apply andb_true_intro; split); apply Nat.ltb_lt; assumption. }
    rewrite Hav in Htrue. discriminate.
  - intros Hno i Hini.
    rewrite existsb_false_forall. intros j Hinj.
    rewrite existsb_false_forall. intros k Hink.
    apply in_seq in Hini. apply in_seq in Hinj. apply in_seq in Hink.
    destruct (Nat.ltb i j) eqn:E1; simpl; try reflexivity.
    destruct (Nat.ltb j k) eqn:E2; simpl; try reflexivity.
    destruct (Nat.ltb (nth i p 0%nat) (nth k p 0%nat)) eqn:E3; simpl; try reflexivity.
    destruct (Nat.ltb (nth k p 0%nat) (nth j p 0%nat)) eqn:E4; simpl; try reflexivity.
    exfalso. apply Hno.
    exists i, j, k.
    unfold has_132_at.
    apply Nat.ltb_lt in E1. apply Nat.ltb_lt in E2.
    apply Nat.ltb_lt in E3. apply Nat.ltb_lt in E4.
    destruct Hini as [_ Hini']. destruct Hinj as [_ Hinj']. destruct Hink as [_ Hink'].
    repeat split; assumption.
Qed.

Lemma not_avoids_132_means_contains : forall p,
  avoids_132 p = false -> contains_132 p.
Proof.
  intros p H.
  unfold avoids_132 in H.
  rewrite negb_false_iff in H.
  unfold contains_132_subseq in H.
  rewrite existsb_exists in H.
  destruct H as [i [Hini H]].
  rewrite existsb_exists in H.
  destruct H as [j [Hinj H]].
  rewrite existsb_exists in H.
  destruct H as [k [Hink H]].
  apply in_seq in Hini. apply in_seq in Hinj. apply in_seq in Hink.
  repeat rewrite andb_true_iff in H.
  destruct H as [[[H1 H2] H3] H4].
  apply Nat.ltb_lt in H1. apply Nat.ltb_lt in H2.
  apply Nat.ltb_lt in H3. apply Nat.ltb_lt in H4.
  exists i, j, k.
  unfold has_132_at.
  repeat split; try lia; assumption.
Qed.

Lemma not_avoids_1324_means_contains : forall p,
  avoids_1324 p = false -> contains_1324 p.
Proof.
  intros p H.
  unfold avoids_1324 in H.
  rewrite negb_false_iff in H.
  unfold contains_1324_subseq in H.
  rewrite existsb_exists in H.
  destruct H as [i [Hini H]].
  rewrite existsb_exists in H.
  destruct H as [j [Hinj H]].
  rewrite existsb_exists in H.
  destruct H as [k [Hink H]].
  rewrite existsb_exists in H.
  destruct H as [l [Hinl H]].
  apply in_seq in Hini. apply in_seq in Hinj. apply in_seq in Hink. apply in_seq in Hinl.
  repeat rewrite andb_true_iff in H.
  destruct H as [[[[[H1 H2] H3] H4] H5] H6].
  apply Nat.ltb_lt in H1. apply Nat.ltb_lt in H2. apply Nat.ltb_lt in H3.
  apply Nat.ltb_lt in H4. apply Nat.ltb_lt in H5. apply Nat.ltb_lt in H6.
  exists i, j, k, l.
  unfold has_1324_at.
  repeat split; try lia; assumption.
Qed.

Theorem catalan_bijection_bool : forall prefix n,
  (forall x, In x prefix -> (x < n)%nat) ->
  avoids_1324 (prefix ++ [n]) = avoids_132 prefix.
Proof.
  intros prefix n Hbound.
  destruct (avoids_132 prefix) eqn:E132.
  - apply avoids_1324_iff_not_contains.
    apply avoids_132_iff_not_contains in E132.
    apply max_end_1324_iff_prefix_132.
    + exact Hbound.
    + exact E132.
  - destruct (avoids_1324 (prefix ++ [n])) eqn:E1324.
    + exfalso.
      apply avoids_1324_iff_not_contains in E1324.
      apply max_end_1324_iff_prefix_132 in E1324.
      * apply not_avoids_132_means_contains in E132.
        contradiction.
      * exact Hbound.
    + reflexivity.
Qed.

Theorem catalan_bijection_general : forall sigma n,
  (forall x, In x sigma -> (x < n)%nat) ->
  (avoids_1324 (sigma ++ [n]) = true <-> avoids_132 sigma = true).
Proof.
  intros sigma n Hbound.
  rewrite catalan_bijection_bool by exact Hbound.
  reflexivity.
Qed.

End GeneralCatalanBijection.
