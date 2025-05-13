(define (domain blocksworld)
  (:requirements :strips :typing :negative-preconditions :conditional-effects :equality)
  (:types
    block column
  )

  (:predicates
    (on ?b1 - block ?b2 - block) ;; block b1 is on block b2
    (inColumn ?b - block ?c - column) ;; block b is in column c
    (clear ?b - block) ;; block b is clear (i.e., nothing is on top of it)
    (rightOf ?c1 - column ?c2 - column) ;; column c1 is to the right of column c2
    (leftOf ?c1 - column ?c2 - column) ;; column c1 is to the left of column c2
  )

  (:action moveBlock
    :parameters (?b1 - block ?c1 - column) ;; move block b1 to column c1
    :precondition (and (clear ?b1) (not (inColumn ?b1 ?c1))) ;; block b1 must be clear and not already in column c1
    :effect (and
      (forall
        (?b2 - block) ;; for all blocks b2
        (and
          (when
            (on ?b1 ?b2)
            (and (not (on ?b1 ?b2)) (clear ?b2)) ;; if block b1 was on block b2, then b1 is no longer on b2 and b2 is clear
          )
          (when
            (and (inColumn ?b2 ?c1) (clear ?b2) (not (= ?b2 ?b1)))
            (and (on ?b1 ?b2) (not (clear ?b2))) ;; if another block b2 was in the column c1 where b1 is moving and b2 was clear, then b1 is now on b2 and b2 is no longer clear
          )
        )
      )

      (forall
        (?c2 - column) ;; for all columns c2
        (when
          (inColumn ?b1 ?c2)
          (not (inColumn ?b1 ?c2))) ;; if block b1 was in column c2, then b1 is no longer in c2
      ) 
      
      (inColumn ?b1 ?c1) ;; block b1 is now in column c1
      (clear ?b1) ;; block b1 is now clear (as it must be if it was moved)
    )
  )
)