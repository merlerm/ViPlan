(define (problem hard_problem_24)
  (:domain blocksworld)
  
  (:objects 
    Y G P O R B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O Y)
    (on P G)
    (on R P)
    (on B O)

    (clear R)
    (clear B)

    (inColumn Y C3)
    (inColumn G C1)
    (inColumn P C1)
    (inColumn O C3)
    (inColumn R C1)
    (inColumn B C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on P Y)
      (on R O)
      (on B R)

      (clear G)
      (clear P)
      (clear B)

      (inColumn Y C4)
      (inColumn G C3)
      (inColumn P C4)
      (inColumn O C2)
      (inColumn R C2)
      (inColumn B C2)
    )
  )
)