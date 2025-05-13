(define (problem hard_problem_6)
  (:domain blocksworld)
  
  (:objects 
    Y R B O P G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O Y)
    (on G P)

    (clear R)
    (clear B)
    (clear O)
    (clear G)

    (inColumn Y C3)
    (inColumn R C1)
    (inColumn B C2)
    (inColumn O C3)
    (inColumn P C4)
    (inColumn G C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on R Y)
      (on B R)
      (on G O)

      (clear B)
      (clear P)
      (clear G)

      (inColumn Y C3)
      (inColumn R C3)
      (inColumn B C3)
      (inColumn O C4)
      (inColumn P C1)
      (inColumn G C4)
    )
  )
)