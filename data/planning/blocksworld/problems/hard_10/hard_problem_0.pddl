(define (problem hard_problem_0)
  (:domain blocksworld)
  
  (:objects 
    Y G B O P R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on G Y)
    (on R B)
    (on P O)

    (clear G)
    (clear P)
    (clear R)

    (inColumn Y C1)
    (inColumn G C1)
    (inColumn B C2)
    (inColumn O C3)
    (inColumn P C3)
    (inColumn R C2)

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
      (on O G)

      (clear B)
      (clear O)
      (clear P)
      (clear R)

      (inColumn Y C3)
      (inColumn G C1)
      (inColumn B C2)
      (inColumn O C1)
      (inColumn P C4)
      (inColumn R C3)
    )
  )
)