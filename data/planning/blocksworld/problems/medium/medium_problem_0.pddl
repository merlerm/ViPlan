(define (problem medium_problem_0)
  (:domain blocksworld)
  
  (:objects 
    G Y P B O - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on P Y)
    (on O B)

    (clear G)
    (clear P)
    (clear O)

    (inColumn G C1)
    (inColumn Y C5)
    (inColumn P C5)
    (inColumn B C3)
    (inColumn O C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)
    (rightOf C5 C4)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
    (leftOf C4 C5)
  )
  (:goal
    (and
      (on B G)
      (on P Y)

      (clear P)
      (clear B)
      (clear O)

      (inColumn G C1)
      (inColumn Y C4)
      (inColumn P C4)
      (inColumn B C1)
      (inColumn O C3)
    )
  )
)