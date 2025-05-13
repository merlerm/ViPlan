(define (problem simple_problem_4)
  (:domain blocksworld)
  
  (:objects 
    G B O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O B)

    (clear G)
    (clear O)

    (inColumn G C3)
    (inColumn B C1)
    (inColumn O C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B G)
      (on O B)

      (clear O)

      (inColumn G C2)
      (inColumn B C2)
      (inColumn O C2)
    )
  )
)