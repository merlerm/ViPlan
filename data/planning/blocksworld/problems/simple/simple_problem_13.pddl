(define (problem simple_problem_13)
  (:domain blocksworld)
  
  (:objects 
    O G B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on B G)

    (clear O)
    (clear B)

    (inColumn O C3)
    (inColumn G C2)
    (inColumn B C2)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B O)

      (clear G)
      (clear B)

      (inColumn O C2)
      (inColumn G C3)
      (inColumn B C2)
    )
  )
)