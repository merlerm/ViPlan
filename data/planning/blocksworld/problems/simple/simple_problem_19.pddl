(define (problem simple_problem_19)
  (:domain blocksworld)
  
  (:objects 
    B G O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O G)

    (clear B)
    (clear O)

    (inColumn B C2)
    (inColumn G C1)
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
      (on G B)
      (on O G)

      (clear O)

      (inColumn B C4)
      (inColumn G C4)
      (inColumn O C4)
    )
  )
)