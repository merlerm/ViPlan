(define (problem medium_problem_22)
  (:domain blocksworld)
  
  (:objects 
    P G Y B O - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B P)

    (clear G)
    (clear Y)
    (clear B)
    (clear O)

    (inColumn P C5)
    (inColumn G C2)
    (inColumn Y C1)
    (inColumn B C5)
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
      (on B P)
      (on O Y)

      (clear G)
      (clear B)
      (clear O)

      (inColumn P C1)
      (inColumn G C2)
      (inColumn Y C4)
      (inColumn B C1)
      (inColumn O C4)
    )
  )
)