(define (problem medium_problem_4)
  (:domain blocksworld)
  
  (:objects 
    Y P B G O - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B Y)
    (on O B)

    (clear P)
    (clear G)
    (clear O)

    (inColumn Y C3)
    (inColumn P C4)
    (inColumn B C3)
    (inColumn G C1)
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
      (on O Y)
      (on B P)

      (clear B)
      (clear G)
      (clear O)

      (inColumn Y C5)
      (inColumn P C2)
      (inColumn B C2)
      (inColumn G C1)
      (inColumn O C5)
    )
  )
)