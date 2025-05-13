(define (problem medium_problem_8)
  (:domain blocksworld)
  
  (:objects 
    O R G B P - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on P R)

    (clear O)
    (clear G)
    (clear B)
    (clear P)

    (inColumn O C3)
    (inColumn R C5)
    (inColumn G C4)
    (inColumn B C1)
    (inColumn P C5)

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
      (on P R)

      (clear O)
      (clear G)
      (clear B)
      (clear P)

      (inColumn O C5)
      (inColumn R C3)
      (inColumn G C4)
      (inColumn B C2)
      (inColumn P C3)
    )
  )
)