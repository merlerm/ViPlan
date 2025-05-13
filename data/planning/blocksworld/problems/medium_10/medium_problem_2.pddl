(define (problem medium_problem_2)
  (:domain blocksworld)
  
  (:objects 
    G P O R B - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on R G)
    (on O P)
    (on B O)

    (clear R)
    (clear B)

    (inColumn G C3)
    (inColumn P C5)
    (inColumn O C5)
    (inColumn R C3)
    (inColumn B C5)

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
      (on R G)
      (on B O)

      (clear P)
      (clear R)
      (clear B)

      (inColumn G C3)
      (inColumn P C1)
      (inColumn O C2)
      (inColumn R C3)
      (inColumn B C2)
    )
  )
)