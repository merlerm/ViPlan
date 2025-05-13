(define (problem medium_problem_21)
  (:domain blocksworld)
  
  (:objects 
    R B G P O - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G B)
    (on P G)

    (clear R)
    (clear P)
    (clear O)

    (inColumn R C4)
    (inColumn B C5)
    (inColumn G C5)
    (inColumn P C5)
    (inColumn O C1)

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
      (on P B)
      (on O P)

      (clear R)
      (clear G)
      (clear O)

      (inColumn R C1)
      (inColumn B C5)
      (inColumn G C2)
      (inColumn P C5)
      (inColumn O C5)
    )
  )
)