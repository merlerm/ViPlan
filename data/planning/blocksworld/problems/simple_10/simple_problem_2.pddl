(define (problem simple_problem_2)
  (:domain blocksworld)
  
  (:objects 
    R G O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O G)

    (clear R)
    (clear O)

    (inColumn R C4)
    (inColumn G C2)
    (inColumn O C2)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and

      (clear R)
      (clear G)
      (clear O)

      (inColumn R C2)
      (inColumn G C1)
      (inColumn O C4)
    )
  )
)