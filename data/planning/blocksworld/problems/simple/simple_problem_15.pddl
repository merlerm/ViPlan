(define (problem simple_problem_15)
  (:domain blocksworld)
  
  (:objects 
    P O R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O P)

    (clear O)
    (clear R)

    (inColumn P C3)
    (inColumn O C3)
    (inColumn R C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on R O)

      (clear P)
      (clear R)

      (inColumn P C2)
      (inColumn O C3)
      (inColumn R C3)
    )
  )
)