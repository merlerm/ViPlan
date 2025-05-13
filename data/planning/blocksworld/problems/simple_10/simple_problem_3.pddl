(define (problem simple_problem_3)
  (:domain blocksworld)
  
  (:objects 
    P O R - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear P)
    (clear O)
    (clear R)

    (inColumn P C1)
    (inColumn O C3)
    (inColumn R C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on O P)

      (clear O)
      (clear R)

      (inColumn P C4)
      (inColumn O C4)
      (inColumn R C1)
    )
  )
)