(define (problem simple_problem_2)
  (:domain blocksworld)
  
  (:objects 
    G R P - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear G)
    (clear R)
    (clear P)

    (inColumn G C4)
    (inColumn R C2)
    (inColumn P C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on P G)

      (clear R)
      (clear P)

      (inColumn G C1)
      (inColumn R C3)
      (inColumn P C1)
    )
  )
)