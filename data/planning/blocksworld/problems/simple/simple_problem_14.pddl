(define (problem simple_problem_14)
  (:domain blocksworld)
  
  (:objects 
    R P G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on G P)

    (clear R)
    (clear G)

    (inColumn R C1)
    (inColumn P C3)
    (inColumn G C3)

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
      (clear P)
      (clear G)

      (inColumn R C3)
      (inColumn P C4)
      (inColumn G C2)
    )
  )
)