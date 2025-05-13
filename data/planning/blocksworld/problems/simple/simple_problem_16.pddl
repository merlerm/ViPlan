(define (problem simple_problem_16)
  (:domain blocksworld)
  
  (:objects 
    Y G P - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear Y)
    (clear G)
    (clear P)

    (inColumn Y C2)
    (inColumn G C3)
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

      (clear Y)
      (clear G)
      (clear P)

      (inColumn Y C4)
      (inColumn G C1)
      (inColumn P C2)
    )
  )
)