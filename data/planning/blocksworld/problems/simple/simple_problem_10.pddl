(define (problem simple_problem_10)
  (:domain blocksworld)
  
  (:objects 
    B G P - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear B)
    (clear G)
    (clear P)

    (inColumn B C3)
    (inColumn G C2)
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
      (on G B)

      (clear G)
      (clear P)

      (inColumn B C1)
      (inColumn G C1)
      (inColumn P C3)
    )
  )
)