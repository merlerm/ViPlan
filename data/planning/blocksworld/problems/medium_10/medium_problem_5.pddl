(define (problem medium_problem_5)
  (:domain blocksworld)
  
  (:objects 
    G Y R B P - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on R G)

    (clear Y)
    (clear R)
    (clear B)
    (clear P)

    (inColumn G C2)
    (inColumn Y C3)
    (inColumn R C2)
    (inColumn B C5)
    (inColumn P C4)

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
      (on R Y)

      (clear G)
      (clear R)
      (clear B)
      (clear P)

      (inColumn G C2)
      (inColumn Y C4)
      (inColumn R C4)
      (inColumn B C3)
      (inColumn P C1)
    )
  )
)