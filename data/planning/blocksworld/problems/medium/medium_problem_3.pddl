(define (problem medium_problem_3)
  (:domain blocksworld)
  
  (:objects 
    R Y P G B - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G Y)

    (clear R)
    (clear P)
    (clear G)
    (clear B)

    (inColumn R C2)
    (inColumn Y C3)
    (inColumn P C4)
    (inColumn G C3)
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
      (on B G)

      (clear R)
      (clear Y)
      (clear P)
      (clear B)

      (inColumn R C1)
      (inColumn Y C3)
      (inColumn P C2)
      (inColumn G C4)
      (inColumn B C4)
    )
  )
)