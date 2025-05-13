(define (problem medium_problem_4)
  (:domain blocksworld)
  
  (:objects 
    P G Y B R - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B G)
    (on R Y)

    (clear P)
    (clear B)
    (clear R)

    (inColumn P C5)
    (inColumn G C4)
    (inColumn Y C3)
    (inColumn B C4)
    (inColumn R C3)

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
      (on R P)

      (clear G)
      (clear Y)
      (clear B)
      (clear R)

      (inColumn P C2)
      (inColumn G C1)
      (inColumn Y C5)
      (inColumn B C4)
      (inColumn R C2)
    )
  )
)