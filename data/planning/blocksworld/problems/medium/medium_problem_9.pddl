(define (problem medium_problem_9)
  (:domain blocksworld)
  
  (:objects 
    B P R G Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on P B)

    (clear P)
    (clear R)
    (clear G)
    (clear Y)

    (inColumn B C2)
    (inColumn P C2)
    (inColumn R C5)
    (inColumn G C4)
    (inColumn Y C1)

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
      (on Y G)

      (clear B)
      (clear P)
      (clear R)
      (clear Y)

      (inColumn B C3)
      (inColumn P C4)
      (inColumn R C5)
      (inColumn G C2)
      (inColumn Y C2)
    )
  )
)