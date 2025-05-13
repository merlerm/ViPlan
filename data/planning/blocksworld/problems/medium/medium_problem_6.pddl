(define (problem medium_problem_6)
  (:domain blocksworld)
  
  (:objects 
    R B Y P G - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G Y)

    (clear R)
    (clear B)
    (clear P)
    (clear G)

    (inColumn R C3)
    (inColumn B C5)
    (inColumn Y C4)
    (inColumn P C1)
    (inColumn G C4)

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
      (on Y B)

      (clear R)
      (clear Y)
      (clear P)
      (clear G)

      (inColumn R C2)
      (inColumn B C1)
      (inColumn Y C1)
      (inColumn P C5)
      (inColumn G C3)
    )
  )
)