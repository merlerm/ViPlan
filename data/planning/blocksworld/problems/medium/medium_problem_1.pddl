(define (problem medium_problem_1)
  (:domain blocksworld)
  
  (:objects 
    B G P Y R - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on R B)

    (clear G)
    (clear P)
    (clear Y)
    (clear R)

    (inColumn B C3)
    (inColumn G C4)
    (inColumn P C2)
    (inColumn Y C1)
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
      (on Y B)

      (clear G)
      (clear P)
      (clear Y)
      (clear R)

      (inColumn B C1)
      (inColumn G C3)
      (inColumn P C5)
      (inColumn Y C1)
      (inColumn R C4)
    )
  )
)