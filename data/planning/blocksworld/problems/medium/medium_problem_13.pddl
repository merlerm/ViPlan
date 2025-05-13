(define (problem medium_problem_13)
  (:domain blocksworld)
  
  (:objects 
    B R G P Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on P B)

    (clear R)
    (clear G)
    (clear P)
    (clear Y)

    (inColumn B C3)
    (inColumn R C1)
    (inColumn G C5)
    (inColumn P C3)
    (inColumn Y C2)

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
      (on P B)

      (clear R)
      (clear G)
      (clear P)
      (clear Y)

      (inColumn B C5)
      (inColumn R C1)
      (inColumn G C4)
      (inColumn P C5)
      (inColumn Y C2)
    )
  )
)