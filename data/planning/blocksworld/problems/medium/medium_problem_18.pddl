(define (problem medium_problem_18)
  (:domain blocksworld)
  
  (:objects 
    R B G P Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init


    (clear R)
    (clear B)
    (clear G)
    (clear P)
    (clear Y)

    (inColumn R C5)
    (inColumn B C4)
    (inColumn G C3)
    (inColumn P C2)
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
      (on Y B)

      (clear R)
      (clear G)
      (clear P)
      (clear Y)

      (inColumn R C3)
      (inColumn B C1)
      (inColumn G C2)
      (inColumn P C4)
      (inColumn Y C1)
    )
  )
)