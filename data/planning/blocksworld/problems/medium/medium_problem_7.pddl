(define (problem medium_problem_7)
  (:domain blocksworld)
  
  (:objects 
    R G B P Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G R)
    (on B G)
    (on P B)

    (clear P)
    (clear Y)

    (inColumn R C4)
    (inColumn G C4)
    (inColumn B C4)
    (inColumn P C4)
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
      (on B R)
      (on Y B)

      (clear G)
      (clear P)
      (clear Y)

      (inColumn R C1)
      (inColumn G C2)
      (inColumn B C1)
      (inColumn P C3)
      (inColumn Y C1)
    )
  )
)