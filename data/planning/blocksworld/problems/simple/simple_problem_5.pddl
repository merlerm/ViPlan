(define (problem simple_problem_5)
  (:domain blocksworld)
  
  (:objects 
    B R G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on G B)

    (clear R)
    (clear G)

    (inColumn B C4)
    (inColumn R C2)
    (inColumn G C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on R B)

      (clear R)
      (clear G)

      (inColumn B C3)
      (inColumn R C3)
      (inColumn G C1)
    )
  )
)