(define (problem simple_problem_0)
  (:domain blocksworld)
  
  (:objects 
    B R G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R B)

    (clear R)
    (clear G)

    (inColumn B C3)
    (inColumn R C3)
    (inColumn G C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and

      (clear B)
      (clear R)
      (clear G)

      (inColumn B C2)
      (inColumn R C4)
      (inColumn G C3)
    )
  )
)