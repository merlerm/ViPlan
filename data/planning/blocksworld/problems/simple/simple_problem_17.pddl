(define (problem simple_problem_17)
  (:domain blocksworld)
  
  (:objects 
    G R B - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear G)
    (clear R)
    (clear B)

    (inColumn G C3)
    (inColumn R C4)
    (inColumn B C2)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B R)

      (clear G)
      (clear B)

      (inColumn G C3)
      (inColumn R C2)
      (inColumn B C2)
    )
  )
)