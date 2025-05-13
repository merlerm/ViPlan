(define (problem simple_problem_8)
  (:domain blocksworld)
  
  (:objects 
    Y R G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on G Y)

    (clear R)
    (clear G)

    (inColumn Y C3)
    (inColumn R C4)
    (inColumn G C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and

      (clear Y)
      (clear R)
      (clear G)

      (inColumn Y C2)
      (inColumn R C3)
      (inColumn G C4)
    )
  )
)