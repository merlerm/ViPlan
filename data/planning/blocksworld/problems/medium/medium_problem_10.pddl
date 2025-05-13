(define (problem medium_problem_10)
  (:domain blocksworld)
  
  (:objects 
    G R O Y P - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on O G)
    (on P R)

    (clear O)
    (clear Y)
    (clear P)

    (inColumn G C2)
    (inColumn R C4)
    (inColumn O C2)
    (inColumn Y C5)
    (inColumn P C4)

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
      (on P O)

      (clear G)
      (clear R)
      (clear Y)
      (clear P)

      (inColumn G C1)
      (inColumn R C3)
      (inColumn O C4)
      (inColumn Y C5)
      (inColumn P C4)
    )
  )
)