(define (problem simple_problem_3)
  (:domain blocksworld)
  
  (:objects 
    O Y R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R Y)

    (clear O)
    (clear R)

    (inColumn O C4)
    (inColumn Y C3)
    (inColumn R C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and

      (clear O)
      (clear Y)
      (clear R)

      (inColumn O C2)
      (inColumn Y C1)
      (inColumn R C4)
    )
  )
)