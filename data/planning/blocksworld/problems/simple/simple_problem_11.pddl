(define (problem simple_problem_11)
  (:domain blocksworld)
  
  (:objects 
    Y R O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R Y)

    (clear R)
    (clear O)

    (inColumn Y C2)
    (inColumn R C2)
    (inColumn O C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on O Y)

      (clear R)
      (clear O)

      (inColumn Y C2)
      (inColumn R C4)
      (inColumn O C2)
    )
  )
)