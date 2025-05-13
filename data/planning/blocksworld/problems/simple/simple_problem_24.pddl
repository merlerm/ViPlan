(define (problem simple_problem_24)
  (:domain blocksworld)
  
  (:objects 
    Y O G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O Y)

    (clear O)
    (clear G)

    (inColumn Y C2)
    (inColumn O C2)
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
      (on G Y)

      (clear O)
      (clear G)

      (inColumn Y C2)
      (inColumn O C3)
      (inColumn G C2)
    )
  )
)