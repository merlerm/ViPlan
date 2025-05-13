(define (problem simple_problem_6)
  (:domain blocksworld)
  
  (:objects 
    Y O B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O Y)

    (clear O)
    (clear B)

    (inColumn Y C3)
    (inColumn O C3)
    (inColumn B C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B O)

      (clear Y)
      (clear B)

      (inColumn Y C1)
      (inColumn O C3)
      (inColumn B C3)
    )
  )
)