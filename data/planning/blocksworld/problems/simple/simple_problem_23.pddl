(define (problem simple_problem_23)
  (:domain blocksworld)
  
  (:objects 
    B O Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O B)

    (clear O)
    (clear Y)

    (inColumn B C1)
    (inColumn O C1)
    (inColumn Y C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on Y B)

      (clear O)
      (clear Y)

      (inColumn B C3)
      (inColumn O C4)
      (inColumn Y C3)
    )
  )
)