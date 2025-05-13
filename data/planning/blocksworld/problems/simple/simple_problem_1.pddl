(define (problem simple_problem_1)
  (:domain blocksworld)
  
  (:objects 
    O Y B - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear O)
    (clear Y)
    (clear B)

    (inColumn O C1)
    (inColumn Y C2)
    (inColumn B C3)

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
      (clear B)

      (inColumn O C3)
      (inColumn Y C1)
      (inColumn B C2)
    )
  )
)