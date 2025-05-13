(define (problem simple_problem_21)
  (:domain blocksworld)
  
  (:objects 
    R Y O - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear R)
    (clear Y)
    (clear O)

    (inColumn R C4)
    (inColumn Y C2)
    (inColumn O C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and

      (clear R)
      (clear Y)
      (clear O)

      (inColumn R C2)
      (inColumn Y C3)
      (inColumn O C1)
    )
  )
)