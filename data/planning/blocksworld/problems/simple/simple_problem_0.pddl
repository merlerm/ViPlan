(define (problem simple_problem_0)
  (:domain blocksworld)
  
  (:objects 
    Y P R - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear Y)
    (clear P)
    (clear R)

    (inColumn Y C2)
    (inColumn P C1)
    (inColumn R C4)

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
      (clear P)
      (clear R)

      (inColumn Y C3)
      (inColumn P C4)
      (inColumn R C1)
    )
  )
)