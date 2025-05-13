(define (problem simple_problem_8)
  (:domain blocksworld)
  
  (:objects 
    B Y R - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear B)
    (clear Y)
    (clear R)

    (inColumn B C4)
    (inColumn Y C2)
    (inColumn R C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and

      (clear B)
      (clear Y)
      (clear R)

      (inColumn B C2)
      (inColumn Y C4)
      (inColumn R C1)
    )
  )
)