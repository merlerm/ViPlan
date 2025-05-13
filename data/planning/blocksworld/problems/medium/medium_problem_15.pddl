(define (problem medium_problem_15)
  (:domain blocksworld)
  
  (:objects 
    R O P Y B - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on Y P)

    (clear R)
    (clear O)
    (clear Y)
    (clear B)

    (inColumn R C3)
    (inColumn O C4)
    (inColumn P C1)
    (inColumn Y C1)
    (inColumn B C2)

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
      (on B P)

      (clear R)
      (clear Y)
      (clear B)

      (inColumn R C5)
      (inColumn O C2)
      (inColumn P C2)
      (inColumn Y C3)
      (inColumn B C2)
    )
  )
)