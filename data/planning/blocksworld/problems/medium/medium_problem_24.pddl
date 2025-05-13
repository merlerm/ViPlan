(define (problem medium_problem_24)
  (:domain blocksworld)
  
  (:objects 
    O Y B P R - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B Y)

    (clear O)
    (clear B)
    (clear P)
    (clear R)

    (inColumn O C4)
    (inColumn Y C1)
    (inColumn B C1)
    (inColumn P C2)
    (inColumn R C5)

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
      (on Y O)

      (clear Y)
      (clear B)
      (clear P)
      (clear R)

      (inColumn O C5)
      (inColumn Y C5)
      (inColumn B C4)
      (inColumn P C3)
      (inColumn R C2)
    )
  )
)