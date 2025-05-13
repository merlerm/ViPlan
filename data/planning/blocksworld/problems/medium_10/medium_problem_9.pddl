(define (problem medium_problem_9)
  (:domain blocksworld)
  
  (:objects 
    P Y O R B - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B Y)

    (clear P)
    (clear O)
    (clear R)
    (clear B)

    (inColumn P C4)
    (inColumn Y C1)
    (inColumn O C5)
    (inColumn R C2)
    (inColumn B C1)

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
      (on B P)
      (on R O)

      (clear Y)
      (clear R)
      (clear B)

      (inColumn P C2)
      (inColumn Y C3)
      (inColumn O C5)
      (inColumn R C5)
      (inColumn B C2)
    )
  )
)