(define (problem medium_problem_12)
  (:domain blocksworld)
  
  (:objects 
    G O P R Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on Y O)
    (on R P)

    (clear G)
    (clear R)
    (clear Y)

    (inColumn G C2)
    (inColumn O C5)
    (inColumn P C1)
    (inColumn R C1)
    (inColumn Y C5)

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
      (on P G)
      (on R O)

      (clear P)
      (clear R)
      (clear Y)

      (inColumn G C1)
      (inColumn O C3)
      (inColumn P C1)
      (inColumn R C3)
      (inColumn Y C4)
    )
  )
)