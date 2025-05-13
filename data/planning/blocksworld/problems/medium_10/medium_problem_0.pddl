(define (problem medium_problem_0)
  (:domain blocksworld)
  
  (:objects 
    B G O P Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G B)
    (on P G)
    (on Y O)

    (clear P)
    (clear Y)

    (inColumn B C3)
    (inColumn G C3)
    (inColumn O C4)
    (inColumn P C3)
    (inColumn Y C4)

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
      (on O B)
      (on Y G)

      (clear O)
      (clear P)
      (clear Y)

      (inColumn B C2)
      (inColumn G C3)
      (inColumn O C2)
      (inColumn P C1)
      (inColumn Y C3)
    )
  )
)