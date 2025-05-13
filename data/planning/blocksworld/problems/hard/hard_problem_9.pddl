(define (problem hard_problem_9)
  (:domain blocksworld)
  
  (:objects 
    G Y O R P B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R Y)
    (on P R)

    (clear G)
    (clear O)
    (clear P)
    (clear B)

    (inColumn G C4)
    (inColumn Y C1)
    (inColumn O C2)
    (inColumn R C1)
    (inColumn P C1)
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
      (on O G)
      (on B Y)
      (on P R)

      (clear O)
      (clear P)
      (clear B)

      (inColumn G C2)
      (inColumn Y C4)
      (inColumn O C2)
      (inColumn R C3)
      (inColumn P C3)
      (inColumn B C4)
    )
  )
)