(define (problem hard_problem_2)
  (:domain blocksworld)
  
  (:objects 
    Y G O R P B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on G Y)
    (on O G)
    (on B O)
    (on P R)

    (clear P)
    (clear B)

    (inColumn Y C4)
    (inColumn G C4)
    (inColumn O C4)
    (inColumn R C1)
    (inColumn P C1)
    (inColumn B C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on P O)
      (on B R)

      (clear Y)
      (clear G)
      (clear P)
      (clear B)

      (inColumn Y C2)
      (inColumn G C3)
      (inColumn O C4)
      (inColumn R C1)
      (inColumn P C4)
      (inColumn B C1)
    )
  )
)