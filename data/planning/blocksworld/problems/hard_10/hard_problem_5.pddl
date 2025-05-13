(define (problem hard_problem_5)
  (:domain blocksworld)
  
  (:objects 
    P R G Y B O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y P)
    (on B R)

    (clear G)
    (clear Y)
    (clear B)
    (clear O)

    (inColumn P C1)
    (inColumn R C4)
    (inColumn G C2)
    (inColumn Y C1)
    (inColumn B C4)
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
      (on R P)
      (on Y G)
      (on O Y)

      (clear R)
      (clear B)
      (clear O)

      (inColumn P C1)
      (inColumn R C1)
      (inColumn G C4)
      (inColumn Y C4)
      (inColumn B C3)
      (inColumn O C4)
    )
  )
)