(define (problem hard_problem_13)
  (:domain blocksworld)
  
  (:objects 
    G Y B P R O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O G)
    (on B Y)

    (clear B)
    (clear P)
    (clear R)
    (clear O)

    (inColumn G C2)
    (inColumn Y C3)
    (inColumn B C3)
    (inColumn P C1)
    (inColumn R C4)
    (inColumn O C2)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on R G)
      (on O R)

      (clear Y)
      (clear B)
      (clear P)
      (clear O)

      (inColumn G C2)
      (inColumn Y C1)
      (inColumn B C3)
      (inColumn P C4)
      (inColumn R C2)
      (inColumn O C2)
    )
  )
)