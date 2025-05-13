(define (problem medium_problem_7)
  (:domain blocksworld)
  
  (:objects 
    G P R O Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on R G)
    (on Y O)

    (clear P)
    (clear R)
    (clear Y)

    (inColumn G C1)
    (inColumn P C3)
    (inColumn R C1)
    (inColumn O C5)
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
      (on O R)

      (clear G)
      (clear P)
      (clear O)
      (clear Y)

      (inColumn G C2)
      (inColumn P C4)
      (inColumn R C1)
      (inColumn O C1)
      (inColumn Y C5)
    )
  )
)