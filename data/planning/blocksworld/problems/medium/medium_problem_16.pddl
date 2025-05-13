(define (problem medium_problem_16)
  (:domain blocksworld)
  
  (:objects 
    G P Y O R - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on R G)

    (clear P)
    (clear Y)
    (clear O)
    (clear R)

    (inColumn G C2)
    (inColumn P C4)
    (inColumn Y C3)
    (inColumn O C1)
    (inColumn R C2)

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
      (on R G)
      (on O P)

      (clear Y)
      (clear O)
      (clear R)

      (inColumn G C4)
      (inColumn P C3)
      (inColumn Y C2)
      (inColumn O C3)
      (inColumn R C4)
    )
  )
)