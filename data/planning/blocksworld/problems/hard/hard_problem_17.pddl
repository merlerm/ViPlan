(define (problem hard_problem_17)
  (:domain blocksworld)
  
  (:objects 
    G P B R O Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R G)
    (on Y P)

    (clear B)
    (clear R)
    (clear O)
    (clear Y)

    (inColumn G C4)
    (inColumn P C3)
    (inColumn B C1)
    (inColumn R C4)
    (inColumn O C2)
    (inColumn Y C3)

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

      (clear P)
      (clear B)
      (clear O)
      (clear Y)

      (inColumn G C2)
      (inColumn P C3)
      (inColumn B C1)
      (inColumn R C2)
      (inColumn O C2)
      (inColumn Y C4)
    )
  )
)