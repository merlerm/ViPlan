(define (problem hard_problem_4)
  (:domain blocksworld)
  
  (:objects 
    R Y O B G P - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P R)
    (on B O)

    (clear Y)
    (clear B)
    (clear G)
    (clear P)

    (inColumn R C1)
    (inColumn Y C3)
    (inColumn O C2)
    (inColumn B C2)
    (inColumn G C4)
    (inColumn P C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B R)
      (on P B)

      (clear Y)
      (clear O)
      (clear G)
      (clear P)

      (inColumn R C4)
      (inColumn Y C2)
      (inColumn O C1)
      (inColumn B C4)
      (inColumn G C3)
      (inColumn P C4)
    )
  )
)