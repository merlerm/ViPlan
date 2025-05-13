(define (problem hard_problem_20)
  (:domain blocksworld)
  
  (:objects 
    G P R O Y B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P G)
    (on B R)
    (on Y O)

    (clear P)
    (clear Y)
    (clear B)

    (inColumn G C3)
    (inColumn P C3)
    (inColumn R C2)
    (inColumn O C1)
    (inColumn Y C1)
    (inColumn B C2)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B P)
      (on O R)

      (clear G)
      (clear O)
      (clear Y)
      (clear B)

      (inColumn G C2)
      (inColumn P C3)
      (inColumn R C4)
      (inColumn O C4)
      (inColumn Y C1)
      (inColumn B C3)
    )
  )
)