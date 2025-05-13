(define (problem hard_problem_8)
  (:domain blocksworld)
  
  (:objects 
    G P O R Y B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O P)
    (on R O)
    (on B R)

    (clear G)
    (clear Y)
    (clear B)

    (inColumn G C4)
    (inColumn P C1)
    (inColumn O C1)
    (inColumn R C1)
    (inColumn Y C3)
    (inColumn B C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on Y G)
      (on O P)
      (on B Y)

      (clear O)
      (clear R)
      (clear B)

      (inColumn G C4)
      (inColumn P C3)
      (inColumn O C3)
      (inColumn R C2)
      (inColumn Y C4)
      (inColumn B C4)
    )
  )
)