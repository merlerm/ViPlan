(define (problem hard_problem_18)
  (:domain blocksworld)
  
  (:objects 
    O P R Y B G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y P)
    (on B R)
    (on G B)

    (clear O)
    (clear Y)
    (clear G)

    (inColumn O C3)
    (inColumn P C2)
    (inColumn R C1)
    (inColumn Y C2)
    (inColumn B C1)
    (inColumn G C1)

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
      (on G B)

      (clear P)
      (clear Y)
      (clear G)

      (inColumn O C3)
      (inColumn P C3)
      (inColumn R C2)
      (inColumn Y C1)
      (inColumn B C2)
      (inColumn G C2)
    )
  )
)