(define (problem hard_problem_22)
  (:domain blocksworld)
  
  (:objects 
    B G O P R Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R B)
    (on P O)
    (on Y R)

    (clear G)
    (clear P)
    (clear Y)

    (inColumn B C1)
    (inColumn G C2)
    (inColumn O C3)
    (inColumn P C3)
    (inColumn R C1)
    (inColumn Y C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on G B)
      (on Y P)

      (clear G)
      (clear O)
      (clear R)
      (clear Y)

      (inColumn B C1)
      (inColumn G C1)
      (inColumn O C2)
      (inColumn P C4)
      (inColumn R C3)
      (inColumn Y C4)
    )
  )
)