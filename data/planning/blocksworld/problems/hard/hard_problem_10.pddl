(define (problem hard_problem_10)
  (:domain blocksworld)
  
  (:objects 
    P O G Y B R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y P)
    (on G O)
    (on B G)
    (on R Y)

    (clear B)
    (clear R)

    (inColumn P C4)
    (inColumn O C3)
    (inColumn G C3)
    (inColumn Y C4)
    (inColumn B C3)
    (inColumn R C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on G P)
      (on R O)

      (clear G)
      (clear Y)
      (clear B)
      (clear R)

      (inColumn P C3)
      (inColumn O C2)
      (inColumn G C3)
      (inColumn Y C1)
      (inColumn B C4)
      (inColumn R C2)
    )
  )
)