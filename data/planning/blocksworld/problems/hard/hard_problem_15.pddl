(define (problem hard_problem_15)
  (:domain blocksworld)
  
  (:objects 
    P R O G Y B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O P)
    (on B R)
    (on G O)

    (clear G)
    (clear Y)
    (clear B)

    (inColumn P C1)
    (inColumn R C3)
    (inColumn O C1)
    (inColumn G C1)
    (inColumn Y C4)
    (inColumn B C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on Y P)
      (on B R)
      (on G O)

      (clear G)
      (clear Y)
      (clear B)

      (inColumn P C2)
      (inColumn R C3)
      (inColumn O C4)
      (inColumn G C4)
      (inColumn Y C2)
      (inColumn B C3)
    )
  )
)