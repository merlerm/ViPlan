(define (problem hard_problem_14)
  (:domain blocksworld)
  
  (:objects 
    P B O G Y R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on B P)
    (on R B)
    (on G O)

    (clear G)
    (clear Y)
    (clear R)

    (inColumn P C4)
    (inColumn B C4)
    (inColumn O C2)
    (inColumn G C2)
    (inColumn Y C1)
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
      (on Y P)
      (on R O)

      (clear B)
      (clear G)
      (clear Y)
      (clear R)

      (inColumn P C2)
      (inColumn B C1)
      (inColumn O C3)
      (inColumn G C4)
      (inColumn Y C2)
      (inColumn R C3)
    )
  )
)