(define (problem hard_problem_3)
  (:domain blocksworld)
  
  (:objects 
    R O Y B G P - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y R)
    (on P O)

    (clear Y)
    (clear B)
    (clear G)
    (clear P)

    (inColumn R C1)
    (inColumn O C3)
    (inColumn Y C1)
    (inColumn B C4)
    (inColumn G C2)
    (inColumn P C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on G R)
      (on P Y)

      (clear O)
      (clear B)
      (clear G)
      (clear P)

      (inColumn R C1)
      (inColumn O C2)
      (inColumn Y C3)
      (inColumn B C4)
      (inColumn G C1)
      (inColumn P C3)
    )
  )
)