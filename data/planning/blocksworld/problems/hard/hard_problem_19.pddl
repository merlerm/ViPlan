(define (problem hard_problem_19)
  (:domain blocksworld)
  
  (:objects 
    Y R O B P G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P R)
    (on G O)

    (clear Y)
    (clear B)
    (clear P)
    (clear G)

    (inColumn Y C2)
    (inColumn R C4)
    (inColumn O C1)
    (inColumn B C3)
    (inColumn P C4)
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
      (on R Y)
      (on P O)
      (on G B)

      (clear R)
      (clear P)
      (clear G)

      (inColumn Y C3)
      (inColumn R C3)
      (inColumn O C1)
      (inColumn B C2)
      (inColumn P C1)
      (inColumn G C2)
    )
  )
)