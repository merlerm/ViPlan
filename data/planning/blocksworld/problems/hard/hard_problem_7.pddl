(define (problem hard_problem_7)
  (:domain blocksworld)
  
  (:objects 
    R P Y B G O - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y P)
    (on O G)

    (clear R)
    (clear Y)
    (clear B)
    (clear O)

    (inColumn R C2)
    (inColumn P C1)
    (inColumn Y C1)
    (inColumn B C3)
    (inColumn G C4)
    (inColumn O C4)

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
      (on O B)

      (clear P)
      (clear Y)
      (clear G)
      (clear O)

      (inColumn R C1)
      (inColumn P C4)
      (inColumn Y C2)
      (inColumn B C3)
      (inColumn G C1)
      (inColumn O C3)
    )
  )
)