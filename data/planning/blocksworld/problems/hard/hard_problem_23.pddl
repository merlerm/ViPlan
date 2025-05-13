(define (problem hard_problem_23)
  (:domain blocksworld)
  
  (:objects 
    P B R O G Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R P)
    (on O B)
    (on Y G)

    (clear R)
    (clear O)
    (clear Y)

    (inColumn P C3)
    (inColumn B C4)
    (inColumn R C3)
    (inColumn O C4)
    (inColumn G C1)
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
      (on B P)
      (on R B)
      (on G O)

      (clear R)
      (clear G)
      (clear Y)

      (inColumn P C2)
      (inColumn B C2)
      (inColumn R C2)
      (inColumn O C4)
      (inColumn G C4)
      (inColumn Y C3)
    )
  )
)