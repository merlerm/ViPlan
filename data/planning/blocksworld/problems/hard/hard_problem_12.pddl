(define (problem hard_problem_12)
  (:domain blocksworld)
  
  (:objects 
    B O G Y R P - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P B)
    (on R O)
    (on Y G)

    (clear Y)
    (clear R)
    (clear P)

    (inColumn B C4)
    (inColumn O C1)
    (inColumn G C2)
    (inColumn Y C2)
    (inColumn R C1)
    (inColumn P C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on O B)
      (on Y O)
      (on P G)

      (clear Y)
      (clear R)
      (clear P)

      (inColumn B C3)
      (inColumn O C3)
      (inColumn G C2)
      (inColumn Y C3)
      (inColumn R C4)
      (inColumn P C2)
    )
  )
)