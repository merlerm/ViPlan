(define (problem medium_problem_20)
  (:domain blocksworld)
  
  (:objects 
    B O R P Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on R O)

    (clear B)
    (clear R)
    (clear P)
    (clear Y)

    (inColumn B C2)
    (inColumn O C1)
    (inColumn R C1)
    (inColumn P C5)
    (inColumn Y C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)
    (rightOf C5 C4)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
    (leftOf C4 C5)
  )
  (:goal
    (and
      (on O B)

      (clear O)
      (clear R)
      (clear P)
      (clear Y)

      (inColumn B C2)
      (inColumn O C2)
      (inColumn R C5)
      (inColumn P C3)
      (inColumn Y C1)
    )
  )
)