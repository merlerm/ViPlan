(define (problem medium_problem_8)
  (:domain blocksworld)
  
  (:objects 
    Y R P B O - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B Y)
    (on P R)
    (on O B)

    (clear P)
    (clear O)

    (inColumn Y C1)
    (inColumn R C3)
    (inColumn P C3)
    (inColumn B C1)
    (inColumn O C1)

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
      (on B P)
      (on O B)

      (clear Y)
      (clear R)
      (clear O)

      (inColumn Y C4)
      (inColumn R C2)
      (inColumn P C3)
      (inColumn B C3)
      (inColumn O C3)
    )
  )
)