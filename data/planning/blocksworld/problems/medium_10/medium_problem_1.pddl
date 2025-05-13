(define (problem medium_problem_1)
  (:domain blocksworld)
  
  (:objects 
    Y P O B R - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init


    (clear Y)
    (clear P)
    (clear O)
    (clear B)
    (clear R)

    (inColumn Y C5)
    (inColumn P C3)
    (inColumn O C1)
    (inColumn B C4)
    (inColumn R C2)

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
      (on O Y)

      (clear P)
      (clear O)
      (clear B)
      (clear R)

      (inColumn Y C1)
      (inColumn P C4)
      (inColumn O C1)
      (inColumn B C2)
      (inColumn R C3)
    )
  )
)