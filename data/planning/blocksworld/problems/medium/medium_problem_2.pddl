(define (problem medium_problem_2)
  (:domain blocksworld)
  
  (:objects 
    R Y G O B - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G R)
    (on O G)

    (clear Y)
    (clear O)
    (clear B)

    (inColumn R C5)
    (inColumn Y C2)
    (inColumn G C5)
    (inColumn O C5)
    (inColumn B C4)

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
      (on G Y)
      (on O G)

      (clear R)
      (clear O)
      (clear B)

      (inColumn R C5)
      (inColumn Y C1)
      (inColumn G C1)
      (inColumn O C1)
      (inColumn B C3)
    )
  )
)