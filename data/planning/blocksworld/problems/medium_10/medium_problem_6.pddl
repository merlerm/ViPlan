(define (problem medium_problem_6)
  (:domain blocksworld)
  
  (:objects 
    B O G Y R - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on Y B)

    (clear O)
    (clear G)
    (clear Y)
    (clear R)

    (inColumn B C3)
    (inColumn O C2)
    (inColumn G C4)
    (inColumn Y C3)
    (inColumn R C1)

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
      (on R O)
      (on Y G)

      (clear Y)
      (clear R)

      (inColumn B C3)
      (inColumn O C3)
      (inColumn G C4)
      (inColumn Y C4)
      (inColumn R C3)
    )
  )
)