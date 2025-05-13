(define (problem medium_problem_11)
  (:domain blocksworld)
  
  (:objects 
    P B O Y G - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on Y B)
    (on G Y)

    (clear P)
    (clear O)
    (clear G)

    (inColumn P C1)
    (inColumn B C5)
    (inColumn O C4)
    (inColumn Y C5)
    (inColumn G C5)

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

      (clear P)
      (clear B)
      (clear O)
      (clear Y)
      (clear G)

      (inColumn P C1)
      (inColumn B C4)
      (inColumn O C2)
      (inColumn Y C3)
      (inColumn G C5)
    )
  )
)