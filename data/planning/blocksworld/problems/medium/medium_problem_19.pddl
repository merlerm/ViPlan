(define (problem medium_problem_19)
  (:domain blocksworld)
  
  (:objects 
    G P B O Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on Y P)
    (on O B)

    (clear G)
    (clear O)
    (clear Y)

    (inColumn G C3)
    (inColumn P C4)
    (inColumn B C5)
    (inColumn O C5)
    (inColumn Y C4)

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
      (on Y O)

      (clear G)
      (clear P)
      (clear B)
      (clear Y)

      (inColumn G C5)
      (inColumn P C3)
      (inColumn B C4)
      (inColumn O C2)
      (inColumn Y C2)
    )
  )
)