(define (problem simple_problem_18)
  (:domain blocksworld)
  
  (:objects 
    G B Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on B G)

    (clear B)
    (clear Y)

    (inColumn G C3)
    (inColumn B C3)
    (inColumn Y C2)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on Y B)

      (clear G)
      (clear Y)

      (inColumn G C2)
      (inColumn B C3)
      (inColumn Y C3)
    )
  )
)