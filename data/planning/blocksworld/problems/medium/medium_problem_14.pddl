(define (problem medium_problem_14)
  (:domain blocksworld)
  
  (:objects 
    P R G B Y - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on Y G)

    (clear P)
    (clear R)
    (clear B)
    (clear Y)

    (inColumn P C5)
    (inColumn R C4)
    (inColumn G C2)
    (inColumn B C3)
    (inColumn Y C2)

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
      (on Y B)

      (clear R)
      (clear G)
      (clear Y)

      (inColumn P C2)
      (inColumn R C1)
      (inColumn G C4)
      (inColumn B C2)
      (inColumn Y C2)
    )
  )
)