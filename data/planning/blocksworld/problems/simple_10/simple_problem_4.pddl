(define (problem simple_problem_4)
  (:domain blocksworld)
  
  (:objects 
    P G Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y P)

    (clear G)
    (clear Y)

    (inColumn P C3)
    (inColumn G C2)
    (inColumn Y C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on G P)

      (clear G)
      (clear Y)

      (inColumn P C1)
      (inColumn G C1)
      (inColumn Y C3)
    )
  )
)