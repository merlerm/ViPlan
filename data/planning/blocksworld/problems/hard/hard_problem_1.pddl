(define (problem hard_problem_1)
  (:domain blocksworld)
  
  (:objects 
    B G P O Y R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O B)
    (on Y G)

    (clear P)
    (clear O)
    (clear Y)
    (clear R)

    (inColumn B C2)
    (inColumn G C3)
    (inColumn P C1)
    (inColumn O C2)
    (inColumn Y C3)
    (inColumn R C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on O B)
      (on Y P)

      (clear G)
      (clear O)
      (clear Y)
      (clear R)

      (inColumn B C2)
      (inColumn G C4)
      (inColumn P C3)
      (inColumn O C2)
      (inColumn Y C3)
      (inColumn R C1)
    )
  )
)