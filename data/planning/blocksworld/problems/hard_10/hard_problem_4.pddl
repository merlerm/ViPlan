(define (problem hard_problem_4)
  (:domain blocksworld)
  
  (:objects 
    R O P B G Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on G P)
    (on Y G)

    (clear R)
    (clear O)
    (clear B)
    (clear Y)

    (inColumn R C4)
    (inColumn O C3)
    (inColumn P C1)
    (inColumn B C2)
    (inColumn G C1)
    (inColumn Y C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on B R)
      (on Y G)

      (clear O)
      (clear P)
      (clear B)
      (clear Y)

      (inColumn R C2)
      (inColumn O C3)
      (inColumn P C4)
      (inColumn B C2)
      (inColumn G C1)
      (inColumn Y C1)
    )
  )
)