(define (problem hard_problem_5)
  (:domain blocksworld)
  
  (:objects 
    P O G R B Y - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on B P)
    (on Y G)

    (clear O)
    (clear R)
    (clear B)
    (clear Y)

    (inColumn P C2)
    (inColumn O C1)
    (inColumn G C4)
    (inColumn R C3)
    (inColumn B C2)
    (inColumn Y C4)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on R O)
      (on B G)

      (clear P)
      (clear R)
      (clear B)
      (clear Y)

      (inColumn P C1)
      (inColumn O C2)
      (inColumn G C3)
      (inColumn R C2)
      (inColumn B C3)
      (inColumn Y C4)
    )
  )
)