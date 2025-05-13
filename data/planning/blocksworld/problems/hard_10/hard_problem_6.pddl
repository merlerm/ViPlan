(define (problem hard_problem_6)
  (:domain blocksworld)
  
  (:objects 
    R G B O Y P - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on B G)
    (on O B)

    (clear R)
    (clear O)
    (clear Y)
    (clear P)

    (inColumn R C3)
    (inColumn G C4)
    (inColumn B C4)
    (inColumn O C4)
    (inColumn Y C2)
    (inColumn P C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on Y R)
      (on B G)

      (clear B)
      (clear O)
      (clear Y)
      (clear P)

      (inColumn R C3)
      (inColumn G C1)
      (inColumn B C1)
      (inColumn O C4)
      (inColumn Y C3)
      (inColumn P C2)
    )
  )
)