(define (problem hard_problem_11)
  (:domain blocksworld)
  
  (:objects 
    P O Y R G B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O P)
    (on R Y)
    (on B G)

    (clear O)
    (clear R)
    (clear B)

    (inColumn P C1)
    (inColumn O C1)
    (inColumn Y C2)
    (inColumn R C2)
    (inColumn G C4)
    (inColumn B C4)

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
      (on R Y)
      (on B G)

      (clear O)
      (clear R)
      (clear B)

      (inColumn P C3)
      (inColumn O C1)
      (inColumn Y C2)
      (inColumn R C2)
      (inColumn G C3)
      (inColumn B C3)
    )
  )
)