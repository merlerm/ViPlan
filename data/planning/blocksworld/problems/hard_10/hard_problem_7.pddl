(define (problem hard_problem_7)
  (:domain blocksworld)
  
  (:objects 
    R Y P G O B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P R)
    (on O Y)
    (on G P)

    (clear G)
    (clear O)
    (clear B)

    (inColumn R C1)
    (inColumn Y C4)
    (inColumn P C1)
    (inColumn G C1)
    (inColumn O C4)
    (inColumn B C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on P Y)
      (on B G)

      (clear R)
      (clear P)
      (clear O)
      (clear B)

      (inColumn R C2)
      (inColumn Y C3)
      (inColumn P C3)
      (inColumn G C1)
      (inColumn O C4)
      (inColumn B C1)
    )
  )
)