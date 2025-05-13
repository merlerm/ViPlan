(define (problem hard_problem_3)
  (:domain blocksworld)
  
  (:objects 
    Y G P R O B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on R Y)
    (on P G)
    (on O P)

    (clear R)
    (clear O)
    (clear B)

    (inColumn Y C2)
    (inColumn G C3)
    (inColumn P C3)
    (inColumn R C2)
    (inColumn O C3)
    (inColumn B C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on O Y)
      (on R P)

      (clear G)
      (clear R)
      (clear O)
      (clear B)

      (inColumn Y C3)
      (inColumn G C1)
      (inColumn P C2)
      (inColumn R C2)
      (inColumn O C3)
      (inColumn B C4)
    )
  )
)