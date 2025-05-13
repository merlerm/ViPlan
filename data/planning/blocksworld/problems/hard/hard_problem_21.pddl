(define (problem hard_problem_21)
  (:domain blocksworld)
  
  (:objects 
    Y P G R O B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P Y)
    (on B O)

    (clear P)
    (clear G)
    (clear R)
    (clear B)

    (inColumn Y C3)
    (inColumn P C3)
    (inColumn G C4)
    (inColumn R C1)
    (inColumn O C2)
    (inColumn B C2)

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
      (on B P)
      (on O G)

      (clear R)
      (clear O)
      (clear B)

      (inColumn Y C3)
      (inColumn P C3)
      (inColumn G C1)
      (inColumn R C4)
      (inColumn O C1)
      (inColumn B C3)
    )
  )
)