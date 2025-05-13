(define (problem hard_problem_1)
  (:domain blocksworld)
  
  (:objects 
    Y P O B R G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P Y)
    (on R O)

    (clear P)
    (clear B)
    (clear R)
    (clear G)

    (inColumn Y C1)
    (inColumn P C1)
    (inColumn O C3)
    (inColumn B C4)
    (inColumn R C3)
    (inColumn G C2)

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
      (on O P)
      (on B O)
      (on G B)

      (clear R)
      (clear G)

      (inColumn Y C1)
      (inColumn P C1)
      (inColumn O C1)
      (inColumn B C1)
      (inColumn R C4)
      (inColumn G C1)
    )
  )
)