(define (problem hard_problem_16)
  (:domain blocksworld)
  
  (:objects 
    Y O G R P B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P Y)
    (on R O)
    (on B P)

    (clear G)
    (clear R)
    (clear B)

    (inColumn Y C2)
    (inColumn O C4)
    (inColumn G C1)
    (inColumn R C4)
    (inColumn P C2)
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
      (on G O)

      (clear G)
      (clear R)
      (clear P)
      (clear B)

      (inColumn Y C4)
      (inColumn O C3)
      (inColumn G C3)
      (inColumn R C1)
      (inColumn P C4)
      (inColumn B C2)
    )
  )
)