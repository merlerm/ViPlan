(define (problem hard_problem_8)
  (:domain blocksworld)
  
  (:objects 
    O Y B R P G - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y O)
    (on G Y)
    (on R B)
    (on P R)

    (clear P)
    (clear G)

    (inColumn O C1)
    (inColumn Y C1)
    (inColumn B C3)
    (inColumn R C3)
    (inColumn P C3)
    (inColumn G C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on Y O)
      (on P Y)
      (on R B)
      (on G R)

      (clear P)
      (clear G)

      (inColumn O C2)
      (inColumn Y C2)
      (inColumn B C3)
      (inColumn R C3)
      (inColumn P C2)
      (inColumn G C3)
    )
  )
)