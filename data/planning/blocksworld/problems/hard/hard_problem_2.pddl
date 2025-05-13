(define (problem hard_problem_2)
  (:domain blocksworld)
  
  (:objects 
    O P B Y G R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on Y O)
    (on B P)
    (on R B)
    (on G Y)

    (clear G)
    (clear R)

    (inColumn O C1)
    (inColumn P C2)
    (inColumn B C2)
    (inColumn Y C1)
    (inColumn G C1)
    (inColumn R C2)

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
      (on G B)

      (clear P)
      (clear Y)
      (clear G)
      (clear R)

      (inColumn O C3)
      (inColumn P C2)
      (inColumn B C1)
      (inColumn Y C4)
      (inColumn G C1)
      (inColumn R C3)
    )
  )
)