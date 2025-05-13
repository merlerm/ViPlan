(define (problem hard_problem_9)
  (:domain blocksworld)
  
  (:objects 
    O P B Y G R - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on P O)
    (on Y P)

    (clear B)
    (clear Y)
    (clear G)
    (clear R)

    (inColumn O C3)
    (inColumn P C3)
    (inColumn B C2)
    (inColumn Y C3)
    (inColumn G C4)
    (inColumn R C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on P O)
      (on R B)

      (clear P)
      (clear Y)
      (clear G)
      (clear R)

      (inColumn O C1)
      (inColumn P C1)
      (inColumn B C3)
      (inColumn Y C2)
      (inColumn G C4)
      (inColumn R C3)
    )
  )
)