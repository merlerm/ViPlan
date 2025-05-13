(define (problem hard_problem_0)
  (:domain blocksworld)
  
  (:objects 
    R O P G Y B - block
    C1 C2 C3 C4 - column
  )
  
  (:init

    (on O R)
    (on Y G)
    (on B Y)

    (clear O)
    (clear P)
    (clear B)

    (inColumn R C2)
    (inColumn O C2)
    (inColumn P C3)
    (inColumn G C1)
    (inColumn Y C1)
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
      (on G R)
      (on B Y)

      (clear O)
      (clear P)
      (clear G)
      (clear B)

      (inColumn R C1)
      (inColumn O C4)
      (inColumn P C2)
      (inColumn G C1)
      (inColumn Y C3)
      (inColumn B C3)
    )
  )
)