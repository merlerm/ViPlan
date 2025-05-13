(define (problem medium_problem_23)
  (:domain blocksworld)
  
  (:objects 
    B Y G R P - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on G Y)
    (on R G)

    (clear B)
    (clear R)
    (clear P)

    (inColumn B C3)
    (inColumn Y C2)
    (inColumn G C2)
    (inColumn R C2)
    (inColumn P C1)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)
    (rightOf C5 C4)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
    (leftOf C4 C5)
  )
  (:goal
    (and
      (on P B)

      (clear Y)
      (clear G)
      (clear R)
      (clear P)

      (inColumn B C1)
      (inColumn Y C3)
      (inColumn G C2)
      (inColumn R C5)
      (inColumn P C1)
    )
  )
)