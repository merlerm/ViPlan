(define (problem medium_problem_3)
  (:domain blocksworld)
  
  (:objects 
    G R Y P B - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init


    (clear G)
    (clear R)
    (clear Y)
    (clear P)
    (clear B)

    (inColumn G C3)
    (inColumn R C5)
    (inColumn Y C2)
    (inColumn P C1)
    (inColumn B C4)

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
      (on R G)

      (clear R)
      (clear Y)
      (clear P)
      (clear B)

      (inColumn G C4)
      (inColumn R C4)
      (inColumn Y C2)
      (inColumn P C3)
      (inColumn B C1)
    )
  )
)