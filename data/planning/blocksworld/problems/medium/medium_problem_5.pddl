(define (problem medium_problem_5)
  (:domain blocksworld)
  
  (:objects 
    O B R Y G - block
    C1 C2 C3 C4 C5 - column
  )
  
  (:init

    (on B O)
    (on G R)

    (clear B)
    (clear Y)
    (clear G)

    (inColumn O C5)
    (inColumn B C5)
    (inColumn R C3)
    (inColumn Y C2)
    (inColumn G C3)

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
      (on R O)

      (clear B)
      (clear R)
      (clear Y)
      (clear G)

      (inColumn O C2)
      (inColumn B C1)
      (inColumn R C2)
      (inColumn Y C3)
      (inColumn G C5)
    )
  )
)