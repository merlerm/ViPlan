(define (problem simple_problem_20)
  (:domain blocksworld)
  
  (:objects 
    G R O - block
    C1 C2 C3 C4 - column
  )
  
  (:init


    (clear G)
    (clear R)
    (clear O)

    (inColumn G C2)
    (inColumn R C1)
    (inColumn O C3)

    (rightOf C2 C1)
    (rightOf C3 C2)
    (rightOf C4 C3)

    (leftOf C1 C2)
    (leftOf C2 C3)
    (leftOf C3 C4)
  )
  (:goal
    (and
      (on O R)

      (clear G)
      (clear O)

      (inColumn G C4)
      (inColumn R C3)
      (inColumn O C3)
    )
  )
)