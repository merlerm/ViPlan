(define (problem unstack-blocks)
  (:domain blocksworld)

  (:objects 
    R G B Y O - block
    C1 C2 C3 C4 C5 - column
  )

  (:init
    (inColumn B C1) (clear B)
    (inColumn Y C2) 
    (inColumn G C2) (on G Y) (clear G)
    (inColumn R C3) 
    (inColumn O C3) (on O R) (clear O)

    (rightOf C2 C1) (leftOf C1 C2)
    (rightOf C3 C2) (leftOf C2 C3)
    (rightOf C4 C3) (leftOf C3 C4)
    (rightOf C5 C4) (leftOf C4 C5)
  )

  (:goal 
    (and 
      (inColumn R C1)
      (clear R)
      (inColumn O C2)
      (clear O)
      (inColumn B C3)
      (clear B)
      (inColumn Y C4)
      (clear Y)
      (inColumn G C5)
      (clear G)
    )
  )
)
