(define (problem stack-all-blocks)
  (:domain blocksworld)

  (:objects 
    R G B Y O - block
    C1 C2 C3 C4 C5 - column
  )

  (:init
    (inColumn R C1) (clear R)
    (inColumn G C2) (clear G)
    (inColumn B C3) (clear B)
    (inColumn Y C4) (clear Y)
    (inColumn O C5) (clear O)

    (rightOf C2 C1) (leftOf C1 C2)
    (rightOf C3 C2) (leftOf C2 C3)
    (rightOf C4 C3) (leftOf C3 C4)
    (rightOf C5 C4) (leftOf C4 C5)
  )

  (:goal 
    (and 
      (on O Y)
      (on Y  B)
      (on B G)
      (on G R)
      (inColumn R C1)
      (clear O)
    )
  )
)
