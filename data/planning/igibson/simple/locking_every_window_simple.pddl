(define (problem locking_every_window_0)
    (:domain igibson)

    (:objects
     	window_1 window_2 window_3 - container
    )
    
    (:init 
        (open window_1) 
        (open window_2) 
        (open window_3) 
    )
    
    (:goal 
        (and 
            (not (open window_1)) 
            (not (open window_2)) 
            (not (open window_3)) 
        )
    )
)