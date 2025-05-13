(define (problem locking_every_door_0)
    (:domain igibson)

    (:objects
     	door_1 door_2 - container
    )
    
    (:init 
        (open door_1) 
        (open door_2) 
    )
    
    (:goal 
        (and 
            (not (open door_1)) 
            (not (open door_2))
        )
    )
)
