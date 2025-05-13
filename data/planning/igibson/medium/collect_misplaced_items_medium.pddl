(define (problem collect_misplaced_items_0)
    (:domain igibson)

    (:objects
        gym_shoe_1 - movable
        sock_1 sock_2 - movable
        table_1 table_2 - object
        cabinet_1 - container
    )
    
    (:init 
    )
    
    (:goal 
        (and 
            (ontop gym_shoe_1 table_2)
            (ontop sock_2 table_2)
        )
    )
)