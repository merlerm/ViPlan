(define (problem packing_food_for_work_0)
    (:domain igibson)

    (:objects
        carton_1 - container
        cabinet_1 - container
        snack_food_1 - movable
    )
    
    (:init 
        (open carton_1)
        (not (open cabinet_1))
        (inside snack_food_1 cabinet_1) 
    )
    
    (:goal 
        (and 
            (inside snack_food_1 carton_1) 
        )
    )
)